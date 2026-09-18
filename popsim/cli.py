"""popsim command line.

Thin. Every subcommand loads one config, snapshots it into ``runs/``, does its
work, and writes ``metrics.json``. A run that produced a number you cannot trace
back to a config snapshot is not a run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ConfigError, load_config, snapshot_run

log = logging.getLogger("popsim")


def _parse_override(s: str):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got {s!r}")
    k, v = s.split("=", 1)
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, v


def cmd_noop(args) -> int:
    """Gate 0, first clause: a no-op run writes a config snapshot to runs/."""
    cfg = load_config(args.config, overrides=dict(args.set or []))
    run_dir = snapshot_run(cfg, note="no-op run (Gate 0)")
    (run_dir / "metrics.json").write_text(json.dumps({"noop": True, "gate": 0}, indent=2))
    print(f"run dir: {run_dir}")
    for f in sorted(p.name for p in run_dir.iterdir()):
        print(f"  {f}")
    return 0


def cmd_validate(args) -> int:
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{cfg.source_path.name}: valid")
    print(f"  bed          {cfg['bed.dataset']} waves {cfg['bed.waves']} (excluding {cfg['bed.exclude_waves']})")
    print(f"  partition    {cfg['partition.axes']} min_cell={cfg['partition.min_cell']} K={cfg['partition.k_target']}")
    print(f"  split        {cfg['item_split.n_targets']} targets / {cfg['item_split.n_anchors']} anchors")
    print(f"  ensemble     {cfg['elicitation.active_profile']} -> {cfg.n_ensemble} calls per (cluster, item)")
    print(f"  budget       ${cfg['llm.budget_cap_usd']}, max {cfg['llm.max_calls_per_run']} calls/run")
    return 0


def cmd_quota(args) -> int:
    from .llm.client import LLMClient
    from .llm.env import load_dotenv

    load_dotenv()
    cfg = load_config(args.config, overrides=dict(args.set or []), validate=False)
    client = LLMClient.from_config(cfg)

    for name in args.clear or []:
        try:
            q = client.quota[name]
        except KeyError:
            print(f"no ledger entry for {name!r}; known: "
                  f"{sorted(client.quota.providers)}", file=sys.stderr)
            return 2
        was = q.exhausted_until_day
        if q.clear_exhaustion():
            client.quota.flush()
            print(f"cleared {name}: was burned until {was}")
        else:
            print(f"{name} was not burned; nothing to clear")

    print(json.dumps(client.report(), indent=2))
    return 0


def cmd_gate1(args) -> int:
    """Layer 0 gate: harmonized marginals vs published toplines (checklist 1.5)."""
    import yaml

    from .evalx.gate1 import run_layer0

    cfg = load_config(args.config, overrides=dict(args.set or []))
    run_dir = snapshot_run(cfg, note="Layer 0 gate")
    pool = sorted(yaml.safe_load(
        (cfg.repo_root / cfg["paths.codebooks"] / "gss_items.yaml").read_text()
    )["items"])
    report = run_layer0(
        dta_path=cfg.bed_file,
        toplines_path=cfg.repo_root / cfg["paths.codebooks"] / "gss2022_published_toplines.yaml",
        margins_path=cfg.data_path(cfg["aggregation.margins"]),
        item_pool=pool,
        tolerance_pp=cfg["evaluation.topline_tolerance_pp"],
        out_dir=run_dir,
    )
    print(report.summary())
    for w in report.weights:
        print(f"  weight  {w.message}")
    for t in report.toplines:
        if t.message and t.ok:
            print(f"  note    {t.item_id}: {t.message}")
    print(f"\nwritten to {run_dir}")
    return 0 if report.ok else 1


def cmd_doctor(args) -> int:
    """Check the local setup before anything spends quota or weeks of wall time.

    The check that matters is the context window. Ollama bakes it into the model
    at create time — the OpenAI-compatible endpoint has nowhere to put a
    `num_ctx`, so setting it in a request does nothing — and an undersized
    window truncates the top of the stat card without any error. That failure
    looks exactly like F2, so it is worth one command to rule out.
    """
    import json as _json
    import os
    import urllib.error
    import urllib.request

    from .llm.env import find_dotenv, load_dotenv

    loaded = load_dotenv()
    # `validate=False` on purpose. Doctor exists to diagnose a setup that is not
    # yet right, and the validator refuses a floating model alias on an active
    # provider — which is precisely the state doctor is being asked to help fix.
    # Validating here would make the one command that can list pinnable tags
    # refuse to run until the tag was already pinned. The errors are reported
    # rather than swallowed, so nothing is hidden.
    cfg = load_config(args.config, overrides=dict(args.set or []), validate=False)
    want_ctx = int(cfg.get("llm.num_ctx", 0))
    try:
        cfg.validate()
        config_errs = ""
    except ConfigError as exc:
        config_errs = str(exc)
    # Only providers the run will actually use can block it. An overflow arm
    # with no key is a spare, not a fault.
    active = set(cfg.get("llm.active_providers") or [])
    problems = 0

    dotenv = find_dotenv()
    print(f"config      {cfg.source_path.name}")
    print(f"num_ctx     {want_ctx} (prompts are built against this)")
    print(f"active      {sorted(active) or 'all providers'}")
    if loaded:
        print(f".env        {dotenv} -> loaded {sorted(loaded)}")
    elif dotenv:
        print(f".env        {dotenv} (every key in it was already set in the shell)")
    else:
        print("  .env      not found — hosted providers need their keys exported")

    for prov in cfg["llm.providers"]:
        name, model = prov["name"], prov["model"]
        required = (not active) or name in active
        if prov.get("enabled") is False:
            print(f"  {name:14s} parked")
            continue
        # By family, not by exact name: several arms of one provider run side by
        # side (mistral / mistral_8b / mistral_3b) and they share a key.
        from .llm.client import _DEFAULT_KEY_ENVS, _family_default
        env = _family_default(name, _DEFAULT_KEY_ENVS, "") or None
        if env:
            have = bool(os.environ.get(env))
            state = "key set" if have else f"no key ({env} unset)"
            if not have and not required:
                state += " — spare, not needed for this run"
            print(f"  {name:14s} {state}  [{model}]")
            if not have and required:
                problems += 1
            elif have and required:
                # List the provider's real tags. A floating alias is refused for
                # an active provider (cache.py keys on the model string), so a
                # concrete one has to come from somewhere, and it should come
                # through the same key path the run uses rather than a curl in a
                # shell that has never seen `.env`.
                url = _family_default(name, {
                    "mistral": "https://api.mistral.ai/v1/models",
                    "groq": "https://api.groq.com/openai/v1/models",
                }, None)
                if not url:
                    continue
                req = urllib.request.Request(
                    url, headers={"Authorization": f"Bearer {os.environ[env]}"}
                )
                try:
                    with urllib.request.urlopen(req, timeout=20) as r:
                        body = _json.loads(r.read().decode())
                except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                    print(f"                 could not list models: {exc}")
                    continue
                ids = sorted(str(m.get("id", "")) for m in body.get("data", []))
                stem = model.rsplit("-", 1)[0] if model.endswith("-latest") else model
                near = [i for i in ids if i.startswith(stem[:14]) and not i.endswith("-latest")]
                print(f"                 {len(ids)} tags available")
                if near:
                    print(f"                 pinnable, matching {stem!r}:")
                    for i in near:
                        print(f"                   {i}")
                    if str(model).endswith(("-latest", ":latest")):
                        print(f"                 ^ put one of these in "
                              f"configs/*.yaml for provider {name!r}; "
                              f"{model!r} is refused while {name!r} is active")
            continue

        # Ollama: is it up, is the model there, and how big is its window?
        base = prov.get("base_url", "http://localhost:11434/v1").replace("/v1", "")
        try:
            req = urllib.request.Request(
                f"{base}/api/show",
                data=_json.dumps({"model": model}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                info = _json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  {name:14s} not reachable at {base} ({exc})")
            problems += 1 if required else 0
            continue
        except urllib.error.HTTPError as exc:
            print(f"  {name:14s} up, but model {model!r} not found ({exc.code}) — `ollama pull {model}`")
            problems += 1
            continue

        params = str(info.get("parameters", ""))
        ctx = None
        for line in params.splitlines():
            if line.strip().startswith("num_ctx"):
                ctx = int(line.split()[-1])
        trained = None
        for k, v in (info.get("model_info") or {}).items():
            if k.endswith(".context_length"):
                trained = int(v)
        effective = ctx or 2048  # Ollama's default when the model sets none
        ok = effective >= want_ctx
        print(f"  {name:14s} up  [{model}]  num_ctx={effective}"
              f"{'' if ctx else ' (DEFAULT — model sets none)'}"
              f"{f', max {trained}' if trained else ''}"
              f"  {'ok' if ok else 'TOO SMALL'}")
        if not ok:
            problems += 1 if required else 0
            print(f"      A {want_ctx}-token prompt would be truncated to {effective} with no error,")
            print("      losing the TOP of the stat card — the cluster definition. Fix:")
            print(f"        printf 'FROM {model}\\nPARAMETER num_ctx {want_ctx}\\n' > Modelfile")
            new_tag = f"{model.split(':')[0]}-ctx{want_ctx // 1024}k:{model.split(':')[-1]}-{want_ctx}"
            print(f"        ollama create {new_tag} -f Modelfile")
            print("      then point llm.providers[ollama].model at that name.")

    if getattr(args, "live", False):
        # One real call per active hosted provider, through the real transport.
        # Everything cheaper than this has already passed at the point where the
        # 16 Sep runs failed: the key was present, the tag was pinned, the config
        # validated. What was left could only be seen by actually sending
        # something, and sending 960 of them to find out is how three runs were
        # spent. This is that check, for one call.
        import time as _time

        from .llm.client import LLMClient, ProviderError, http_transport

        print()
        print("live probe (one real call per active provider)")
        client = LLMClient.from_config(cfg)
        for spec in client.providers:
            if not spec.api_key_env:
                continue
            t0 = _time.monotonic()
            try:
                resp = http_transport(
                    spec, 'Reply with only this JSON: {"ok": 1}',
                    temperature=0.0, max_tokens=16, json_mode=True,
                )
            except ProviderError as exc:
                dt = _time.monotonic() - t0
                print(f"  {spec.name:14s} FAILED after {dt:.1f}s")
                for line in str(exc).split(" | "):
                    print(f"                 {line}")
                problems += 1
            else:
                dt = _time.monotonic() - t0
                print(f"  {spec.name:14s} ok in {dt:.1f}s  "
                      f"[{resp.model}]  {resp.completion_tokens} completion tokens")
                print(f"                 said: {resp.text.strip()[:80]!r}")

    if config_errs:
        print()
        print("the config itself does not validate yet:")
        for line in config_errs.splitlines()[1:]:
            print(f"  {line.strip()}" if line.strip().startswith("-") else f"  {line}")
        problems += 1

    print()
    print("all good" if problems == 0 else f"{problems} thing(s) to fix before elicitation")
    return 0 if problems == 0 else 1


def cmd_gate2(args) -> int:
    """Gate 2: partition, ClusterStats, noise floor, frozen item split."""
    from .evalx.gate2 import run_gate2

    cfg = load_config(args.config, overrides=dict(args.set or []))
    run_dir = snapshot_run(cfg, note="Gate 2")
    report = run_gate2(cfg, out_dir=run_dir)
    print(report.summary())
    print()
    print(report.noise_summary)
    print(f"\nwritten to {run_dir}")
    return 0 if report.ok else 1


def cmd_gate3(args) -> int:
    """Gate 3: the permutation test (checklist 3.5)."""
    import numpy as np
    import yaml

    from .agents.elicit import load_paraphrases
    from .agents.statcard import battery_near_duplicates, make_statcard
    from .calibration.crossfit import make_crossfit_plan
    from .clustering.partition import build_cluster_tree
    from .clustering.stats import compute_cluster_stats
    from .data.adapters.gss import load_gss
    from .data.pool import POST_FREEZE_DOCTRINE_FLAGS
    from .evalx.gate3 import PreflightFailed, run_permutation_test
    from .evalx.split import load_effective_split
    from .llm.client import LLMClient

    cfg = load_config(args.config, overrides=dict(args.set or []))
    run_dir = snapshot_run(cfg, note="Gate 3 — permutation test")

    codebook = yaml.safe_load(
        (cfg.repo_root / cfg["paths.codebooks"] / "gss_items.yaml").read_text()
    )["items"]
    split, amendments = load_effective_split(
        cfg.repo_root / cfg["item_split.freeze_path"]
    )
    for a in amendments:
        print(f"  split amendment {a['amendment']} ({a['file']}): "
              f"+{len(a['added_anchors'])} anchors {a['added_anchors']}", file=sys.stderr)
        print(f"    rule: {a['rule']}", file=sys.stderr)
        if a["cannot_serve"]:
            print(f"    cannot serve: {', '.join(a['cannot_serve'])} "
                  f"(no unassigned items) — those targets are unchanged",
                  file=sys.stderr)
    all_items = sorted(codebook)
    codes = {i: codebook[i]["codes"] for i in all_items}

    table = load_gss(cfg.bed_file, all_items, waves=cfg["bed.waves"])
    tree = build_cluster_tree(
        table.frame, axes=cfg["partition.axes"], item_codes=codes,
        min_cell=cfg["partition.min_cell"], k_target=cfg["partition.k_target"],
    )
    assign = tree.assign(table.frame)
    stats = compute_cluster_stats(table.frame, assign, all_items, codes)
    plan = make_crossfit_plan(
        split["anchors"], split["targets"],
        {i: codebook[i]["topic"] for i in all_items},
        n_folds=cfg["calibration.crossfit_folds"],
    )

    # Items: the highest-SNR targets, which is where a real effect should show
    # most clearly. If the model cannot read the card on these it will not on
    # anything.
    #
    # Minus the ones the pool's own doctrine should have excluded. Taking the
    # top of the SNR ranking is what surfaced them: the most demographically
    # determined items rank highest, so a fact that slipped the attitudinal
    # filter lands near the front. Scoring the premise on those would be
    # answering the §F3 objection by conceding it.
    ranked = [
        i for i in sorted(split["targets"], key=lambda i: -split["snr"].get(i, 0.0))
        if i not in POST_FREEZE_DOCTRINE_FLAGS
    ]
    skipped = [i for i in split["targets"] if i in POST_FREEZE_DOCTRINE_FLAGS]
    for i in skipped:
        print(f"  skipping {i} (SNR {split['snr'].get(i, 0):.2f}): "
              f"{POST_FREEZE_DOCTRINE_FLAGS[i]}", file=sys.stderr)
    items = ranked[: args.n_items]

    leaves = sorted(tree.leaves(), key=lambda n: -n.pop_share)
    if args.n_clusters and args.n_clusters < len(leaves):
        # Evenly spaced through the population-share ordering, so the subset
        # spans large and small clusters rather than only the big ones.
        idx = np.linspace(0, len(leaves) - 1, args.n_clusters).round().astype(int)
        leaves = [leaves[i] for i in sorted(set(idx))]
    cluster_ids = [n.cluster_id for n in leaves]

    # One card per (item, cluster), not one per cluster. 3.4's contract is
    # per target item — no near-duplicate of *that* item, no anchor from *that*
    # item's fold — so a card built for items[0] and reused for items[1:] is
    # only correct for the first item. Costs no extra calls: a call is made per
    # (item, cluster, arm) either way.
    near_dupes = battery_near_duplicates(all_items)
    cards = {}
    for item_id in items:
        for cid in cluster_ids:
            cards[(item_id, cid)] = make_statcard(
                cluster_id=cid, tree=tree, stats=stats, frame=table.frame,
                cluster_of=assign, target_item=item_id, codebook=codebook,
                anchor_pool=plan.anchors_for(item_id), near_duplicates=near_dupes,
                anchor_fold=plan.target_fold.get(item_id),
                anchors_per_card=cfg["elicitation.anchors_per_card"],
                fold_of=plan.fold_of,
            )

    # A profile name may carry a suffix that names a variant of the run rather
    # than a different ensemble — "permutation-k18" is the same 1x3 ensemble on a
    # different partition, and it needs its own store so two partitions' cluster
    # ids never share a file.
    profiles = cfg["elicitation.ensemble_profiles"]
    profile = profiles.get(args.profile) or profiles[args.profile.split("-", 1)[0]]
    n_calls = len(items) * len(cluster_ids) * 2 * profile["n_paraphrase"] * profile["n_repeat"]
    print(f"{len(items)} items x {len(cluster_ids)} clusters x 2 arms x "
          f"{profile['n_paraphrase']}x{profile['n_repeat']} = {n_calls} calls")
    if n_calls > cfg["llm.max_calls_per_run"]:
        print(f"  that exceeds llm.max_calls_per_run={cfg['llm.max_calls_per_run']}; "
              f"raise it deliberately or reduce --n-clusters", file=sys.stderr)
        return 2
    print()

    client = LLMClient.from_config(cfg, run_dir=run_dir)
    try:
        result = run_permutation_test(
            items=items, cluster_ids=cluster_ids, cards=cards, codebook=codebook,
            stats=stats, frame=table.frame, cluster_of=assign, codes=codes,
            client=client, paraphrases=load_paraphrases(),
            n_paraphrase=profile["n_paraphrase"], n_repeat=profile["n_repeat"],
            temperature=cfg["elicitation.temperature"],
            noise_floor=cfg["evaluation.pass_marks"]["all_targets"]["noise_floor"],
            seed=cfg["seed"], out_dir=run_dir,
        )
    except PreflightFailed as exc:
        print(f"\nSTOPPED before spending the run:\n  {exc}", file=sys.stderr)
        print(f"\nrun dir {run_dir}", file=sys.stderr)
        return 3
    print()
    print(result.summary())
    print(f"\nwritten to {run_dir}")
    return 0 if result.ok else 1


# ---------------------------------------------------------------- Phase 4/5

def _scope_items(bed, scope: str, n_items: int) -> list[str]:
    if scope == "anchors":
        return sorted(bed.split["anchors"])
    if scope == "population":
        # Both roles. The targets' population histograms are the B0b baseline
        # and the `predicted_level` mode's level input; the anchors' are what
        # the level isotonic is fitted on, and without them that mode has no
        # supervision at all.
        items = bed.ranked_targets()
        items = items[:n_items] if n_items else items
        return items + sorted(bed.split["anchors"])
    items = bed.ranked_targets()
    return items[:n_items] if n_items else items


def cmd_elicit(args) -> int:
    """Fill the elicitation store, resumably, inside a wall-clock budget.

    Both shells this project is driven from freeze between tool calls, so a long
    run is a sequence of short windows. Every window appends to the store and the
    next one skips what is already there; the response cache makes a repeated
    call free on top of that.
    """
    from .agents.elicit import load_paraphrases
    from .agents.runner import ElicitationStore, run_elicitation
    from .config import load_config
    from .llm.client import LLMClient
    from .pipeline import build_bed

    cfg = load_config(args.config, overrides=dict(args.set or []))
    bed = build_bed(cfg, verbose=True)
    # A profile name may carry a suffix that names a variant of the run rather
    # than a different ensemble — "permutation-k18" is the same 1x3 ensemble on a
    # different partition, and it needs its own store so two partitions' cluster
    # ids never share a file.
    profiles = cfg["elicitation.ensemble_profiles"]
    profile = profiles.get(args.profile) or profiles[args.profile.split("-", 1)[0]]
    items = _scope_items(bed, args.scope, args.n_items)
    if args.n_shards > 1:
        # Item-disjoint shards. The store is one file per item, so parallel
        # processes never write the same file, and the response cache is keyed
        # per call and written atomically. Sharding is how a window of the
        # chunked runner gets more than one call in flight at a time without
        # any locking.
        items = items[args.shard :: args.n_shards]
    clusters = bed.clusters(args.n_clusters)
    model = next(p["model"] for p in cfg["llm.providers"]
                 if p["name"] == (cfg["llm.active_providers"] or ["ollama"])[0])
    profile_key = args.profile if args.anchors_per_card is None else \
        f"{args.profile}-anchors{args.anchors_per_card}"
    store = ElicitationStore(
        root=cfg.repo_root / cfg["paths.runs_root"] / "_elicit_store",
        model=model, profile=profile_key)

    n_cells = len(items) if args.scope == "population" else len(items) * len(clusters)
    n_calls = n_cells * profile["n_paraphrase"] * profile["n_repeat"]
    print(f"  scope    {args.scope}: {len(items)} items"
          + ("" if args.scope == "population" else f" x {len(clusters)} clusters")
          + f" x {profile['n_paraphrase']}x{profile['n_repeat']} = {n_calls} calls")
    print(f"  store    {store.dir}")
    if n_calls > cfg["llm.max_calls_per_run"]:
        print(f"  that exceeds llm.max_calls_per_run={cfg['llm.max_calls_per_run']}; "
              f"raise it deliberately", file=sys.stderr)
        return 2

    client = LLMClient.from_config(cfg)
    sl = run_elicitation(
        bed, client, items=items, cluster_ids=clusters,
        paraphrases=load_paraphrases(),
        n_paraphrase=profile["n_paraphrase"], n_repeat=profile["n_repeat"],
        store=store, temperature=cfg["elicitation.temperature"],
        time_budget_s=args.seconds, population=(args.scope == "population"),
        anchors_per_card=args.anchors_per_card,
    )
    print(sl.summary())
    if sl.failure_rates:
        print(f"  failures {sl.failure_rates}")
    return 0 if (sl.finished or sl.cells_this_slice) else 1


def cmd_gate4(args) -> int:
    """Layer 3 — the oracle pass-through (checklist 4.5)."""
    from .config import load_config
    from .evalx.gate4 import run_oracle_passthrough
    from .pipeline import build_bed

    cfg = load_config(args.config, overrides=dict(args.set or []))
    run_dir = snapshot_run(cfg, note="Gate 4 — oracle pass-through")
    bed = build_bed(cfg, verbose=True)
    res = run_oracle_passthrough(
        bed, anchor_items=sorted(bed.split["anchors"]),
        target_items=bed.ranked_targets(), cluster_ids=bed.clusters(args.n_clusters),
        tolerance=args.tolerance, min_neff=cfg["evaluation.truth_min_neff"],
        out_dir=run_dir)
    print()
    print(res.summary())
    print(f"\nwritten to {run_dir}")
    return 0 if res.ok else 1


def cmd_layer4(args) -> int:
    """Layer 4 — the held-out-item verdict against the pre-registered pass marks."""
    from .agents.runner import ElicitationStore
    from .config import load_config
    from .evalx.harness import (
        collect_cells,
        evaluate_targets,
        fit_adversarial,
        fit_from_store,
    )
    from .pipeline import build_bed

    cfg = load_config(args.config, overrides=dict(args.set or []))
    run_dir = snapshot_run(cfg, note=f"Layer 4 — {args.mode}")
    bed = build_bed(cfg, verbose=True)
    model = args.model or next(
        p["model"] for p in cfg["llm.providers"]
        if p["name"] == (cfg["llm.active_providers"] or ["ollama"])[0])
    root = cfg.repo_root / cfg["paths.runs_root"] / "_elicit_store"
    store = ElicitationStore(root=root, model=model, profile=args.profile)

    anchors = sorted(bed.split["anchors"])
    targets = bed.ranked_targets()[: args.n_items] if args.n_items else bed.ranked_targets()
    clusters = bed.clusters(args.n_clusters)

    anchor_cells = collect_cells(store.load(anchors), max_members=args.ensemble)
    target_cells = collect_cells(store.load(targets), max_members=args.ensemble)
    level_cells = collect_cells(store.load(targets + anchors))
    level_cells = {k: v for k, v in level_cells.items() if k[1] == "all"}
    print(f"  store    {store.dir}")
    print(f"  cells    {len(anchor_cells)} anchor, {len(target_cells)} target, "
          f"{len(level_cells)} population")
    if not anchor_cells:
        print("  no anchor elicitations in the store — the calibrator is fitted on "
              "anchors, so run `popsim elicit --scope anchors` first", file=sys.stderr)
        return 2

    cal, diag = fit_from_store(
        bed, anchor_cells, mode=args.mode,
        min_neff=cfg["evaluation.truth_min_neff"],
        variance_restoration=(False if args.no_variance else None),
        fit_per_cluster=not args.no_pooling, level_cells=level_cells,
        use_subspace=not args.no_subspace)
    print()
    print(cal.summary())
    cal.to_json(run_dir / "calibrator.json")

    leaky = None
    if args.leakage:
        leaky = json.loads(Path(args.leakage).read_text()).get("leaky") or []
        print(f"  leaky    {len(leaky)} item(s) flagged by {args.leakage}: {leaky}")
    adversarial = None
    if args.adversarial:
        adversarial = fit_adversarial(
            bed, anchor_cells, {t: bed.topics.get(t, "other") for t in targets},
            min_neff=cfg["evaluation.truth_min_neff"])
        print(f"  adversarial  {len(adversarial)} topic-disjoint calibrators")
    report = evaluate_targets(
        bed, target_cells, cal, items=targets, cluster_ids=clusters, model=model,
        level_cells=level_cells, n_boot=args.n_boot, with_b3=not args.no_b3,
        leaky_items=leaky, adversarial=adversarial, profile=args.profile,
        ensemble_limit=args.ensemble, out_dir=run_dir)
    report.notes.update(diag)
    (run_dir / "layer4_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, default=str))
    print()
    print(report.summary())
    print(f"\nwritten to {run_dir}")
    return 0 if report.ok else 1

def cmd_leakage(args) -> int:
    """Layer 5 — the direct leakage probe (§5.6)."""
    from .config import load_config
    from .evalx.leakage import run_leakage_probe
    from .llm.client import LLMClient
    from .pipeline import build_bed

    cfg = load_config(args.config, overrides=dict(args.set or []))
    run_dir = snapshot_run(cfg, note="Layer 5 — leakage probe")
    bed = build_bed(cfg, verbose=True)
    items = bed.ranked_targets()[: args.n_items] if args.n_items else bed.ranked_targets()
    client = LLMClient.from_config(cfg, run_dir=run_dir)
    rep = run_leakage_probe(bed, client, items=items, n_repeat=args.n_repeat,
                            temperature=cfg["elicitation.temperature"], out_dir=run_dir)
    print()
    print(rep.summary())
    print(f"\nwritten to {run_dir}")
    return 0


def cmd_router(args) -> int:
    """M7 — route a question, or report the threshold curve (checklist 6.1-6.2)."""
    import yaml

    from .config import load_config
    from .scenarios.router import Router, evaluate_threshold

    cfg = load_config(args.config, overrides=dict(args.set or []))
    codebook = yaml.safe_load(
        (cfg.repo_root / cfg["paths.codebooks"] / "gss_items.yaml").read_text()
    )["items"]
    router = Router(codebook, threshold=float(cfg["router.threshold"]))

    if args.question:
        d = router.route(args.question)
        print(d.explain())
        for item, sim in d.runners_up:
            print(f"    also close: {item} {sim:.3f}")
        return 0

    path = cfg.repo_root / cfg["router.labeled_pairs"]
    pairs = [(p["text"], p["item"], bool(p["duplicate"]))
             for p in yaml.safe_load(path.read_text())["pairs"]]
    rows = evaluate_threshold(router, pairs)
    print(f"duplicate detection on {len(pairs)} hand-labeled pairs "
          f"({sum(1 for _, _, d in pairs if d)} duplicates), {path.name}")
    print(f"  {'thr':>5s} {'recall':>7s} {'precision':>10s} {'F1':>6s} {'flagged':>8s}")
    for r in rows:
        mark = "  <- configured" if abs(r["threshold"] - router.threshold) < 1e-9 else ""
        print(f"  {r['threshold']:5.2f} {r['recall']:7.3f} {r['precision']:10.3f} "
              f"{r['f1']:6.3f} {r['n_flagged']:8d}{mark}")
    best = max(rows, key=lambda r: r["f1"])
    print(f"\n  best F1 {best['f1']:.3f} at threshold {best['threshold']:.2f} "
          f"(recall {best['recall']:.3f}, precision {best['precision']:.3f})")
    print("  Routing a genuinely new question to `observed` is the worse of the two "
          "errors: the product would answer a question nobody asked, labelled as "
          "coming from data. Precision is held above recall for that reason.")
    return 0


def cmd_report(args) -> int:
    """M10 — build the results report from whatever is in runs/."""
    from .config import load_config
    from .report.build import build_report, collect_runs

    cfg = load_config(args.config, overrides=dict(args.set or []))
    runs_root = cfg.repo_root / cfg["paths.runs_root"]
    inputs = collect_runs(runs_root)
    out = Path(args.out) if args.out else runs_root / "report" / "b17_results.html"
    path = build_report(inputs, out, headline_model=args.model,
                        fragment=args.fragment)
    print(f"models with a Layer 4 report: {sorted(inputs.layer4)}")
    print(f"written to {path}  ({path.stat().st_size // 1024} KB)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="popsim", description="B17 population simulation")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common(sp):
        sp.add_argument("-c", "--config", default="configs/gss_main.yaml", type=Path)
        sp.add_argument("--set", action="append", type=_parse_override, metavar="KEY=VALUE")
        return sp

    _common(sub.add_parser("noop", help="write a config snapshot and exit")).set_defaults(fn=cmd_noop)
    _common(sub.add_parser("validate", help="validate a config against the checklist")).set_defaults(fn=cmd_validate)
    _common(sub.add_parser("gate1", help="run the Layer 0 topline gate")).set_defaults(fn=cmd_gate1)
    _common(sub.add_parser("gate2", help="run the Layer 1 gate")).set_defaults(fn=cmd_gate2)
    g3 = _common(sub.add_parser("gate3", help="run the Layer 2 permutation test"))
    g3.add_argument("--n-items", type=int, default=10)
    g3.add_argument("--n-clusters", type=int, default=0, help="0 = all leaves")
    g3.add_argument("--profile", default="permutation",
                    help="ensemble profile from the config (dev|permutation|minieval|headline)")
    g3.set_defaults(fn=cmd_gate3)
    e = _common(sub.add_parser("elicit", help="fill the elicitation store, resumably"))
    e.add_argument("--scope", choices=["targets", "anchors", "population"], default="targets")
    e.add_argument("--n-items", type=int, default=0, help="0 = all")
    e.add_argument("--n-clusters", type=int, default=0, help="0 = all leaves")
    e.add_argument("--profile", default="permutation")
    e.add_argument("--seconds", type=float, default=None,
                   help="wall-clock budget for this window; the store resumes")
    e.add_argument("--anchors-per-card", type=int, default=None,
                   help="override the card's anchor count; 0 is the §5.5 "
                        "no-anchor ablation. Writes to its own store profile.")
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--n-shards", type=int, default=1,
                   help="split the item list into disjoint shards for parallel windows")
    e.set_defaults(fn=cmd_elicit)
    g4 = _common(sub.add_parser("gate4", help="Layer 3 oracle pass-through"))
    g4.add_argument("--n-clusters", type=int, default=0)
    g4.add_argument("--tolerance", type=float, default=0.005)
    g4.set_defaults(fn=cmd_gate4)
    l4 = _common(sub.add_parser("layer4", help="the held-out-item verdict"))
    l4.add_argument("--n-items", type=int, default=0)
    l4.add_argument("--n-clusters", type=int, default=0)
    l4.add_argument("--profile", default="permutation")
    l4.add_argument("--model", default=None, help="store key; defaults to the active provider's model")
    l4.add_argument("--mode", choices=["observed_level", "predicted_level"],
                    default="observed_level")
    l4.add_argument("--n-boot", type=int, default=200)
    l4.add_argument("--no-variance", action="store_true",
                    help="ablation: force the variance-restoration step off "
                         "(by default it is used only if it lowers anchor W1)")
    l4.add_argument("--no-pooling", action="store_true", help="ablation: no per-cluster scale")
    l4.add_argument("--no-subspace", action="store_true",
                    help="ablation: no subgroup-subspace projection")
    l4.add_argument("--ensemble", type=int, default=0,
                    help="truncate each cell's ensemble to N members (§7.3 sweep); 0 = all")
    l4.add_argument("--adversarial", action="store_true",
                    help="also score a topic-disjoint calibrator (§5.1)")
    l4.add_argument("--leakage", type=Path, default=None,
                    help="a leakage_report.json; adds the minus-leaky subset")
    l4.add_argument("--no-b3", action="store_true", help="skip the supervised skyline")
    l4.set_defaults(fn=cmd_layer4)
    rp = _common(sub.add_parser("report", help="build the results report from runs/"))
    rp.add_argument("--out", default=None)
    rp.add_argument("--model", default=None, help="which arm is the headline")
    rp.add_argument("--fragment", action="store_true",
                    help="omit the document skeleton, for a host that supplies one")
    rp.set_defaults(fn=cmd_report)
    rt = _common(sub.add_parser("router", help="route a question, or tune the threshold"))
    rt.add_argument("question", nargs="?", help="omit to report the threshold curve")
    rt.set_defaults(fn=cmd_router)
    lk = _common(sub.add_parser("leakage", help="Layer 5 direct leakage probe"))
    lk.add_argument("--n-items", type=int, default=0)
    lk.add_argument("--n-repeat", type=int, default=3)
    lk.set_defaults(fn=cmd_leakage)
    d = _common(sub.add_parser("doctor", help="check local model + API key setup"))
    d.add_argument(
        "--live", action="store_true",
        help="also make ONE real call per active hosted provider and report "
             "latency, or the full 429 with its rate-limit headers. Costs one "
             "call per provider and answers what no dry check can.",
    )
    d.set_defaults(fn=cmd_doctor)
    q = _common(sub.add_parser("quota", help="show provider quota ledger"))
    q.add_argument(
        "--clear", action="append", metavar="PROVIDER",
        help="un-burn a provider marked exhausted (e.g. after a throttle was "
             "wrongly read as the day's allowance). Repeatable.",
    )
    q.set_defaults(fn=cmd_quota)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Before anything reads a key. `.env` is where `.env.example` says to put
    # them and nothing loaded it, so every hosted-provider call went out with an
    # empty bearer token and came back 401 — recorded per cell as a `provider`
    # failure, which is indistinguishable from the provider being down.
    from .llm.env import load_dotenv
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.fn(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
