"""How much of the store is filled, in one line per scope."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from popsim.config import load_config
from popsim.pipeline import build_bed

model, profile = sys.argv[1], sys.argv[2]
cfg = load_config("configs/gss_main.yaml")
bed = build_bed(cfg)
d = Path("runs/_elicit_store") / model.replace(":", "_") / profile
want_c = len(bed.clusters(0))
anchors = sorted(bed.split["anchors"]); targets = bed.ranked_targets()
per = {p.stem: sum(1 for x in p.read_text().splitlines() if x.strip())
       for p in d.glob("*.jsonl")} if d.exists() else {}
for name, items, per_item in (("anchors", anchors, want_c * 3),
                              ("targets", targets, want_c * 3)):
    got = sum(min(per.get(i, 0), per_item) for i in items)
    tot = len(items) * per_item
    print(f"  {name:8s} {got:6d} / {tot:6d}  {100*got/tot:5.1f}%"
          f"   items complete: {sum(1 for i in items if per.get(i,0) >= per_item)}/{len(items)}")
print(f"  records  {sum(per.values())} in {len(per)} files")
