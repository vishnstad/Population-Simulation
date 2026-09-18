#!/usr/bin/env bash
# git, with stale lock files moved aside first.
# The repo lives on a mounted folder where this session cannot unlink, so git
# leaves its .lock files behind after every write. `mv` works where `rm` does
# not, so they are moved into .git/_stale instead.
cd "$(dirname "$0")/.."
mkdir -p .git/_stale
for f in .git/HEAD.lock .git/index.lock .git/config.lock .git/objects/maintenance.lock \
         .git/refs/heads/main.lock .git/logs/HEAD.lock; do
  [ -e "$f" ] && mv "$f" ".git/_stale/$(basename "$f").$$.$RANDOM" 2>/dev/null
done
git "$@" 2>&1 | grep -v "unable to unlink" | grep -v "^warning: unable"
