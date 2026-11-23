#!/usr/bin/env bash
# setup-tree.sh [DIR] : builds the llama.cpp tree these patches apply to, then applies them.
#   base  = GodGMN/llama.cpp inkling-rdna3 (b12600 + Inkling 2 Mini HIP fusions)
#         + merge of ggml-org PR #32718 (loop KV/state on the device of the physical layer, Devasy's branch)
#         + merge of ggml-org PR #32218 (Inkling 2 DSpark draft model)
#   then  inkling2/patches/*.patch with git am
set -euo pipefail
DIR=${1:-llama.cpp}
HERE=$(cd "$(dirname "$0")" && pwd)
GODGMN=cb0e469607c19b63b3d06a8b654fcebdd76fb889
LOOPSTATE=0e9f83e21697cb9135886f0ef7ef080aa50f6aac
DSPARK=e4b1cb5318f1ac55d148be228c6347c6a94cc4c6
BASE_TREE=7f5eed3430d091f1511b75260b3ee4faf2485754
PATCHED_TREE=19c4b58504f22c161763c43036766a55cf7c032d

git init -q "$DIR" && cd "$DIR"
git fetch -q https://github.com/GodGMN/llama.cpp "$GODGMN"
git checkout -q -b inkling2-rocm FETCH_HEAD
git fetch -q https://github.com/Devasy/llama.cpp "$LOOPSTATE"
git -c user.name=setup -c user.email=setup@localhost merge -q --no-edit -m "merge PR #32718" FETCH_HEAD
git fetch -q https://github.com/ggml-org/llama.cpp "pull/32218/head"
git cat-file -e "$DSPARK" 2>/dev/null || { echo "PR #32218 moved past $DSPARK; fetch that commit by hand" >&2; exit 1; }
git -c user.name=setup -c user.email=setup@localhost merge -q --no-edit -m "merge PR #32218" "$DSPARK"
[ "$(git rev-parse HEAD^{tree})" = "$BASE_TREE" ] || { echo "base tree differs from the one the patches were made on" >&2; exit 1; }
ID=(); git config user.email >/dev/null || ID=(-c user.name=setup -c user.email=setup@localhost)   # git am needs a committer
git "${ID[@]}" am -q "$HERE"/patches/*.patch
[ "$(git rev-parse HEAD^{tree})" = "$PATCHED_TREE" ] || echo "note: patched tree differs from the tested one" >&2
echo "ready: $(pwd) at $(git log --oneline -1)"
