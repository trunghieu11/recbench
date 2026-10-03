#!/usr/bin/env bash
# Clone the third-party model repositories at the exact commits recbench was tested with.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party"
mkdir -p "$DEST"
fetch() {
  local url="$1" name="$2" commit="$3"
  if [[ ! -d "$DEST/$name/.git" ]]; then
    git init -q "$DEST/$name"
    git -C "$DEST/$name" remote add origin "$url"
  fi
  if [[ "$(git -C "$DEST/$name" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]]; then
    git -C "$DEST/$name" fetch -q --depth 1 origin "$commit"
    git -C "$DEST/$name" checkout -q --detach FETCH_HEAD
  fi
  echo "$name $(git -C "$DEST/$name" rev-parse HEAD)"
}
fetch https://github.com/Coder-Yu/SELFRec.git SELFRec 5b0229423cb1c727e85a704d63e460368c8b9dde
fetch https://github.com/facebookresearch/generative-recommenders.git generative-recommenders ea7b85f16647766cbd7188a54bb447b20cee14bd
# Official GRU4Rec (PyTorch) by its author. Its licence restricts commercial use; it is fetched here, never vendored.
fetch https://github.com/hidasib/GRU4Rec_PyTorch_Official.git GRU4Rec_PyTorch_Official d1fc31105577665d2f105de1c8d9a38d223a9a22
