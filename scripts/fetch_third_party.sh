#!/usr/bin/env bash
# Clone the pinned third-party model repos. They are gitignored.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party"
mkdir -p "$DEST"
clone() {
  local url="$1"
  local name="$2"
  local commit="$3"
  if [[ ! -d "$DEST/$name/.git" ]]; then
    git clone --depth 1 "$url" "$DEST/$name"
  fi
  echo "$name $(git -C "$DEST/$name" rev-parse HEAD)"
}
clone https://github.com/Coder-Yu/SELFRec.git SELFRec 5b0229423cb1c727e85a704d63e460368c8b9dde
clone https://github.com/facebookresearch/generative-recommenders.git generative-recommenders ea7b85f16647766cbd7188a54bb447b20cee14bd
