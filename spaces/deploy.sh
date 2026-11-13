#!/usr/bin/env bash
# Build the Hugging Face Space repo content from this folder.
# Usage:
#   1. Create the Space on HF (owner Kaustubh73, name prefkit-replay, SDK Gradio, CPU).
#   2. hf login   # one-time; needs HF token
#   3. git clone https://huggingface.co/spaces/Kaustubh73/prefkit-replay space_repo
#   4. ./spaces/deploy.sh space_repo
# This copies the 5 app files + a fresh frozen library/ into the Space repo.
set -euo pipefail
REPO_DIR="${1:?usage: deploy.sh <path-to-cloned-space-repo>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../results"
cp "$HERE/app.py" "$HERE/load_lib.py" "$HERE/plot_lib.py" "$HERE/requirements.txt" "$HERE/README.md" "$REPO_DIR/"
rm -rf "$REPO_DIR/library"
mkdir -p "$REPO_DIR/library"
for d in assistant_seed0 assistant_seed1 empty persona gemma; do
  cp -r "$SRC/$d" "$REPO_DIR/library/"
done
echo "Copied app + $(find "$REPO_DIR/library" -name '*.json' | wc -l) result blobs into $REPO_DIR"
echo "Next: cd $REPO_DIR && git add -A && git commit -m 'prefkit replay space' && git push"
