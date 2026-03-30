#!/usr/bin/env bash
# Download all VibeVoice-Realtime voice prompt files (.pt).
#
# Default voices (25) come from the upstream VibeVoice GitHub repo.
# Experimental voices (36) come from GitHub user-attachment archives.
#
# Usage:
#   bash demo/download_voices.sh              # download everything
#   bash demo/download_voices.sh --default    # default voices only
#   bash demo/download_voices.sh --experimental  # experimental voices only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_DIR="$SCRIPT_DIR/voices/streaming_model"
EXPERIMENTAL_DIR="$VOICE_DIR/experimental_voices"

GITHUB_RAW="https://github.com/microsoft/VibeVoice/raw/refs/heads/main/demo/voices/streaming_model"

# ── Default voices (from upstream repo) ──────────────────────────────────────
DEFAULT_VOICES=(
  de-Spk0_man.pt
  de-Spk1_woman.pt
  en-Carter_man.pt
  en-Davis_man.pt
  en-Emma_woman.pt
  en-Frank_man.pt
  en-Grace_woman.pt
  en-Mike_man.pt
  fr-Spk0_man.pt
  fr-Spk1_woman.pt
  in-Samuel_man.pt
  it-Spk0_woman.pt
  it-Spk1_man.pt
  jp-Spk0_man.pt
  jp-Spk1_woman.pt
  kr-Spk0_woman.pt
  kr-Spk1_man.pt
  nl-Spk0_man.pt
  nl-Spk1_woman.pt
  pl-Spk0_man.pt
  pl-Spk1_woman.pt
  pt-Spk0_woman.pt
  pt-Spk1_man.pt
  sp-Spk0_woman.pt
  sp-Spk1_man.pt
)

# ── Experimental voice archives (GitHub user-attachments) ────────────────────
EXPERIMENTAL_ARCHIVES=(
  "experimental_voices_de.tar.gz|https://github.com/user-attachments/files/24035887/experimental_voices_de.tar.gz"
  "experimental_voices_fr.tar.gz|https://github.com/user-attachments/files/24035880/experimental_voices_fr.tar.gz"
  "experimental_voices_jp.tar.gz|https://github.com/user-attachments/files/24035882/experimental_voices_jp.tar.gz"
  "experimental_voices_kr.tar.gz|https://github.com/user-attachments/files/24035883/experimental_voices_kr.tar.gz"
  "experimental_voices_pl.tar.gz|https://github.com/user-attachments/files/24035885/experimental_voices_pl.tar.gz"
  "experimental_voices_pt.tar.gz|https://github.com/user-attachments/files/24035886/experimental_voices_pt.tar.gz"
  "experimental_voices_sp.tar.gz|https://github.com/user-attachments/files/24035884/experimental_voices_sp.tar.gz"
  "experimental_voices_en1.tar.gz|https://github.com/user-attachments/files/24189272/experimental_voices_en1.tar.gz"
  "experimental_voices_en2.tar.gz|https://github.com/user-attachments/files/24189273/experimental_voices_en2.tar.gz"
)

# ── Helpers ──────────────────────────────────────────────────────────────────

download_default_voices() {
  echo "[INFO] Downloading default voices to $VOICE_DIR ..."
  mkdir -p "$VOICE_DIR"

  local count=0
  for name in "${DEFAULT_VOICES[@]}"; do
    local dest="$VOICE_DIR/$name"
    if [[ -f "$dest" ]]; then
      echo "  [SKIP] $name (already exists)"
      continue
    fi
    echo "  [GET]  $name"
    curl -fSL --retry 3 -o "$dest" "$GITHUB_RAW/$name"
    ((count++))
  done
  echo "[INFO] Default voices: downloaded $count, skipped $((${#DEFAULT_VOICES[@]} - count))."
}

download_experimental_voices() {
  echo "[INFO] Downloading experimental voices to $EXPERIMENTAL_DIR ..."
  mkdir -p "$EXPERIMENTAL_DIR"

  for entry in "${EXPERIMENTAL_ARCHIVES[@]}"; do
    IFS="|" read -r fname url <<< "$entry"
    echo "  [GET]  $fname"
    curl -fSL --retry 3 -o "/tmp/$fname" "$url"
    echo "  [TAR]  $fname"
    tar -xzf "/tmp/$fname" -C "$EXPERIMENTAL_DIR"
    rm -f "/tmp/$fname"
  done
  echo "[INFO] Experimental voices installed."
}

# ── Main ─────────────────────────────────────────────────────────────────────

MODE="${1:-all}"

case "$MODE" in
  --default)
    download_default_voices
    ;;
  --experimental)
    download_experimental_voices
    ;;
  all|"")
    download_default_voices
    echo ""
    download_experimental_voices
    ;;
  *)
    echo "Usage: $0 [--default|--experimental]"
    exit 1
    ;;
esac

echo ""
echo "[DONE] Voice prompts ready at: $VOICE_DIR"
