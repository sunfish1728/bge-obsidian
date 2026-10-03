#!/usr/bin/env bash
# bge-obsidian installer for macOS / Linux.
# Clones or updates the repository, builds the project-local venv, configures the vault
# and installs the skill into the vault. Re-run it any time to update.
#
#   curl -fsSL https://raw.githubusercontent.com/sunfish1728/bge-obsidian/main/install.sh | bash -s -- --vault ~/MyVault
#
# Options: --dir <install folder> --vault <vault> --cpu | --cuda --skip-models --index --branch <name> --dry-run
set -euo pipefail

DIR="$PWD/bge-obsidian"
VAULT=""
BRANCH="main"
REPO="https://github.com/sunfish1728/bge-obsidian.git"
SETUP_ARGS=()
INDEX=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --vault) VAULT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --cpu|--cuda|--skip-models|--dry-run) SETUP_ARGS+=("$1"); shift ;;
    --index) INDEX=1; shift ;;
    -h|--help) sed -n '2,8p' "$0" 2>/dev/null || true; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }

command -v git >/dev/null || fail "git is required"
PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    PY="$cand"; break
  fi
done
[ -n "$PY" ] || fail "Python 3.10+ is required"
echo "Using Python: $PY"

if [ -d "$DIR/.git" ]; then
  echo "Updating $DIR"
  git -C "$DIR" pull --ff-only origin "$BRANCH" || fail "git pull failed (local changes?)"
else
  echo "Cloning into $DIR"
  git clone --branch "$BRANCH" "$REPO" "$DIR"
fi

if [ -z "$VAULT" ] && [ -t 0 ]; then
  read -r -p "Obsidian vault path (leave empty to set it later): " VAULT
fi
if [ -n "$VAULT" ]; then
  SETUP_ARGS+=(--vault "$VAULT")
  [ "$INDEX" = 1 ] && SETUP_ARGS+=(--index)
fi

"$PY" "$DIR/scripts/setup_env.py" ${SETUP_ARGS[@]+"${SETUP_ARGS[@]}"}
