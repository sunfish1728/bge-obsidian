"""Install the bge-obsidian skill into a vault (project-level), for Claude Code and Codex.

Writes <vault>/.claude/skills/bge-obsidian/SKILL.md and <vault>/.agents/skills/bge-obsidian/SKILL.md
with the absolute path of this project's bge-obs executable. Nothing is written outside the vault.

    .venv/Scripts/python scripts/install_skill.py [--vault PATH] [--only claude|codex] [--uninstall]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bge_obs.config import load_config  # noqa: E402

TARGETS = {"claude": ".claude/skills/bge-obsidian", "codex": ".agents/skills/bge-obsidian"}


def exe_path() -> Path:
    win = ROOT / ".venv" / "Scripts" / "bge-obs.exe"
    return win if win.exists() else ROOT / ".venv" / "bin" / "bge-obs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", help="vault path (default: vault in config.yaml)")
    ap.add_argument("--only", choices=sorted(TARGETS))
    ap.add_argument("--uninstall", action="store_true")
    a = ap.parse_args()

    vault = Path(a.vault).resolve() if a.vault else load_config().vault
    exe = exe_path()
    if not exe.exists() and not a.uninstall:
        print(f"bge-obs executable not found: {exe}", file=sys.stderr)
        return 1
    text = (ROOT / "skill" / "SKILL.md").read_text(encoding="utf-8")
    text = text.replace("{{BGE_OBS}}", exe.as_posix())

    for name, rel in TARGETS.items():
        if a.only and a.only != name:
            continue
        dest = vault / rel
        if a.uninstall:
            if dest.exists():
                shutil.rmtree(dest)
                print(f"removed {dest}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "SKILL.md").write_text(text, encoding="utf-8")
        print(f"installed {dest / 'SKILL.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
