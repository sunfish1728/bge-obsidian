"""Bootstrap a project-local environment using only the standard library."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    flavour = parser.add_mutually_exclusive_group()
    flavour.add_argument("--cpu", action="store_true")
    flavour.add_argument("--cuda", action="store_true")
    parser.add_argument("--cuda-index", default="cu128")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--vault", help="Obsidian vault: write it to config.yaml and install the skill there")
    parser.add_argument("--index", action="store_true", help="build the vault index after setup (needs --vault)")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(PIP_CACHE_DIR=str(ROOT / ".pip-cache"),
               HF_HOME=str(ROOT / "models" / ".hf-cache"),
               TMP=str(ROOT / "state" / "setup-tmp"),
               TEMP=str(ROOT / "state" / "setup-tmp"),
               TMPDIR=str(ROOT / "state" / "setup-tmp"))
    print(f"PIP_CACHE_DIR={env['PIP_CACHE_DIR']}")
    print(f"HF_HOME={env['HF_HOME']}")

    def run(command, probe=False, **kwargs):
        """Run a command; in dry-run only read-only probes actually execute."""
        command = [str(part) for part in command]
        print("+ " + (subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)), flush=True)
        if args.dry_run and not probe:
            return None
        return subprocess.run(command, cwd=ROOT, env=env, **kwargs)

    if not args.dry_run:
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    cuda = args.cuda
    if not args.cpu and not args.cuda:
        smi = shutil.which("nvidia-smi")
        if smi:
            try:
                result = run([smi], probe=True, capture_output=True, timeout=10)
                cuda = result is not None and result.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                cuda = False
    index = f"https://download.pytorch.org/whl/{args.cuda_index if cuda else 'cpu'}"
    print(f"Torch flavour: {'CUDA' if cuda else 'CPU'}; index: {index}")
    target = ROOT / ".venv"
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if args.recreate and target.exists():
        # Refuse to follow a link outside the project when recursively deleting.
        if target.is_symlink() or target.resolve() != ROOT / ".venv":
            raise RuntimeError(f"Refusing to delete linked environment: {target}")
        print(f"Remove and recreate {target}")
        if not args.dry_run:
            shutil.rmtree(target)
    if args.recreate or not target.exists():
        print(f"Create {target} with venv (with_pip=True)")
        if args.dry_run:
            run([sys.executable, "-m", "venv", target])
        else:
            # EnvBuilder's pip subprocess inherits these project-local paths.
            previous = {key: os.environ.get(key) for key in env}
            try:
                os.environ.update(env)
                venv.EnvBuilder(with_pip=True).create(target)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    matching = False
    if not args.recreate and python.exists():
        result = run([python, "-c", "import torch; print(torch.version.cuda)"],
                     probe=True, capture_output=True, text=True)
        if result is not None and result.returncode == 0:
            matching = (result.stdout.strip() != "None") == cuda
    if matching:
        print("Existing torch flavour matches; keeping it.")
    else:
        run([python, "-m", "pip", "install", "torch", "torchvision",
             "--index-url", index, "--force-reinstall"], check=True)
    run([python, "-m", "pip", "install", "-e", ".[local,dev]"], check=True)
    vendor = ROOT / "vendor" / "FlagEmbedding"
    if not vendor.exists():
        if not args.dry_run:
            vendor.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "-c", "credential.helper=", "-c", "core.autocrlf=false",
             "clone", "--depth", "1", "https://github.com/FlagOpen/FlagEmbedding.git", vendor], check=True)
    else:
        print(f"Keep existing vendor source: {vendor}")
    if not args.skip_models:
        run([python, ROOT / "scripts" / "download_model.py"], check=True)
    config = ROOT / "config.yaml"
    if not config.exists():
        print(f"Copy config.example.yaml to {config}")
        if not args.dry_run:
            shutil.copyfile(ROOT / "config.example.yaml", config)
    cli = target / ("Scripts/bge-obs.exe" if os.name == "nt" else "bin/bge-obs")
    if args.vault:
        vault = Path(args.vault).expanduser().resolve()
        if not vault.is_dir():
            raise RuntimeError(f"vault folder not found: {vault}")
        print(f"Set vault in {config}: {vault}")
        if not args.dry_run:
            set_vault(config, vault)
        run([python, ROOT / "scripts" / "install_skill.py", "--vault", vault], check=True)
        if args.index:
            run([cli, "index"], check=True)
        print("Done. Open the vault in Claude Code or Codex and ask it to organise _inbox.")
        if not args.index:
            print(f'Build the index once with:  "{cli}" index')
    else:
        print("Next: set vault in config.yaml, then run:")
        print(f'  "{cli}" index')
        print(f'  "{python}" scripts/install_skill.py')


def set_vault(config: Path, vault: Path) -> None:
    """Set the top-level `vault:` key in config.yaml, keeping every other line."""
    line = f'vault: "{vault.as_posix()}"'
    lines = config.read_text(encoding="utf-8").splitlines() if config.exists() else []
    for i, existing in enumerate(lines):
        if existing.startswith("vault:"):
            lines[i] = line
            break
    else:
        lines.insert(0, line)
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        sys.exit(1)
