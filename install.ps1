# bge-obsidian installer for Windows (PowerShell 5.1+).
# Clones or updates the repository, builds the project-local venv, configures the vault
# and installs the skill into the vault. Re-run it any time to update.
#
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/sunfish1728/bge-obsidian/main/install.ps1))) -Vault "D:\MyVault"
#
# Options: -Dir <install folder> -Vault <vault> -Cpu | -Cuda -SkipModels -Index -Branch <name> -DryRun
param(
    [string]$Dir = (Join-Path (Get-Location) "bge-obsidian"),
    [string]$Vault = "",
    [switch]$Cpu,
    [switch]$Cuda,
    [switch]$SkipModels,
    [switch]$Index,
    [switch]$DryRun,
    [string]$Branch = "main",
    [string]$Repo = "https://github.com/sunfish1728/bge-obsidian.git"
)
$ErrorActionPreference = "Stop"

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail "git is required: https://git-scm.com/download/win" }

# Find a Python >= 3.10 (the py launcher first, then python on PATH).
$python = $null
foreach ($cand in @(@("py", "-3"), @("python"), @("python3"))) {
    if (-not (Get-Command $cand[0] -ErrorAction SilentlyContinue)) { continue }
    $args_ = @($cand | Select-Object -Skip 1) + @("-c", "import sys; print(sys.version_info >= (3, 10))")
    try { $ok = & $cand[0] @args_ 2>$null } catch { continue }
    if ($LASTEXITCODE -eq 0 -and "$ok".Trim() -eq "True") { $python = $cand; break }
}
if (-not $python) { Fail "Python 3.10+ is required: https://www.python.org/downloads/" }
Write-Host "Using Python: $($python -join ' ')"

if (Test-Path (Join-Path $Dir ".git")) {
    Write-Host "Updating $Dir"
    git -C $Dir pull --ff-only origin $Branch
    if ($LASTEXITCODE -ne 0) { Fail "git pull failed (local changes?)" }
} else {
    Write-Host "Cloning into $Dir"
    git clone --branch $Branch $Repo $Dir
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed" }
}

if (-not $Vault -and [Environment]::UserInteractive) {
    $Vault = Read-Host "Obsidian vault path (leave empty to set it later)"
}

$setup = @(Join-Path $Dir "scripts\setup_env.py")
if ($Cpu) { $setup += "--cpu" }
if ($Cuda) { $setup += "--cuda" }
if ($SkipModels) { $setup += "--skip-models" }
if ($Vault) { $setup += @("--vault", $Vault) }
if ($Index -and $Vault) { $setup += "--index" }
if ($DryRun) { $setup += "--dry-run" }

$exe = $python[0]
$pre = @($python | Select-Object -Skip 1)
& $exe @pre @setup
if ($LASTEXITCODE -ne 0) { Fail "setup failed" }
