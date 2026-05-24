param(
    [string]$ProjectRoot = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl",
    [string]$IsaacLabRoot = "G:\rt_isaaclab_ws\repos\IsaacLab_v2.3.2",
    [string]$OutputDir = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl\assets\motions"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PythonBat = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"
$Generator = Join-Path $ProjectRoot "src\g1_rl\data\g1_synthetic_motions.py"

Write-Host ""
Write-Host "============================================================"
Write-Host "Generate G1 synthetic motion references on Windows"
Write-Host "============================================================"
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "IsaacLabRoot = $IsaacLabRoot"
Write-Host "PythonBat = $PythonBat"
Write-Host "OutputDir = $OutputDir"
Write-Host "Generator = $Generator"
Write-Host "============================================================"

$required = @($ProjectRoot, $PythonBat, $Generator)

foreach ($p in $required) {
    if (-not (Test-Path $p)) {
        throw "Required path not found: $p"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"

Set-Location $ProjectRoot

& $PythonBat $Generator all `
    --output-dir $OutputDir `
    --task1-file g1_walk.pt `
    --task2-file g1_omni_walk.pt `
    --fps 50.0 `
    --task1-frames 600 `
    --frames-per-mode 600 `
    --gait-freq 1.45 `
    --target-vx 0.50 `
    --fade-ratio 0.08

if ($LASTEXITCODE -ne 0) {
    throw "G1 synthetic motion generation failed."
}

& $PythonBat $Generator validate --file (Join-Path $OutputDir "g1_walk.pt")
if ($LASTEXITCODE -ne 0) {
    throw "Task1 motion validation failed."
}

& $PythonBat $Generator validate --file (Join-Path $OutputDir "g1_omni_walk.pt")
if ($LASTEXITCODE -ne 0) {
    throw "Task2 motion validation failed."
}

Write-Host ""
Write-Host "[OK] G1 synthetic motion generation completed."
Write-Host ""
Write-Host "Generated:"
Write-Host "  $(Join-Path $OutputDir 'g1_walk.pt')"
Write-Host "  $(Join-Path $OutputDir 'g1_omni_walk.pt')"
Write-Host ""
Write-Host "Use in Windows scripts:"
Write-Host "  -MotionFile $(Join-Path $OutputDir 'g1_omni_walk.pt')"
Write-Host ""
