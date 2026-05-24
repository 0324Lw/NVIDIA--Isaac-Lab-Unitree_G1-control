param(
    [Parameter(Mandatory=$true)]
    [string]$Checkpoint,

    [string]$ProjectRoot = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl",
    [string]$IsaacLabRoot = "G:\rt_isaaclab_ws\repos\IsaacLab_v2.3.2",
    [string]$G1UsdPath = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl\assets\robots\g1.usd",
    [string]$MotionFile = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl\assets\motions\g1_walk.pt",

    [int]$NumEnvs = 4,
    [int]$Steps = 2000,
    [double]$StartK = 0.10,
    [string]$Device = "cuda:0"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PythonBat = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"
$EvalPy = Join-Path $ProjectRoot "src\g1_rl\tasks\task1\task1_model_test.py"
$LogRoot = "G:\rt_isaaclab_ws\logs\unitree_g1_isaaclab_rl\task1"
$DriverLogRoot = Join-Path $LogRoot "windows_driver"

New-Item -ItemType Directory -Force -Path $DriverLogRoot | Out-Null

$RunName = "win_g1_task1_eval_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$DriverLog = Join-Path $DriverLogRoot ($RunName + "_driver.log")

Write-Host ""
Write-Host "============================================================"
Write-Host "G1 Task1 Windows skrl PPO model evaluation runner"
Write-Host "============================================================"
Write-Host "Checkpoint   = $Checkpoint"
Write-Host "ProjectRoot  = $ProjectRoot"
Write-Host "IsaacLabRoot = $IsaacLabRoot"
Write-Host "G1UsdPath    = $G1UsdPath"
Write-Host "MotionFile   = $MotionFile"
Write-Host "NumEnvs      = $NumEnvs"
Write-Host "Steps        = $Steps"
Write-Host "StartK       = $StartK"
Write-Host "Device       = $Device"
Write-Host "DriverLog    = $DriverLog"
Write-Host "============================================================"

$required = @($PythonBat, $EvalPy, $Checkpoint, $G1UsdPath, $MotionFile)
foreach ($p in $required) {
    if (-not (Test-Path $p)) {
        throw "Required path not found: $p"
    }
}

$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"
$env:RT_G1_TASK1_LOG_ROOT = $LogRoot
$env:G1_USD_PATH = $G1UsdPath
$env:G1_TASK1_MOTION_FILE = $MotionFile

Set-Location $ProjectRoot

$ArgsList = @(
    $EvalPy,
    "--checkpoint", "$Checkpoint",
    "--num-envs", "$NumEnvs",
    "--steps", "$Steps",
    "--start-k", "$StartK",
    "--print-interval", "100",
    "--usd-path", "$G1UsdPath",
    "--motion-file", "$MotionFile",
    "--headless",
    "--device", "$Device"
)

& $PythonBat @ArgsList 2>&1 | Tee-Object -FilePath $DriverLog

$exitCode = $LASTEXITCODE
Write-Host ""
Write-Host "============================================================"
Write-Host "G1 Task1 Windows model eval finished with exit code: $exitCode"
Write-Host "Driver log: $DriverLog"
Write-Host "============================================================"

exit $exitCode
