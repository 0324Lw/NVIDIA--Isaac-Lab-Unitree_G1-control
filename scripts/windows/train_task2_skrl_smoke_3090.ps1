param(
    [string]$ProjectRoot = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl",
    [string]$IsaacLabRoot = "G:\rt_isaaclab_ws\repos\IsaacLab_v2.3.2",
    [string]$G1UsdPath = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl\assets\robots\g1.usd",
    [string]$MotionFile = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl\assets\motions\g1_omni_walk.pt",
    [string]$Task1Pretrained = "",

    [int]$NumEnvs = 8,
    [Int64]$TotalEnvSteps = 16384,
    [int]$Rollouts = 16,
    [int]$LearningEpochs = 3,
    [int]$MiniBatches = 2,
    [string]$Device = "cuda:0",
    [string]$RunName = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PythonBat = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"
$TrainPy = Join-Path $ProjectRoot "src\g1_rl\tasks\task2\task2_train.py"
$LogRoot = "G:\rt_isaaclab_ws\logs\unitree_g1_isaaclab_rl\task2"
$DriverLogRoot = Join-Path $LogRoot "windows_driver"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
New-Item -ItemType Directory -Force -Path $DriverLogRoot | Out-Null

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "win_g1_task2_smoke_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$DriverLog = Join-Path $DriverLogRoot ($RunName + "_driver.log")

Write-Host ""
Write-Host "============================================================"
Write-Host "G1 Task2 Windows RTX 3090 skrl PPO SMOKE runner"
Write-Host "============================================================"
Write-Host "ProjectRoot      = $ProjectRoot"
Write-Host "IsaacLabRoot     = $IsaacLabRoot"
Write-Host "TrainPy          = $TrainPy"
Write-Host "G1UsdPath        = $G1UsdPath"
Write-Host "MotionFile       = $MotionFile"
Write-Host "Task1Pretrained  = $Task1Pretrained"
Write-Host "NumEnvs          = $NumEnvs"
Write-Host "TotalEnvSteps    = $TotalEnvSteps"
Write-Host "RunName          = $RunName"
Write-Host "DriverLog        = $DriverLog"
Write-Host "============================================================"

if ($env:G1_TASK2_WINDOWS_SMOKE_APPROVED -ne "1") {
    Write-Host ""
    Write-Host "[STOP] To actually start G1 Task2 smoke training, run first:" -ForegroundColor Yellow
    Write-Host '  $env:G1_TASK2_WINDOWS_SMOKE_APPROVED = "1"'
    exit 0
}

$required = @($PythonBat, $TrainPy, $G1UsdPath, $MotionFile)
foreach ($p in $required) {
    if (-not (Test-Path $p)) {
        throw "Required path not found: $p"
    }
}

$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"
$env:RT_G1_TASK2_LOG_ROOT = $LogRoot
$env:G1_USD_PATH = $G1UsdPath
$env:G1_TASK2_MOTION_FILE = $MotionFile
if (-not [string]::IsNullOrWhiteSpace($Task1Pretrained)) {
    $env:G1_TASK1_PRETRAINED = $Task1Pretrained
}

Set-Location $ProjectRoot

$ArgsList = @(
    $TrainPy,
    "--num-envs", "$NumEnvs",
    "--total-env-steps", "$TotalEnvSteps",
    "--rollouts", "$Rollouts",
    "--learning-epochs", "$LearningEpochs",
    "--mini-batches", "$MiniBatches",
    "--lr", "1e-4",
    "--min-lr", "7e-5",
    "--max-lr", "2e-4",
    "--summary-interval", "1",
    "--tb-log-interval-steps", "10",
    "--skrl-write-interval", "1000000",
    "--skrl-checkpoint-interval", "0",
    "--save-freq-env-steps", "$TotalEnvSteps",
    "--log-root", "$LogRoot",
    "--run-name", "$RunName",
    "--usd-path", "$G1UsdPath",
    "--motion-file", "$MotionFile",
    "--headless",
    "--device", "$Device"
)

if (-not [string]::IsNullOrWhiteSpace($Task1Pretrained)) {
    $ArgsList += @("--pretrained-task1", "$Task1Pretrained")
}

& $PythonBat @ArgsList 2>&1 | Tee-Object -FilePath $DriverLog

$exitCode = $LASTEXITCODE
Write-Host ""
Write-Host "============================================================"
Write-Host "G1 Task2 Windows smoke finished with exit code: $exitCode"
Write-Host "Driver log: $DriverLog"
Write-Host "============================================================"

exit $exitCode
