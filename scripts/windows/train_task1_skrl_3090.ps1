param(
    [string]$ProjectRoot = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl",
    [string]$IsaacLabRoot = "G:\rt_isaaclab_ws\repos\IsaacLab_v2.3.2",
    [string]$G1UsdPath = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl\assets\robots\g1.usd",
    [string]$MotionFile = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl\assets\motions\g1_walk.pt",

    [int]$NumEnvs = 512,
    [Int64]$TotalEnvSteps = 300000000,
    [int]$Rollouts = 64,
    [int]$LearningEpochs = 5,
    [int]$MiniBatches = 8,
    [double]$Lr = 0.0002,
    [double]$MinLr = 0.00002,
    [double]$MaxLr = 0.0003,
    [double]$StartK = 0.0,
    [string]$Device = "cuda:0",
    [string]$RunName = "",
    [string]$Resume = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PythonBat = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"
$TrainPy = Join-Path $ProjectRoot "src\g1_rl\tasks\task1\task1_train.py"
$LogRoot = "G:\rt_isaaclab_ws\logs\unitree_g1_isaaclab_rl\task1"
$DriverLogRoot = Join-Path $LogRoot "windows_driver"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
New-Item -ItemType Directory -Force -Path $DriverLogRoot | Out-Null

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "win_g1_task1_3090_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$DriverLog = Join-Path $DriverLogRoot ($RunName + "_driver.log")

Write-Host ""
Write-Host "============================================================"
Write-Host "G1 Task1 Windows RTX 3090 skrl PPO formal training runner"
Write-Host "============================================================"
Write-Host "ProjectRoot   = $ProjectRoot"
Write-Host "IsaacLabRoot  = $IsaacLabRoot"
Write-Host "TrainPy       = $TrainPy"
Write-Host "G1UsdPath     = $G1UsdPath"
Write-Host "MotionFile    = $MotionFile"
Write-Host "NumEnvs       = $NumEnvs"
Write-Host "TotalEnvSteps = $TotalEnvSteps"
Write-Host "Rollouts      = $Rollouts"
Write-Host "LR            = $Lr / $MinLr / $MaxLr"
Write-Host "StartK        = $StartK"
Write-Host "Device        = $Device"
Write-Host "RunName       = $RunName"
Write-Host "Resume        = $Resume"
Write-Host "DriverLog     = $DriverLog"
Write-Host "============================================================"
Write-Host "Note: This is an educational pure-RL humanoid baseline, not HoloSoma / BeyondMimic imitation pipeline."

if ($env:G1_TASK1_WINDOWS_TRAIN_APPROVED -ne "1") {
    Write-Host ""
    Write-Host "[STOP] To actually start Windows G1 Task1 training, run first:" -ForegroundColor Yellow
    Write-Host '  $env:G1_TASK1_WINDOWS_TRAIN_APPROVED = "1"'
    Write-Host ""
    Write-Host "Suggested first formal run:"
    Write-Host "  .\scripts\windows\train_task1_skrl_3090.ps1 -NumEnvs 128 -TotalEnvSteps 20000000"
    Write-Host ""
    exit 0
}

$required = @($PythonBat, $TrainPy, $G1UsdPath, $MotionFile)
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
    $TrainPy,
    "--num-envs", "$NumEnvs",
    "--total-env-steps", "$TotalEnvSteps",
    "--rollouts", "$Rollouts",
    "--learning-epochs", "$LearningEpochs",
    "--mini-batches", "$MiniBatches",
    "--lr", "$Lr",
    "--min-lr", "$MinLr",
    "--max-lr", "$MaxLr",
    "--gamma", "0.99",
    "--gae-lambda", "0.95",
    "--kl-threshold", "0.015",
    "--entropy-coef", "0.002",
    "--value-coef", "2.0",
    "--init-log-std", "-1.35",
    "--start-k", "$StartK",
    "--summary-interval", "20",
    "--tb-log-interval-steps", "100",
    "--skrl-write-interval", "1000000",
    "--skrl-checkpoint-interval", "0",
    "--save-freq-env-steps", "20000000",
    "--log-root", "$LogRoot",
    "--run-name", "$RunName",
    "--usd-path", "$G1UsdPath",
    "--motion-file", "$MotionFile",
    "--headless",
    "--device", "$Device"
)

if (-not [string]::IsNullOrWhiteSpace($Resume)) {
    $ArgsList += @("--resume", "$Resume")
}

& $PythonBat @ArgsList 2>&1 | Tee-Object -FilePath $DriverLog

$exitCode = $LASTEXITCODE
Write-Host ""
Write-Host "============================================================"
Write-Host "G1 Task1 Windows RTX 3090 training finished with exit code: $exitCode"
Write-Host "Driver log: $DriverLog"
Write-Host "============================================================"

exit $exitCode
