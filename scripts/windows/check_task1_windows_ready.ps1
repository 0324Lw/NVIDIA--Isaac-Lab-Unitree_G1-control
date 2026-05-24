param(
    [string]$ProjectRoot = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl",
    [string]$IsaacLabRoot = "G:\rt_isaaclab_ws\repos\IsaacLab_v2.3.2",
    [string]$G1UsdPath = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl\assets\robots\g1.usd",
    [string]$MotionFile = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl\assets\motions\g1_walk.pt"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PythonBat = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"

$ConfigPy = Join-Path $ProjectRoot "src\g1_rl\tasks\task1\task1_config.py"
$EnvPy = Join-Path $ProjectRoot "src\g1_rl\tasks\task1\task1_env.py"
$TrainPy = Join-Path $ProjectRoot "src\g1_rl\tasks\task1\task1_train.py"
$EvalPy = Join-Path $ProjectRoot "src\g1_rl\tasks\task1\task1_model_test.py"
$ModelPy = Join-Path $ProjectRoot "src\g1_rl\common\g1_skrl_models.py"
$InfoPy = Join-Path $ProjectRoot "src\g1_rl\common\info_utils.py"
$PathsPy = Join-Path $ProjectRoot "src\g1_rl\common\paths.py"

Write-Host ""
Write-Host "============================================================"
Write-Host "G1 Task1 Windows non-training readiness check"
Write-Host "============================================================"
Write-Host "ProjectRoot  = $ProjectRoot"
Write-Host "IsaacLabRoot = $IsaacLabRoot"
Write-Host "PythonBat    = $PythonBat"
Write-Host "G1UsdPath    = $G1UsdPath"
Write-Host "MotionFile   = $MotionFile"
Write-Host "TrainPy      = $TrainPy"
Write-Host "EvalPy       = $EvalPy"
Write-Host "============================================================"

$required = @(
    $ProjectRoot,
    $IsaacLabRoot,
    $PythonBat,
    $ConfigPy,
    $EnvPy,
    $TrainPy,
    $EvalPy,
    $ModelPy,
    $InfoPy,
    $PathsPy,
    $G1UsdPath,
    $MotionFile
)

$missing = @()
foreach ($p in $required) {
    if (-not (Test-Path $p)) {
        $missing += $p
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "[WARN] Missing required paths:" -ForegroundColor Yellow
    foreach ($m in $missing) {
        Write-Host "  - $m" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Please copy project / assets to Windows path or pass parameters explicitly:"
    Write-Host "  .\scripts\windows\check_task1_windows_ready.ps1 -ProjectRoot <path> -IsaacLabRoot <path> -G1UsdPath <path> -MotionFile <path>"
    exit 1
}

$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"
$env:RT_G1_TASK1_LOG_ROOT = "G:\rt_isaaclab_ws\logs\unitree_g1_isaaclab_rl\task1"
$env:G1_USD_PATH = $G1UsdPath
$env:G1_TASK1_MOTION_FILE = $MotionFile

Write-Host ""
Write-Host "[CHECK] Python / torch / isaaclab / skrl import check..."
& $PythonBat -c "import sys; print('[PYTHON]', sys.executable); import torch; print('[TORCH]', torch.__version__, 'cuda=', torch.cuda.is_available()); import isaaclab; print('[ISAACLAB] ok'); import skrl; print('[SKRL]', getattr(skrl, '__version__', 'unknown'))"

if ($LASTEXITCODE -ne 0) {
    throw "Python environment check failed."
}

Write-Host ""
Write-Host "[CHECK] Project import check..."
& $PythonBat -c "import sys, os; sys.path.insert(0, r'$ProjectRoot\src'); from g1_rl.tasks.task1.task1_config import Task1Config; cfg=Task1Config(); cfg.usd_path=r'$G1UsdPath'; cfg.motion_file=r'$MotionFile'; cfg.validate(); print('[G1 Task1Config] ok'); print('[OBS]', cfg.num_observations, '[ACT]', cfg.num_actions, '[STACK]', cfg.stacked_obs_dim)"

if ($LASTEXITCODE -ne 0) {
    throw "Project import check failed."
}

Write-Host ""
Write-Host "[OK] G1 Task1 Windows non-training readiness check passed."
Write-Host "No training has been launched."
