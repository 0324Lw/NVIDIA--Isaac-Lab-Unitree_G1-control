param(
    [string]$ProjectRoot = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl",
    [string]$IsaacLabRoot = "G:\rt_isaaclab_ws\repos\IsaacLab_v2.3.2",
    [string]$G1UsdPath = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl\assets\robots\g1.usd",
    [string]$MotionFile = "G:\rt_isaaclab_ws\projects\unitree_g1_isaaclab_rl\assets\motions\g1_omni_walk.pt",
    [string]$Task1Pretrained = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PythonBat = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"

$ConfigPy = Join-Path $ProjectRoot "src\g1_rl\tasks\task2\task2_config.py"
$EnvPy = Join-Path $ProjectRoot "src\g1_rl\tasks\task2\task2_env.py"
$TrainPy = Join-Path $ProjectRoot "src\g1_rl\tasks\task2\task2_train.py"
$EvalPy = Join-Path $ProjectRoot "src\g1_rl\tasks\task2\task2_model_test.py"
$ModelPy = Join-Path $ProjectRoot "src\g1_rl\common\g1_skrl_models.py"
$WrapperPy = Join-Path $ProjectRoot "src\g1_rl\common\g1_skrl_wrappers.py"

Write-Host ""
Write-Host "============================================================"
Write-Host "G1 Task2 Windows non-training readiness check"
Write-Host "============================================================"
Write-Host "ProjectRoot     = $ProjectRoot"
Write-Host "IsaacLabRoot    = $IsaacLabRoot"
Write-Host "PythonBat       = $PythonBat"
Write-Host "G1UsdPath       = $G1UsdPath"
Write-Host "MotionFile      = $MotionFile"
Write-Host "Task1Pretrained = $Task1Pretrained"
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
    $WrapperPy,
    $G1UsdPath,
    $MotionFile
)

if (-not [string]::IsNullOrWhiteSpace($Task1Pretrained)) {
    $required += $Task1Pretrained
}

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
    exit 1
}

$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"
$env:RT_G1_TASK2_LOG_ROOT = "G:\rt_isaaclab_ws\logs\unitree_g1_isaaclab_rl\task2"
$env:G1_USD_PATH = $G1UsdPath
$env:G1_TASK2_MOTION_FILE = $MotionFile
if (-not [string]::IsNullOrWhiteSpace($Task1Pretrained)) {
    $env:G1_TASK1_PRETRAINED = $Task1Pretrained
}

Write-Host ""
Write-Host "[CHECK] Python / torch / isaaclab / skrl import check..."
& $PythonBat -c "import sys; print('[PYTHON]', sys.executable); import torch; print('[TORCH]', torch.__version__, 'cuda=', torch.cuda.is_available()); import isaaclab; print('[ISAACLAB] ok'); import skrl; print('[SKRL]', getattr(skrl, '__version__', 'unknown'))"

if ($LASTEXITCODE -ne 0) {
    throw "Python environment check failed."
}

Write-Host ""
Write-Host "[CHECK] Project import check..."
& $PythonBat -c "import sys; sys.path.insert(0, r'$ProjectRoot\src'); from g1_rl.tasks.task2.task2_config import Task2Config; cfg=Task2Config(); cfg.usd_path=r'$G1UsdPath'; cfg.motion_file=r'$MotionFile'; cfg.validate(); print('[G1 Task2Config] ok'); print('[OBS]', cfg.num_observations, '[ACT]', cfg.num_actions, '[STACK]', cfg.stacked_obs_dim); print('[MOTION]', cfg.motion_file)"

if ($LASTEXITCODE -ne 0) {
    throw "Project import check failed."
}

Write-Host ""
Write-Host "[OK] G1 Task2 Windows non-training readiness check passed."
Write-Host "No training has been launched."
