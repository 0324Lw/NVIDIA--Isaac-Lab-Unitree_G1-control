# 🧍 基于 NVIDIA Isaac Lab 的 Unitree G1 人形机器人纯 RL 控制项目
 
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.x-brightgreen)
![skrl](https://img.shields.io/badge/RL-skrl%20PPO%20%7C%20RMA-purple)
![OS](https://img.shields.io/badge/OS-Ubuntu%20%7C%20Windows-green)
 
本项目是一个基于 NVIDIA Isaac Lab 的 Unitree G1 人形机器人强化学习控制项目。项目包含 4 个递进任务：基础低速行走、全向速度跟踪、全身协同行走、Sim2Real / RMA 鲁棒训练。
 
这个仓库是我在学习人形机器人强化学习控制过程中整理出来的一版纯 RL baseline,复杂人形机器人动作不能只依赖手写奖励和纯强化学习，专业路线通常需要动捕数据、重定向、模仿学习、生成式动作先验和 Sim2Real 体系。因此，这个仓库更适合作为一个早期探索版、学习版、纯 RL baseline 保存下来，希望能为在学习 Isaac Lab、人形机器人控制和强化学习的同学提供一个可以参考、可以运行、可以继续修改的基础工程。项目重点不是追求完美动作效果，而是尽量把每个任务的环境、测试、训练、评估和日志拆清楚。代码中仍然有很多可以继续改进的地方，欢迎大家根据自己的 Isaac Lab 版本、显卡配置和研究目标继续修改。
 
---
 
## 🎬 训练效果展示
 
| Scene | Preview |
|---|---|
| 基础行走 / 全向运动 | ![G1 locomotion demo](assets/gifs/g1_locomotion_demo.gif) |
| 全身协同 / Sim2Real 鲁棒训练 | ![G1 whole-body sim2real demo](assets/gifs/g1_whole_body_sim2real_demo.gif) |
 
---
 
## ✨ 项目特点
 
- 基于 NVIDIA Isaac Lab 和 Unitree G1 人形机器人资产。
- 包含 4 个递进任务，从低速行走到全向运动、全身协同和 Sim2Real 鲁棒训练。
- Task1 / Task2 / Task3 使用 `skrl` PPO 训练流程。
- Task4 使用独立 RMA PPO 训练流程，包含 teacher / student latent、privileged observation 和 student-only 部署模型导出。
- 所有 G1 任务环境代码独立实现，不依赖其他任务环境继承，避免任务之间互相污染。
- 每个任务提供独立环境测试、训练脚本和模型测试脚本。
- 使用 `tqdm` 风格训练进度条，方便实时查看训练速度、奖励、摔倒率和关键遥测指标。
 
---
 
## 📁 项目结构
 
```text
unitree_g1_isaaclab_rl/
├── configs/
│   ├── task1_locomotion.yaml
│   ├── task2_omni_locomotion.yaml
│   ├── task3_whole_body.yaml
│   └── task4_sim2real.yaml
├── src/
│   └── g1_rl/
│       ├── common/
│       │   ├── g1_eval_utils.py
│       │   ├── g1_skrl_models.py
│       │   ├── g1_skrl_wrappers.py
│       │   ├── info_utils.py
│       │   └── paths.py
│       └── tasks/
│           ├── task1/
│           │   ├── task1_config.py
│           │   ├── task1_env.py
│           │   ├── task1_train.py
│           │   └── task1_model_test.py
│           ├── task2/
│           │   ├── task2_config.py
│           │   ├── task2_env.py
│           │   ├── task2_train.py
│           │   └── task2_model_test.py
│           ├── task3/
│           │   ├── task3_config.py
│           │   ├── task3_env.py
│           │   ├── task3_train.py
│           │   └── task3_model_test.py
│           └── task4/
│               ├── task4_config.py
│               ├── task4_env.py
│               ├── task4_train.py
│               └── task4_model_test.py
├── tests/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
├── scripts/
│   └── ubuntu/
├── logs/
├── assets/
│   ├── gifs/
│   └── images/
├── LICENSE
└── README.md
```
 
| 目录 | 说明 |
|---|---|
| `configs/` | 每个任务的配置文件，便于统一管理任务参数。 |
| `src/g1_rl/common/` | 通用网络模型、评估工具、日志工具、路径工具和 frame stack wrapper。 |
| `src/g1_rl/tasks/taskX/` | 每个任务的配置、环境、训练脚本和模型测试脚本。 |
| `tests/` | 每个任务的环境测试脚本。 |
| `scripts/ubuntu/` | Ubuntu 下的测试、训练、评估脚本。 |
| `logs/` | 默认训练日志和 checkpoint 输出目录。 |
| `assets/` | README 图片、GIF 和展示素材。 |
 
---
 
## 🛠️ 建议硬件与系统配置
 
### 最低测试配置
 
用于环境测试、smoke training、低并发调试和模型测试：
 
- Ubuntu 22.04 / 24.04
- NVIDIA GPU，建议显存 16GB 以上
- Python 3.11
- PyTorch 2.x
- Isaac Sim / Isaac Lab
- `skrl`, `tensorboard`, `tqdm`, `numpy`
 
在显存较小的设备上，建议从很小的并发开始：
 
```bash
--num-envs 1
--num-envs 4
--num-envs 8
--num-envs 16
```
 
### 推荐训练配置
 
用于较大规模训练和长时间实验：
 
- NVIDIA RTX 3090 / 4090 或同级别 GPU
- 显存 24GB 或更高
- Ubuntu 环境优先
- Isaac Lab 环境能够稳定运行
 
较大显存设备可以逐步尝试：
 
```bash
--num-envs 128
--num-envs 256
--num-envs 512
```
 
人形机器人比四足机器人更容易出现仿真不稳定、显存占用高、训练震荡等问题。不要一开始直接使用最大并发，建议先运行环境测试和 smoke training。
 
---
 
## 🚀 基础准备
 
### 1. 安装 Isaac Lab 环境
 
请先按照 NVIDIA Isaac Lab 官方文档安装 Isaac Sim / Isaac Lab，并确认 Isaac Lab 的 Python 环境可以正常导入：
 
```bash
python -c "import isaaclab; print('isaaclab ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
 
### 2. 克隆项目
 
```bash
git clone https://github.com/0324Lw/NVIDIA--Isaac-Lab-Unitree_G1-control.git unitree_g1_isaaclab_rl
cd unitree_g1_isaaclab_rl
```
 
### 3. 设置 PYTHONPATH
 
```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```
 
也可以直接使用 `scripts/ubuntu/` 下的脚本，这些脚本会自动设置项目路径。
 
### 4. 设置 G1 资产与 motion 文件路径
 
根据你的本地 Isaac Lab 路径设置：
 
```bash
export G1_USD_PATH="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd"
export G1_TASK1_MOTION_FILE="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_walk.pt"
export G1_TASK2_MOTION_FILE="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_omni_walk.pt"
export G1_TASK3_MOTION_FILE="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_whole_body_walk.pt"
export G1_TASK4_MOTION_FILE="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_omni_walk.pt"
```
 
说明：
 
- `g1.usd` 是 Unitree G1 机器人模型。
- `g1_walk.pt` 可用于 Task1 的基础行走参考。
- `g1_omni_walk.pt` 可用于 Task2 全向运动和 Task4 鲁棒训练。
- `g1_whole_body_walk.pt` 用于 Task3，全身参考中需要包含 `arm_swing_ref` 字段。
- 这些 `.pt` motion 文件通常较大，建议不要上传到 GitHub。
 
### 5. 安装 Python 依赖
 
在 Isaac Lab 对应的 Python 环境中安装必要依赖：
 
```bash
pip install skrl tensorboard tqdm numpy
```
 
如果你的 Isaac Lab 环境已经包含部分依赖，可以按需跳过。
 
---
 
## ⚡ 快速开始
 
### 1. 环境测试
 
建议先从 Task1 开始测试，再进入后续任务。
 
```bash
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/test_task2_env.sh
bash scripts/ubuntu/test_task3_env.sh
bash scripts/ubuntu/test_task4_env.sh
```
 
如果显存不足，可以打开对应脚本，降低 `--num-envs`。
 
### 2. Smoke 训练
 
Smoke training 只用于确认训练管线可以启动、日志可以写入、checkpoint 可以保存，不用于评估最终效果。
 
```bash
bash scripts/ubuntu/train_task1_skrl_smoke.sh
bash scripts/ubuntu/train_task2_skrl_smoke.sh
bash scripts/ubuntu/train_task3_skrl_smoke.sh
bash scripts/ubuntu/train_task4_rma_smoke.sh
```
 
### 3. 模型测试
 
训练完成后，可以使用 eval 脚本加载 checkpoint 做推理测试。
 
```bash
bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run_name>/final_checkpoint/g1_task1_model.pt 1.0
bash scripts/ubuntu/eval_task2_skrl.sh logs/task2/<run_name>/final_checkpoint/g1_task2_omni_model.pt 1.0
bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run_name>/final_checkpoint/g1_task3_whole_body_model.pt 1.0
bash scripts/ubuntu/eval_task4_rma.sh logs/task4/<run_name>/final_checkpoint 1.0
```
 
---
 
## 🧩 任务设计总览
 
| Task | 目标 | 环境特点 | 训练重点 | 主要脚本 |
|---|---|---|---|---|
| Task1 | 基础低速行走 | G1 23 DoF 控制、基础 motion 参考、低速命令课程 | 稳定站立、低速前进、速度跟踪 | `task1_env.py`, `task1_train.py`, `task1_model_test.py` |
| Task2 | 全向速度跟踪 | 前进、后退、侧向、转向命令；无 world 文件 | 全向移动、低速稳定、从 Task1 过渡 | `task2_env.py`, `task2_train.py`, `task2_model_test.py` |
| Task3 | 全身协同行走 | 上肢摆臂参考、arm action gain 课程、whole-body reward | 腿部运动与手臂摆动协同 | `task3_env.py`, `task3_train.py`, `task3_model_test.py` |
| Task4 | Sim2Real / RMA 鲁棒训练 | action delay、obs delay、motor efficiency、payload、push、privileged obs | 低速抗扰、teacher/student latent、student-only 部署 | `task4_env.py`, `task4_train.py`, `task4_model_test.py` |
 
---
 
## ➡️ Task 1：基础低速行走
 
Task1 是最基础的人形机器人 locomotion 任务，用于让 Unitree G1 在平地上保持稳定姿态，并尝试低速前进和速度跟踪。
 
### 任务目标
 
- G1 在平地上保持站立和低速行走。
- 跟踪简单线速度命令。
- 学习基础步态，为 Task2 / Task3 提供可 warm-start 的 checkpoint。
- 保持 23 DoF 动作空间和 123 维单帧观测结构。
 
### 环境设计
 
- 使用 Isaac Lab 中的 Unitree G1 USD 资产。
- 动作输出为 23 个受控关节的目标位置残差。
- 两个传感器关节 `xl330_joint` 和 `d455_joint` 固定，不参与策略控制。
- 单帧 actor observation 为 123 维，5 帧堆叠后 policy input 为 615 维。
- 训练代码使用 `skrl` PPO。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/train_task1_skrl_smoke.sh
bash scripts/ubuntu/train_task1_skrl_laptop.sh
bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run_name>/final_checkpoint/g1_task1_model.pt 1.0
```
 
### 训练时重点观察
 
- `Actual_Vx` 是否逐步接近 `Cmd_Vx`
- `Base_Height` 是否稳定在目标高度附近
- `Fall_Rate` 是否接近 0
- `Contact_Count` 是否合理
- `P_Foot_Slip` 是否过大
- PPO 的 `approx_kl`、`clip_fraction` 是否稳定
 
---
 
## ➡️ Task 2：全向速度跟踪
 
Task2 在 Task1 的基础上增加全向速度命令，让 G1 不只学习向前走，还要学习低速后退、侧向移动和转向。
 
### 任务目标
 
- 学习低速全向 locomotion。
- 支持前进、后退、侧向和 yaw turning。
- 为 Task3 whole-body 协同和 Task4 Sim2Real 鲁棒训练提供基础运动能力。
 
### 环境设计
 
- Task2 没有 world 文件，所有逻辑都在独立环境中实现。
- 使用 `g1_omni_walk.pt` 作为简单 synthetic / reference motion。
- 观测维度保持 123，5 帧堆叠后为 615。
- 动作维度保持 23。
- 使用 `skrl` PPO 训练。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task2_env.sh
bash scripts/ubuntu/train_task2_skrl_smoke.sh
bash scripts/ubuntu/train_task2_skrl_laptop.sh
bash scripts/ubuntu/eval_task2_skrl.sh logs/task2/<run_name>/final_checkpoint/g1_task2_omni_model.pt 1.0
```
 
### 训练时重点观察
 
- `Cmd_Vx` / `Actual_Vx`
- `Cmd_Vy` / `Actual_Vy`
- `Cmd_Wz` / `Actual_Wz`
- `Lin_Error`
- `Yaw_Error`
- `Fall_Rate`
- `Base_Height`
 
---
 
## ➡️ Task 3：全身协同行走
 
Task3 在 Task2 全向运动基础上加入上肢动作，目标是让 G1 在低速运动时尝试形成更自然的全身协同，而不是只控制双腿。
 
### 任务目标
 
- 在低速 locomotion 的基础上引入上肢摆臂。
- 使用 `arm_swing_ref` 提供手臂参考。
- 使用 `Arm_Action_Gain` 课程逐步解冻上肢动作。
- 尝试训练腿部运动和手臂摆动之间的同步。
 
### 环境设计
 
Task3 仍然没有 world 文件，重点在环境内部的 whole-body reward 和 motion manager：
 
- `G1WholeBodyMotionManager` 读取 `g1_whole_body_walk.pt`。
- motion 文件需要包含 `pos`、`vel`、`cmd`、`phase`、`contact_ref`、`mode_id`、`mode_names`、`joint_names`、`arm_swing_ref` 等字段。
- 上肢 action gain 随课程从 0 逐步增加到 1。
- 奖励中包含 `R_Arm_Ref`、`R_Arm_Vel_Ref`、`R_Arm_Leg_Sync`、`R_Arm_Cross` 等 whole-body 项。
- 使用 `skrl` PPO 训练，建议从 Task2 checkpoint warm-start。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task3_env.sh
bash scripts/ubuntu/train_task3_skrl_smoke.sh
bash scripts/ubuntu/train_task3_skrl_laptop.sh
bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run_name>/final_checkpoint/g1_task3_whole_body_model.pt 1.0
```
 
### 训练时重点观察
 
- `Fall_Rate`
- `Base_Height`
- `Cmd_Vx` / `Actual_Vx`
- `Cmd_Wz` / `Actual_Wz`
- `Arm_Action_Gain`
- `Style_Scale`
- `R_Arm_Ref`
- `R_Arm_Leg_Sync`
 
Task3 的动作效果不应被理解为专业的人形机器人舞蹈或武术控制。它只是一个纯 RL whole-body baseline，用于学习人形机器人全身奖励设计的难点。
 
---
 
## ➡️ Task 4：Sim2Real / RMA 鲁棒训练
 
Task4 面向低速 Sim2Real 鲁棒控制。它不追求复杂动作，而是测试 G1 在动作延迟、观测延迟、电机效率变化、传感器噪声、负载、摩擦变化和外部推力下是否仍能保持基础运动稳定。
 
### 任务目标
 
- 在低速命令下保持站立和运动稳定。
- 引入 Sim2Real 域随机化和扰动机制。
- 使用 privileged observation 训练 teacher / latent 表征。
- 导出 student-only 部署模型，推理时只使用 actor observation。
 
### 环境设计
 
Task4 是独立环境，不继承 Task1 / Task2 / Task3：
 
- actor observation：123 维
- privileged observation：162 维
- frame stack：5
- stacked actor observation：615 维
- action dimension：23
 
Sim2Real 随机化包括：
 
- motor efficiency randomization
- actuator lag / alpha scale
- action delay
- observation delay
- action deadzone
- action noise
- action quantization
- payload force proxy
- friction / slip stress proxy
- IMU / joint / height / foot noise
- state dropout
- contact dropout / false positive
- external push disturbance
 
训练代码是独立 RMA PPO，不使用 `skrl`。训练后会保存两个文件：
 
```text
g1_task4_rma_full_checkpoint.pt
g1_task4_student_deploy.pt
```
 
其中 `g1_task4_student_deploy.pt` 是后续部署和模型测试优先使用的 student-only 模型。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task4_env.sh
bash scripts/ubuntu/train_task4_rma_smoke.sh
bash scripts/ubuntu/train_task4_rma_laptop.sh
bash scripts/ubuntu/eval_task4_rma.sh logs/task4/<run_name>/final_checkpoint 1.0
```
 
### 训练时重点观察
 
- `DR_Scale`
- `Cmd_Vx` / `Actual_Vx`
- `Fall_Rate`
- `Base_Height`
- `Push_Active_Rate`
- `Motor_Eff_Mean`
- `Action_Delay`
- `Obs_Delay`
- `Payload_Mass`
- `Friction_Proxy`
- `approx_kl`
- `teacher_ratio`
 
---
 
## 📊 日志与模型保存
 
训练日志默认保存在：
 
```text
logs/task1/
logs/task2/
logs/task3/
logs/task4/
```
 
每个训练 run 通常包含：
 
```text
checkpoint_<env_steps>/
final_checkpoint/
train_metadata.pt
```
 
Task4 还会额外保存：
 
```text
g1_task4_rma_full_checkpoint.pt
g1_task4_student_deploy.pt
```
 
可以使用 TensorBoard 查看训练过程：
 
```bash
tensorboard --logdir logs
```
 
训练过程中会记录以下类型的信息：
 
- `reward_components`：各奖励项。
- `events`：摔倒、超时、推力扰动等事件。
- `telemetry`：速度、高度、课程阶段、DR scale 等训练指标。
- `debug`：观测维度、reward 范围、异常值检查等。
- `ppo` / `rma`：PPO 更新信息、KL、loss、学习率、teacher ratio 等。
 
---
 
## 💻 Ubuntu 使用说明
 
当前仓库以 Ubuntu / Isaac Lab 环境为主。常用脚本在：
 
```text
scripts/ubuntu/
```
 
推荐顺序是：
 
```bash
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/train_task1_skrl_smoke.sh
bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run_name>/final_checkpoint/g1_task1_model.pt 1.0
```
 
后续任务同理，先测试环境，再 smoke training，再进行长训练和模型测试。
 
Windows 训练可以参考机器狗项目中的 Windows runner 方式移植，但本仓库 README 以 Ubuntu 为主。如果你在 Windows 上运行，需要根据本机的 Isaac Lab 路径、Python 路径、项目路径和显卡状态修改脚本。
 
---
 
## 🧭 推荐训练顺序
 
推荐顺序：
 
1. 先训练 Task1，获得基础低速行走 checkpoint。
2. Task2 从 Task1 warm-start，训练全向速度跟踪。
3. Task3 从 Task2 warm-start，训练上肢摆臂和全身协同。
4. Task4 使用独立 RMA PPO 从零训练，重点测试低速 Sim2Real 鲁棒性。
 
也可以每个任务从零开始训练，但人形机器人从零训练难度很高，早期 reward 容易陷入局部最优，调参成本会更大。
 
---
 
## 📌 当前状态与限制
 
- 本项目主要用于学习、复现实验和开源交流。
- 当前代码完成了四个任务的 Isaac Lab 环境、环境测试、训练脚本和模型测试脚本。
- 这个仓库是 pure-RL baseline，不代表专业人形机器人动作控制最终路线。
- 复杂舞蹈、武术、拟人动作更适合结合动捕数据、重定向、模仿学习、HoloSoma、OmniRetarget、BeyondMimic 等技术路线。
- Task3 的 whole-body reward 只能提供简单上肢协同约束，不等同于高质量动作模仿。
- Task4 的 RMA 结构是学习版实现，后续仍可继续扩展 adaptation module、部署接口和真机安全检查。
- 不同 Isaac Lab / Isaac Sim 版本之间可能存在 API 差异，需要根据本地环境做少量适配。
- 训练效果会受到 GPU、并发数、随机种子、训练步数和超参数影响。
- 本项目不是官方 Unitree 或 NVIDIA 项目，只是个人学习和开源整理。
 
---
 
## ❓ 常见问题
 
### 1. `ModuleNotFoundError: No module named torch`
 
通常是没有进入 Isaac Lab 对应的 Python / conda 环境。请先确认：
 
```bash
which python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
 
### 2. Isaac Lab / `pxr` 导入报错
 
涉及 Isaac Lab、USD、`pxr` 的文件需要在 Isaac Sim / Isaac Lab 环境中运行。测试脚本中如果需要 AppLauncher，应保证先启动 AppLauncher，再导入依赖 Isaac Lab 的环境文件。
 
### 3. 训练启动后显存不足怎么办？
 
先降低并发数：
 
```bash
--num-envs 1
--num-envs 4
--num-envs 8
--num-envs 16
--num-envs 32
```
 
确认能跑通后再逐步增加。
 
### 4. Smoke training 的效果不好正常吗？
 
正常。Smoke training 只用于检查训练流程是否能启动和保存模型，不代表最终策略效果。
 
### 5. 为什么 G1 的纯 RL 训练比 Go2 难很多？
 
人形机器人自由度更高、接触更复杂、稳定区域更小，奖励函数也更难设计。纯 RL 可以作为学习 baseline，但复杂拟人动作通常需要结合模仿学习、动捕数据、重定向和更完整的 Sim2Real 流程。
 
### 6. 为什么 Task4 不使用 skrl？
 
Task4 当前保留了原始 RMA 训练思路，使用独立 PPO 实现 teacher / student latent、privileged observation 和 student-only deploy。这样更方便展示 RMA 的结构，也避免把 Task4 强行改成普通 skrl PPO 后丢失原始代码中的学习价值。
 
### 7. 为什么要先跑环境测试？
 
人形机器人训练中的很多问题不是 PPO 本身造成的，而是 reset、观测维度、关节映射、传感器关节、接触检测、奖励项或终止条件有问题。先跑测试可以减少后续训练调参的时间。
 
### 8. 这个项目能直接真机部署吗？
 
不能直接保证。这个仓库目前是 Isaac Lab 仿真学习 baseline。真机部署还需要安全限幅、低层控制接口、状态估计、延迟测试、动力学参数校准、实机保护逻辑和大量 Sim2Real 验证。
 
---
 
## 📄 License
 
This project is released under the MIT License.
 
See the `LICENSE` file for details.
 
---
 
## 🙏 Acknowledgements
 
感谢以下开源项目和工具：
 
- NVIDIA Isaac Sim / Isaac Lab
- Unitree G1 robot asset and related documentation
- PyTorch
- skrl reinforcement learning library
- TensorBoard
- tqdm
- 机器人强化学习、模仿学习和 Isaac Lab 开源社区
 
如果这个项目对你有帮助，欢迎参考、修改和继续完善。也欢迎指出代码或文档中的问题。

联系邮箱：2559906288@qq.com  
小红书账号：574661219

