# Phase 5B.4：视觉 BC 在线推理与闭环评测

> 历史实验记录；本机路径已脱敏，诊断工具路径已更新。原始结果保留在本机，数据、权重和配图未随仓库发布。当前安装与入口以[项目 README](../../README.md)为准。

2026-10-15，项目 `<project>`。

**工程正确性：通过。** 原始 E5 模型确实在 RTX 5090 D v2 上持续推理，驱动原生 LIBERO、记录轨迹并正确结束。

**算法有效性：在本次固定5个官方初态中成功3次（60%）。** 两次失败均为300步超时。结果仅覆盖炉灶单任务和这5个初态，不能外推整个 LIBERO Benchmark；这些初态也未被证明与训练/验证示范一一对应，不能称为严格的未见初态泛化率。

## 1. 实际代码变动

| 文件 | 职责 |
|---|---|
| 新增 `src/embodiedlab/policies/bc.py` | 60行推理桥接：严格加载checkpoint、共享预处理、设备迁移、动作解码与耗时记录 |
| 修改 `src/embodiedlab/runners/rollout.py` | 可选只读`on_step`回调，接收动作执行后的Observation、已发出动作的副本和Info |
| 修改 `src/embodiedlab/evaluation.py` | `evaluate_task`接收现有Policy对象，逐次reset；增加首次成功步和运行耗时 |
| 新增 `scripts/evaluate_bc.py` | 固定炉灶协议，预先保存初态/种子/权重指纹，复用Evaluator、Runner与Recorder |
| 新增 `tests/test_bc_policy.py` | 原始模型及真实观测一致性、设备回退、错误输入、原生三步闭环等11项测试 |
| 修改 `README.md` | 最小API、运行命令、输出与结果说明 |
| 新增本报告 | 实验结果及局限 |

未修改VisualBC结构、共享预处理、Dataset、Recorder、Replay、LiberoEnv、原生控制器、奖励或成功判定。没有训练、微调或保存替代模型。

## 2. 推理与记录流程

`BCPolicy(best.pt)`先在CPU严格恢复全部`state_dict`（含`state_mean/state_std`），检查格式、架构、图像约定、对齐约定、夹爪类别与参数有效性，然后迁移到设备，执行`eval()`并冻结梯度。

一次Episode依次调用：`evaluate_task → run_episode → env.reset(seed, init_state_index) → policy.reset → recorder.start`；之后反复执行`policy.act(obs) → env.step(action) → recorder.append → on_step`，直到原生成功或300步截断。Recorder在`finally`关闭，环境由`with`关闭。

每次`act`内部调用既有`preprocess_observation()`：

- Agentview、Wrist顺序保持；HWC uint8 RGB转CHW float32 `[0,1]`，不旋转、不翻转、不增强。
- 状态为8维：末端xyz（m）、轴角（rad）、两指qpos（m），只由模型内部已恢复的训练统计量标准化。
- Tensor加batch维并迁移到设备，在`torch.inference_mode()`下Forward。
- 前6维保持Tanh输出；夹爪`argmax`解码为`-1=open / +1=close`。返回有限、范围`[-1,1]`的NumPy float64 `(7,)`。

动作是原生**OSC_POSE相对增量**，不是七关节角度。前三维世界坐标平移缩放为每次最多0.05 m，后三维轴角增量每分量最多0.5 rad，由原控制器左乘当前旋转矩阵设置目标；不额外缩放、平滑或裁剪。20 Hz控制频率保持不变。

原官方训练配对仍为`obs[t-1] → action[t]`。本次在线自录为标准转移：`obs[k] → action[k] → obs[k+1]`，T个动作对应T+1个观测，不能再对这些自录数据套用官方移位规则。

旋钮角度、接触及任务成功诊断仅在环境步后读取，**不送入模型**。两路真实RGB、状态、实际动作、奖励与结束标志由原HDF5 Recorder保存；逐步JSON保存推理延迟及旋钮/接触信息。接触仅在20 Hz控制步末采样，不能解释为所有物理子步接触或力峰值。

## 3. Checkpoint与设备验证

- 原模型：`training_results/phase5b3_train_seed42/best.pt`，epoch **5**，参数194,920。
- SHA256：`00e37e0be92296c2a9a2753d83320768e62165cfbf6526196c860b79a405978b`，实验前后相同。
- GPU：NVIDIA GeForce RTX 5090 D v2，计算能力12.0；驱动580.173.02；PyTorch2.9.1+cu128；原生MuJoCo3.3.2、robosuite1.4.0、hf-libero0.1.4。
- 默认`device=auto`优先CUDA，可显式选择CPU。Codex沙箱内隐藏GPU，实际GPU实验使用普通主机用户的沙箱外执行，没有sudo或驱动改动。
- 3个官方观测和3个真实在线自录观测：Policy与同设备直接离线Forward的解码动作最大差 **0**。
- 原CPU Torch2.5.1与GPU输出的最大动作差 **1.19e-7**，夹爪解码全部相同。
- 额外原生三步测试直接检查每次在线观测的Policy/离线Forward一致性。全部权重与归一化buffer逐元素正确恢复且推理后不变，无梯度，原始观测未修改。

官方HDF5实验前后SHA256均为`387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5`。没有修改原始数据、原生任务、环境依赖或既有训练结果。

## 4. 固定协议与逐条结果

任务：`libero_goal / task_id=7 / turn_on_the_stove`，128×128双RGB。原生hard reset、官方初态、10步开夹settling；策略时限300步（15 s），settling不计入策略步数。未恢复专家历史状态，也未播放专家动作。

先单独运行Smoke：初态0、seed42，第225步原生成功；**不计入正式5条分母**。随后评测预先写入`protocol.json`的初态0～4、seed42～46，未根据结果挑选初态。

| 初态 | Seed | 结果 | 首次成功步 | 长度 | 仿真时间s | 旋钮最终/最高rad | 采样旋钮接触步数 |
|---:|---:|---|---:|---:|---:|---|---:|
| 0 | 42 | 成功，terminated | 225 | 225 | 11.25 | 0.500549 / 0.500549 | 48 |
| 1 | 43 | 超时，truncated | — | 300 | 15.00 | 0.428485 / 0.428485 | 32 |
| 2 | 44 | 超时，truncated | — | 300 | 15.00 | 0.011029 / 0.109022 | 47 |
| 3 | 45 | 成功，terminated | 113 | 113 | 5.65 | 0.516622 / 0.516622 | 59 |
| 4 | 46 | 成功，terminated | 109 | 109 | 5.45 | 0.510200 / 0.510200 | 59 |

原生判定为`flat_stove_1_button`关节角度≥0.5 rad；源码为固定版本`libero/libero/envs/objects/articulated_objects.py:241-251`，通过原任务Turnon谓词调用。没有自行增加或放宽成功条件，也不把“接触过”当成功。初态0在最后一次采样旋钮接触后才跨过阈值；该成功有效，但不代表持续稳定抓握或保持。

正式BC共1047个真实模型动作，成功率**3/5**，平均长度209.4步，程序总耗时**18.49 s**（含初始化、记录、预览输出）。相同初态/种子的Zero Policy全部300步超时，**0/5**，旋钮均为0，耗时20.56 s；它只是接口参照，不是强算法Baseline。

Formal推理`act()`耗时包含预处理、传输、Forward、解码和同步取回：平均**1.577 ms**，中位数**1.378 ms**，P95 **1.752 ms**；首次冷启动最大**156.734 ms**。不能据此声称所有推理均小于一个50 ms控制周期。Smoke与正式初态0的动作、状态和关节数值最大差均为0。

本机保留图片（未随仓库发布）：正式轨迹及失败行为，`evaluation_results/phase5b4/20261015T085852Z_8dde3e4f/rollout_analysis.png`。

## 5. 失败行为与证据边界

**初态1：接触后推进不足，后期离开操作区。** 第55步首次采样旋钮接触，第73步开始关夹，第209步最后采样到旋钮接触；最终0.428485 rad，距阈值0.071515 rad。132～170步末端累计平移仅6.00 mm，但仍累计旋转1.2196 rad；最后100步末端路径0.594 m，因此不能称整机卡死。第284步开夹，晚于主要接触阶段，不能仅凭它断言失败原因。

**初态2：接触后退转，未形成持续有效操作。** 第55步首次接触，第76步开始关夹；205步旋钮达0.109022 rad，224步回到约−0.005867 rad，最终0.011029 rad。退转阶段始终输出+1关夹，第297步才开夹，因此末尾开夹不能解释先前退转。末尾存在297/299步开关切换，但不是已证实的主要原因。

两条失败都能接近并实际接触目标；图像和数值显示后期离开目标、姿态变化较大，并非一直重复零动作。位置运动小也不等于机器人完全停滞。成功3/4均在第55步关夹，而失败1/2较晚，但成功0也到75步才关夹，故不能把关夹早晚直接当因果结论。

**闭环分布偏移有证据，唯一根因尚未确定。** 只用训练40条轨迹的3539个监督输入作为参照：在线8维标准化状态最近邻距离中位数，成功0/失败1/失败2/成功3/成功4依次为3.84/5.04/4.44/0.94/0.87。两失败偏离较大，但成功0也是反例，偏离不是充分失败条件。这只是数值状态支持诊断，没有证明视觉分布偏移。

失败1/2途中与最近训练姿态的SO(3)角距离最大约2.495/2.500 rad，支持明显姿态偏移。另一个问题是原始轴角表征：初态1末11个输入与最近训练旋转的轴角向量距离约6.27～6.34 rad，实际SO(3)差仅0.127～0.178 rad；模型确实接收了这种表征偏移，但不能将其称为同等大小的物理姿态错误。没有检测到大于π的单步轴角跳变，本阶段也没有更改该表征或模型输入。

目前最强证据指向**接触阶段的闭环姿态/动作稳定性和恢复不足**。预处理不一致、动作维度错误已被本次验证排除；视觉感知、连续动作误差、夹爪决策各自的因果贡献尚未隔离。未做图像消融、状态干预或额外模型训练，不能宣称视觉编码器是根因。

失败预览上下两行为Agentview/Wrist；仅展示旋转180度，模型及HDF5像素不变。

本机保留图片（未随仓库发布）：初态1失败预览，`evaluation_results/phase5b4/20261015T085852Z_8dde3e4f/episode_001_preview.png`。

本机保留图片（未随仓库发布）：初态2失败预览，`evaluation_results/phase5b4/20261015T085852Z_8dde3e4f/episode_002_preview.png`。

另保留两段15 s、10 fps双相机小视频：同目录`episode_001_failure.mp4`、`episode_002_failure.mp4`，均由已有HDF5生成，没有追加仿真。详细只读分析见同目录`behavior_analysis.json`。

## 6. 离线指标与在线结果

E5原离线验证：Loss0.076097、平移MSE0.019974、旋转MSE0.003946、夹爪准确率97.82%。这些指标在专家提供的观测上计算，不能推出97.82%的任务成功率，也不评价偏离专家轨迹后的恢复。本次在线60%体现了这种差异；不把两个失败直接归因于Phase5B.2历史示范/环境不一致。

## 7. 实际运行命令与测试

在能访问GPU的主机终端：

```bash
cd <project>
export LIBERO_CONFIG_PATH="$PWD/.libero-gpu"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/embodiedlab-gpu-numba
export MPLCONFIGDIR=/tmp/embodiedlab-libero-matplotlib
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISPLAY WAYLAND_DISPLAY

.venv-libero-gpu/bin/python scripts/evaluate_bc.py --device cuda --episodes 1 \
  --output-dir evaluation_results/phase5b4_smoke
.venv-libero-gpu/bin/python scripts/evaluate_bc.py --device cuda --episodes 5 --seed 42
.venv-libero-gpu/bin/python scripts/evaluate_bc.py --policy zero --episodes 5 --seed 42 \
  --output-dir evaluation_results/phase5b4_zero
.venv-libero-gpu/bin/python -m pytest -q -p no:cacheprovider

LIBERO_CONFIG_PATH="$PWD/.libero" NUMBA_CACHE_DIR=/tmp/embodiedlab-cpu-numba \
  .venv-libero/bin/python -m pytest -q -p no:cacheprovider
```

| 环境 | 实际全部测试结果 |
|---|---|
| GPU独立环境 | **114 passed，3 warnings，25.10 s** |
| 原CPU环境 | **111 passed，3 skipped，3 warnings，28.32 s** |

CPU跳过3项实际CUDA测试；warnings均为上游robosuite缺少private macro提示调用弃用日志API。数据、Runner、Recorder、Replay、评测及原生环境测试无回归。

## 8. 实验文件与下一步

项目中保留的独立运行目录：

- Smoke：`evaluation_results/phase5b4_smoke/20261015T085722Z_7f7aeb1f`
- 正式BC：`evaluation_results/phase5b4/20261015T085852Z_8dde3e4f`
- Zero：`evaluation_results/phase5b4_zero/20261015T085958Z_c7fbe796`

每个目录含实际协议、结果、逐步诊断、预览和完整HDF5。正式目录另含汇总图、数值分析、CPU/CUDA推理一致性与验证记录。报告交付副本的配图与JSON/video位于`phase5b4_results/`，完整HDF5仍保留项目内，不重复打包。

下一阶段优先解决一个问题：**接触后姿态偏离时，策略为什么无法恢复有效操作。** 应先用这些固定失败片段隔离姿态表征和动作预测的影响，再决定模型或数据改动；保持原E5结果作为对照。本阶段已停止，没有开始重训、模型改进或Phase5B.5。
