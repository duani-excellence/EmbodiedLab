# Phase 5C：通用化与稳定性验收

> 历史实验记录；本机路径已脱敏，诊断工具路径已更新。原始结果保留在本机，数据、权重和配图未随仓库发布。当前安装与入口以[项目 README](../../README.md)为准。

2026-10-15，项目 `<project>`。

**结论：PASS，满足本次 EmbodiedLab v0.1 工程验收条件。** 通过配置，在第二个官方任务上完成真实GPU训练、Checkpoint恢复、原生闭环评测与轨迹Replay；旧炉灶模型无回归。没有修改VisualBC结构、动作解码、原生任务或控制器。没有新增算法或性能调参。

## 1. 项目结构与改动

核心结构如下；历史数据、Checkpoint、评测目录及报告均保留：

```text
embodiedlab/
├── src/embodiedlab/
│   ├── interfaces.py
│   ├── envs/libero_env.py
│   ├── data/{episode.py,torch_dataset.py,replay.py,verified_sources.json}
│   ├── policies/{simple.py,visual_bc.py,bc.py,checkpoint.py}
│   ├── runners/rollout.py
│   └── evaluation.py
├── scripts/
│   ├── train_bc.py
│   ├── evaluate_bc.py
│   ├── evaluate_libero.py
│   ├── download_libero.py
│   ├── inspect_dataset.py / inspect_torch_dataset.py
│   ├── replay_libero.py
│   └── benchmark_bc.py / diagnose_libero_consistency.py
├── tests/                      # 10个测试文件，162个case
├── data/libero/libero_goal/     # 两份官方HDF5及来源sidecar
├── training_results/phase5c_drawer_seed42/
├── evaluation_results/phase5c_source_audit/
├── evaluation_results/phase5c_drawer/
├── evaluation_results/phase5c_stove_regression/
├── requirements-libero-gpu.in / .lock
├── requirements-libero.in / .lock
├── upstream.json / pyproject.toml
└── README.md / PHASE5C_REPORT.md
```

| 文件 | 本次改动与实际用途 |
|---|---|
| `scripts/train_bc.py` | 增加suite/task与比例/划分种子配置，移除40/10强制要求，保存新版契约；允许合法的单夹爪类别数据 |
| `data/torch_dataset.py` | 复用原读取器；从已审核来源推断身份或核对显式身份，验证真实文件SHA、相机与控制语义 |
| 新增`data/verified_sources.json` | 两条真实审核来源与输入/动作约定；不是自动发现或通用插件系统 |
| 新增`policies/checkpoint.py` | 训练和推理共用版本化任务/来源/I/O/划分契约，避免两端独立猜测配置 |
| `policies/bc.py` | 自动恢复新旧Checkpoint，提供任务属性和`assert_compatible(env)` |
| `scripts/evaluate_bc.py` | 移除炉灶场景/关节/300步硬编码，支持任务、显式或规则初态、设备和输出配置；通用诊断与资源采样 |
| `evaluation.py` | 有序官方初态索引和可选Policy契约检查，Python评测同样拒绝错配；Episode核心循环无任务分支 |
| `envs/libero_env.py`、`evaluate_libero.py` | 增加无仿真的任务身份查询入口；环境动力学/控制/结束语义不改 |
| `download_libero.py` | 按suite/task选择一份审核过的官方示范，保留默认炉灶下载，不覆盖既有数据/sidecar |
| `pyproject.toml` | 将审核清单包含进package data；实际构建wheel并检查包含该文件 |
| 新增4个测试文件 | `test_checkpoint.py`、`test_data_sources.py`、`test_stability.py`、`test_training_entry.py` |
| 修改`test_bc_policy.py` | 更新有效环境fixture并验证直接Python评测在reset前拒绝任务错配 |
| `README.md` | 重整环境、数据、训练、恢复、评测、Policy接口与复现实验说明 |

`VisualBC`、`bc_loss`、`decode_actions`、`RolloutRunner`、原`EpisodeDataset/Recorder/Replay`本阶段均无需修改。训练继续使用原`training_statistics()`和`run_epoch()`，不增加Trainer基类。

## 2. 数据来源和对齐审核

第二任务是 **`libero_goal / task_id=0 / open_the_middle_drawer_of_the_cabinet`**，与旋钮操作不同。数据来自[LIBERO官方数据下载说明所指向的Hub来源](https://github.com/Lifelong-Robot-Learning/LIBERO#datasets)，仓库`yifengzhu-hf/LIBERO-datasets`、revision `147fa5eb607bb696dd922c3518084a90954cd2f3`。

- 文件：`data/libero/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5`，702,223,367 bytes。
- SHA256：`20252c7cf98cd7437061f7f200ae7b6cb6219fabbd53b4536dfaa8abda6ab737`，下载后及全部实验后相同。
- 50条Episode、7,027个原始动作，移位后6,977个监督样本。seed42、80%/20%分为40/10条，**5,633 / 1,344样本**；准确Episode名字保存在config和Checkpoint，交集为空。
- 50/50轨迹中，`obs/joint_states[:-1]`与`states[1:,1:8]`逐值相等，finger同样误差0，验证其post-action约定。监督使用`obs[t-1] → action[t]`，丢弃首个不可完整配对动作，不猜测缺失初始观测。
- 两路RGB实际为128×128×3 uint8；共享预处理保持原OpenGL方向，输出CHW float32 `[0,1]`；状态8维，动作7维。3次隔离的原生EGL单步检查确认在线图像/预处理逐值一致。
- 全部7,027条夹爪命令均为**−1**。允许这类合法数据；只对常量对照概率采用Laplace平滑，防止缺失类别导致log(0)，不补造监督标签。
- 数据`env_args`存有旧别名`open_the_middle_layer_of_the_drawer`；精确SHA审核记录将其明确对应当前canonical task0。不是运行时模糊匹配，也没有修改HDF5。

新来源必须先审核再加入清单。Dataset每次构造流式核验一次完整文件SHA，逐Episode检查schema/语义，RGB仍懒加载；复制来源sidecar或自行重算另一个文件的SHA均不能绕过审核。

仍需保留科学边界：历史状态的3次部分恢复单步实验，最大关节偏差约0.05120 rad、最大末端位置偏差约10.93 mm；控制器历史状态不完整，不能称为同一完整状态下动力学不一致，也未证明专家轨迹完全复现。证据和脚本在`evaluation_results/phase5c_source_audit/`。

## 3. 训练、Checkpoint和推理接口

训练参数支持`--dataset --suite --task-id --epochs --batch-size --lr --seed --device --output-dir`，另有`--train-fraction --split-seed`。默认划分种子为`--seed`，以完整Episode划分。输出目录必须不存在，防止覆盖历史结果。训练前拒绝未审核来源、错误任务、错误图像或动作语义。

Checkpoint格式升至2，保存`state_dict`（含归一化buffer）、epoch/验证指标、config与contract。contract明确记录架构`visual_bc_v1`、suite/id/name、数据repo/revision/SHA、128px相机与预处理、8维状态顺序、OSC增量尺度、夹爪类别、控制周期及准确划分。

`BCPolicy(checkpoint, device='auto')`自动恢复，`policy.suite/task_id/image_size`供环境创建使用；`assert_compatible(env)`在首次reset前检查任务和I/O/控制契约。实际验证两个方向的task0/task7误用均报错，没有执行错误任务。

旧E5格式1仍可加载，但只在已知provenance、revision和数据SHA匹配时**显式警告**并记录兼容身份为goal/task7。未知历史身份不静默猜测。新版config与contract的矛盾也会报错。

评测支持`--checkpoint --suite --task-id --init-state-indices`，或`--episodes --init-state-rule first/random`，以及seed/device/output-dir。默认可从Checkpoint读取身份。种子按Episode序号递增，不按初态编号递增。时限直接使用既有suite配置：spatial/object=280、goal=300、10=520、90=400；本次两个goal任务均为300步。

原始Observation经共享预处理、保存的状态归一化、`eval()/inference_mode()`、原动作解码进入环境；没有滤波、手工夹爪规则或专家状态注入。诊断对象状态和接触不进入模型。

## 4. 第二任务真实训练与闭环结果

RTX5090 D v2、Torch2.9.1+cu128；原CNN+MLP共194,920参数。5轮、Adam、lr0.001、batch32、seed42，无超参数搜索。

| 指标 | 实测 |
|---|---:|
| 初始验证Loss | 0.772272 |
| 最佳Epoch | 5 |
| 最佳验证Loss | 0.028770 |
| 平移 / 旋转验证MSE | 0.053232 / 0.003998 |
| 常量动作对照验证Loss | 0.072558 |
| 夹爪验证Accuracy | 100%（原数据全部open，常量open同样100%） |
| 训练总耗时 / 每轮训练+验证均值 | 30.75 s / 3.79 s |
| Checkpoint恢复motion/logits最大差 | 0 / 0 |

固定两类别宏F1为0.5，因为未出现的close类按0计；既不代表分类器随机，也不证明学会夹爪切换。离线Loss与在线任务成功分别报告。

新模型闭环使用预先指定的官方初态0～7和seed42～49，原生reset、10步settling、300策略步；不根据结果选初态：

| 初态 | Seed | 结果 | 步数 | 仿真时间s |
|---:|---:|---|---:|---:|
| 0 | 42 | 超时 | 300 | 15.00 |
| 1 | 43 | 超时 | 300 | 15.00 |
| 2 | 44 | 原生成功 | 293 | 14.65 |
| 3 | 45 | 原生成功 | 242 | 12.10 |
| 4 | 46 | 原生成功 | 208 | 10.40 |
| 5 | 47 | 超时 | 300 | 15.00 |
| 6 | 48 | 原生成功 | 214 | 10.70 |
| 7 | 49 | 超时 | 300 | 15.00 |

**4/8成功（50%）**，共2,157个真实模型动作，运行28.70 s。推理含预处理/传输/同步取回：均值0.959 ms、中位数0.876 ms、P95 1.009 ms、冷启动最大135.02 ms。只代表本次单任务所选初态，不是标准全套Benchmark成绩，也未证明这些初态独立于训练示范。

每条轨迹均用原Recorder保存T动作/T+1观测并可读取；一条208步成功轨迹实际Replay后，robot_state、joint、reward、timestamp最大差均0，RGB像素差0，长度、语言、success/terminated/truncated一致。

本机保留图片（未随仓库发布）：第二任务训练曲线，`training_results/phase5c_drawer_seed42/losses.png`。

本机保留图片（未随仓库发布）：真实抽屉操作预览，`evaluation_results/phase5c_drawer/20261015T124429Z_604dda1e/episode_004_preview.png`。

## 5. 旧任务、设备与稳定性

旧`training_results/phase5b3_train_seed42/best.pt`通过同一通用入口，初态0～4/seed42～46再次得到 **3/5成功**，步数225/300/300/113/109，与Phase5B.4一致。权重SHA仍为`00e37e0be92296c2a9a2753d83320768e62165cfbf6526196c860b79a405978b`，原数据SHA也未改变。

新Checkpoint `training_results/phase5c_drawer_seed42/best.pt` SHA为`343578c254125ad0dbfbe00ce09b5b8bf7dfe35066193849f3319860359ccfd1`。GPU保存的全部参数/buffer可由原CPU Torch2.5.1逐元素准确恢复；3个真实观测的同设备Policy/直接Forward差0，CPU/GPU最大动作差**8.94e-8**。

真实连续8条Episode的监测结果：

- Torch allocated/reserved在推理期间保持**33.75/36 MiB**。另由`nvidia-smi`逐秒采样整个进程，包含原生渲染的NVML显存峰值**1,129 MiB**，reset/关闭时下降，没有逐Episode上升。
- CPU RSS从首条开始约2,430 MiB到末条约2,533 MiB，中间多次下降，峰值约2,542 MiB；close后约2,140 MiB。没有观察到单调失控增长，但这不是无限期稳定性证明，进程缓存也未全部回到冷启动水平。
- rollout文件描述符始终**69**，环境/Recorder关闭后62；没有逐Episode增加。驱动/库的进程级句柄仍存在，不把它表述为全部资源回到初始值。
- 原CPU环境下连续3轮×20个batch32的懒读，均值9.91/9.64/9.32 ms每Batch，RSS286/280/283 MiB，FD5；Dataset关闭后FD回到4。未将702 MB的全部图像读入内存。
- 成功、超时之外，还实际在原生一步后触发observer异常：模拟器和HDF5均关闭，记录明确为incomplete，正式读取器拒绝将其当完整Episode。异常不会伪记为算法失败。

本机保留图片（未随仓库发布）：连续评测资源采样，`evaluation_results/phase5c_drawer/20261015T124429Z_604dda1e/resources.png`。

## 6. 实际命令与自动化结果

本机GPU环境配置沿用README，没有改驱动、重装环境或覆盖旧依赖。实验前设置：

```bash
cd <project>
export LIBERO_CONFIG_PATH="$PWD/.libero-gpu"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/embodiedlab-gpu-numba MPLCONFIGDIR=/tmp/embodiedlab-libero-matplotlib
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISPLAY WAYLAND_DISPLAY

.venv-libero-gpu/bin/python scripts/download_libero.py --demo --suite libero_goal --task-id 0
.venv-libero-gpu/bin/python scripts/evaluate_libero.py --suite libero_goal --list-tasks
.venv-libero-gpu/bin/python scripts/train_bc.py \
  --dataset data/libero/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5 \
  --suite libero_goal --task-id 0 --epochs 5 --batch-size 32 --lr 0.001 \
  --seed 42 --device cuda --output-dir training_results/phase5c_drawer_seed42

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv-libero-gpu/bin/python scripts/evaluate_bc.py \
  --checkpoint training_results/phase5c_drawer_seed42/best.pt --suite libero_goal --task-id 0 \
  --init-state-indices 0 1 2 3 4 5 6 7 --seed 42 --device cuda \
  --output-dir evaluation_results/phase5c_drawer
.venv-libero-gpu/bin/python scripts/evaluate_bc.py \
  --checkpoint training_results/phase5b3_train_seed42/best.pt --suite libero_goal --task-id 7 \
  --init-state-indices 0 1 2 3 4 --seed 42 --device cuda \
  --output-dir evaluation_results/phase5c_stove_regression
.venv-libero-gpu/bin/python scripts/replay_libero.py \
  --episode evaluation_results/phase5c_drawer/20261015T124429Z_604dda1e/records/episode_004.hdf5
.venv-libero-gpu/bin/python -m pytest -q -p no:cacheprovider
LIBERO_CONFIG_PATH="$PWD/.libero" NUMBA_CACHE_DIR=/tmp/embodiedlab-cpu-numba \
  .venv-libero/bin/python -m pytest -q -p no:cacheprovider
```

数据下载后再次执行下载命令实际验证既有文件，不重新覆盖；训练复现必须换一个尚不存在的输出目录。抽屉评测实际额外由只读进程采样器包裹，上述命令是相同内部评测命令。

| 检查 | 实际结果 |
|---|---|
| GPU全部新旧测试 | **162 passed，14 warnings，28.17 s** |
| 原CPU环境全部测试 | **158 passed，4 skipped，13 warnings，27.27 s** |
| 新版wheel含来源审核清单 | 通过，构建产物位于`/tmp/embodiedlab-phase5c-dist/` |
| 输出目录防覆盖、错误来源/同size伪造、跨Episode隔离 | 自动化测试通过 |
| 数据/权重不变、CPU/GPU恢复、真实原生异常关闭 | 实际验证与自动化测试通过 |

CPU仅跳过4项CUDA测试；warnings是明确的旧Checkpoint身份兼容提示及3条上游robosuite弃用日志提示。没有用Mock代替第二任务训练或闭环验收。

## 7. 结果位置、限制与验收判断

- 新训练：`training_results/phase5c_drawer_seed42/`。
- 第二任务：`evaluation_results/phase5c_drawer/20261015T124429Z_604dda1e/`。
- 旧E5回归：`evaluation_results/phase5c_stove_regression/20261015T123915Z_b2c9db53/`。
- 来源证据：`evaluation_results/phase5c_source_audit/`。
- 交付副本：本报告旁`phase5c_results/`含曲线、协议、指标、预览、资源日志、数据审核与新Checkpoint；完整HDF5轨迹仍在项目内，避免重复打包。

当前限制是有意保留的明确边界：只有两份已审核训练来源；新数据必须先审核，不是任意HDF5自动适配。VisualBC仍为128px双相机、8维状态、单任务、二类夹爪，不支持零夹爪监督或多任务联合训练。num_workers=0，不实现通用Trainer或新算法；Policy替换沿用reset/act，训练入口可独立。

历史示范与当前仿真的版本/隐藏控制器状态差异、轴角表征问题仍需在后续算法研究中说明；本次没有为提高成功率修改数据、物理或评测规则。资源检查只覆盖本次有限连续运行，未声称长期压力测试通过。

**PASS依据是第二任务完整真实流程已通过配置走通、旧任务无回归、身份/数据/输出/资源边界明确。50%与60%成功率是实验观测，不是v0.1工程通过的替代条件。Phase5C完成后停止。**
