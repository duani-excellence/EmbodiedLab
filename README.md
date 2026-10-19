# EmbodiedLab — 仿真机器人学习实验

一个简洁的机器人学习研究框架。LIBERO主线支持单任务VisualBC训练、Checkpoint恢复、原生闭环评测、HDF5记录与回放；另包含官方Wipe与Factory GearMesh接触任务。各任务保留固定上游环境的控制器、动力学、奖励和成功判定。

源码版本为 **0.1.0**。仓库不分发官方示范、仿真资产、Checkpoint或本地实验输出；依赖和来源版本分别记录在锁文件与 [upstream.json](upstream.json)。

## 安装

已验证 Linux / Python 3.10.20。以下命令在仓库根目录执行，使用 [uv](https://docs.astral.sh/uv/) 创建环境。CPU与GPU环境选其一；已有环境不需要重建。

```bash
git clone https://github.com/duani-excellence/EmbodiedLab.git
cd EmbodiedLab
```

CPU：

```bash
uv venv --python 3.10 .venv-libero
uv pip install --python .venv-libero/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -r requirements-libero.lock
export EL_PY=.venv-libero/bin/python
export LIBERO_CONFIG_PATH="$PWD/.libero"
```

NVIDIA GPU：

```bash
uv venv --python 3.10 .venv-libero-gpu
uv pip install --python .venv-libero-gpu/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match -r requirements-libero-gpu.lock
export EL_PY=.venv-libero-gpu/bin/python
export LIBERO_CONFIG_PATH="$PWD/.libero-gpu"
```

GPU锁使用 Torch 2.9.1+cu128 / torchvision 0.24.1+cu128，已在 RTX5090 D v2 上验证；CPU锁使用 Torch 2.5.1+cpu / torchvision 0.20.1+cpu。两锁分别维护，避免用CPU锁重装GPU环境。CUDA wheel需要兼容的NVIDIA驱动，无需另装完整CUDA Toolkit。

安装项目并配置无窗口运行：

```bash
uv pip install --python "$EL_PY" --no-deps -e .
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR="$LIBERO_CONFIG_PATH/cache/numba"
export MPLCONFIGDIR="$LIBERO_CONFIG_PATH/cache/matplotlib"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISPLAY WAYLAND_DISPLAY
```

CPU训练不需要CUDA；在线仿真仍需要可用的EGL/OpenGL渲染环境。容器或受限沙箱还需能访问相应设备。现有环境可能通过符号链接共享Python或资产目录，整理文件时应保留链接目标。

## 准备任务与数据

首次仿真下载固定资产快照，并建立原生包到资产目录的链接：

```bash
$EL_PY scripts/download_libero.py --assets
$EL_PY scripts/evaluate_libero.py --suite libero_goal --list-tasks
```

`--assets`取得固定的仿真资产快照；示范数据使用`--demo`按任务单文件下载。环境支持`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`、`libero_90`，训练数据目前只接受以下两份已审核来源：

| Suite / task ID | 任务 | 官方HDF5，位于`data/libero/libero_goal/` |
|---|---|---|
| libero_goal / 0 | 打开柜子中层抽屉 | `open_the_middle_drawer_of_the_cabinet_demo.hdf5` |
| libero_goal / 7 | 打开炉灶 | `turn_on_the_stove_demo.hdf5` |

例如，只获取抽屉示范并读取一个时间步：

```bash
$EL_PY scripts/download_libero.py --demo --suite libero_goal --task-id 0
$EL_PY tools/inspect_dataset.py \
  --dataset data/libero/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5 \
  --episode demo_0 --suite libero_goal --task-id 0
```

炉灶数据使用相同下载命令并指定`--task-id 7`。下载器验证完整文件大小和SHA256，不覆盖不匹配的已有文件。每份HDF5旁保留`.source.json`；训练还独立核对[审核清单](src/embodiedlab/data/verified_sources.json)中的真实指纹，不能靠复制sidecar批准未知数据。来源、历史任务别名和实际对齐检查见[数据审核证据](docs/data_sources/)。

## 训练与评测

训练一份指定任务的数据：

```bash
$EL_PY scripts/train_bc.py \
  --dataset data/libero/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5 \
  --suite libero_goal --task-id 0 --epochs 5 --batch-size 32 --lr 0.001 \
  --seed 42 --device auto --output-dir training_results/drawer_seed42
```

默认按完整Episode进行80%/20%划分，准确名单写入配置；`--split-seed`、`--train-fraction`可显式设置。RGB按帧读取，`DataLoader`使用`num_workers=0`。训练输出包含`best.pt`、配置、逐轮指标、常量基线和曲线；已有训练输出目录会报错，重复实验应换新目录。`--device cpu/cuda`可固定设备；`--mode overfit --steps 100`用于固定真实Batch检查。

恢复刚训练的模型并记录5个官方初态：

```bash
$EL_PY scripts/evaluate_bc.py \
  --checkpoint training_results/drawer_seed42/best.pt \
  --suite libero_goal --task-id 0 --init-state-indices 0 1 2 3 4 \
  --seed 42 --device auto --output-dir evaluation_results/drawer
```

也可省略suite/task，采用Checkpoint中的身份；任务或输入/控制契约冲突会报错。评测创建独立运行目录：`protocol.json`保存初态、种子与模型指纹，`results.json`保存逐次结果，`records/`保存HDF5，另有诊断和预览。`--init-state-rule random --episodes 5`可固定种子无放回选初态，种子按Episode序号递增。

不需要模型的短环境/记录检查：

```bash
$EL_PY scripts/evaluate_libero.py --suite libero_goal --task-ids 0 \
  --policy zero --episodes 1 --seed 42 --max-episode-steps 5 \
  --output-dir evaluation_results/smoke --record-dir data/smoke_records
```

这是接口检查，五步时限不用于任务成功率比较。正式评测使用原生reset、10步settling和suite时限：spatial/object为280步、goal为300步、libero_10为520步、libero_90为400步。原生成功对应`terminated`，超时对应`truncated`。

回放时，将下列示例路径替换为本次`results.json`中的实际`record_path`：

```bash
$EL_PY scripts/replay_libero.py --episode "path/to/episode_000.hdf5"
```

回放真实执行自录动作并报告状态、RGB与终止标志偏差；它不承诺当前仿真能精确复现历史官方专家文件。Checkpoint不随仓库发布，评测需使用自己训练或另外提供的兼容模型；仅一个已知来源的历史E5格式提供显式身份映射，其他未知旧权重拒绝猜测。

## OpenPI、OpenVLA-OFT与固定挑战集

官方OpenPI π₀.₅与联合训练OpenVLA-OFT使用各自独立模型环境/服务，两大模型顺序加载。安装、权重获取、服务与真实评测命令见[OpenPI指南](docs/openpi/README.md)和[OFT/挑战集指南](docs/benchmark/README.md)。版本/权重身份分别固定在[OpenPI pin](openpi-upstream.json)与[OFT pin](openvla-upstream.json)。

OpenPI保留官方图像转换、10步预测/默认5步消费；OFT保留其center crop、状态/动作归一化、夹爪转换与8步消费。不交换模型协议，不调整权重。[固定manifest](benchmark_manifest.json)保存六任务/初态/seed，含Spatial220步等正式协议差异。完成运行后只读分析：

```bash
$EL_PY tools/analyze_challenge.py --manifest benchmark_manifest.json \
  --openpi-dir evaluation_results/phase6b_formal_openpi \
  --openvla-dir evaluation_results/phase6b_formal_openvla \
  --output-dir evaluation_results/challenge_analysis_new
```

## 可选接触任务

Wipe使用主线GPU环境，比较触碰定位固定深度OSC和腕力导纳；两者是oracle污点状态传统基线。六维OSC，无夹爪；接触/恢复指标与T动作/T+1观测保存在专属NPZ：

```bash
$EL_PY scripts/evaluate_wipe.py --seeds 42 --scenarios nominal \
  --output-dir evaluation_results/wipe_nominal_new
```

Factory GearMesh使用独立Python3.11、Isaac Lab2.3.0 / Sim5.1、Torch2.7.0+cu128与官方循环PPO。安装和Checkpoint校验见[Factory安装附录](docs/archive/CONTACT_RICH_REPORT.md#附录factory-安装与运行)。已有环境直接运行：

```bash
env PYTHONPATH="$PWD/src" OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-factory/bin/python scripts/evaluate_gearmesh.py --headless --device cuda:0 \
  --checkpoint .deps/isaac-factory/checkpoint.pth --contacts --seeds 42 \
  --output-dir evaluation_results/factory_gearmesh_new
```

两任务只共享Policy/Runner，尚未接入LIBERO HDF5 Dataset/Replay。各自接触指标的采样/定义不同，不混成总分。结果与局限见[接触任务资料](docs/archive/CONTACT_RICH_REPORT.md)。

## 代码与数据约定

`src/embodiedlab/`中的`envs/`接原生仿真，`data/`负责LIBERO记录/样本/Replay，`policies/`保留模型与Adapter，`runners/rollout.py`执行Episode，`evaluation.py/benchmark.py`负责LIBERO评测/统计。`scripts/`是正式运行入口，[tools/](tools/README.md)提供数据查看、模型服务和离线分析。

LIBERO动作是七维归一化OSC：世界系平移缩放0.05m，轴角增量缩放0.5rad并左乘，夹爪-1开/+1关。VisualBC输入双路128px RGB及8维`eef_xyz + axis_angle + finger_qpos`；图像转CHW float32/255，保持原始像素方向，训练集状态归一化buffer随权重保存。损失为前六维MSE+夹爪二类CE。

仅对已审核官方来源使用`obs[t-1] → action[t]`，舍弃缺少完整输入的首动作；自录T+1观测不做该移位。新增Policy实现`reset()`、`act(observation)`即可复用Runner；各算法保留清晰训练脚本，无通用Trainer要求。

## 测试与历史

```bash
$EL_PY -m pytest -q -p no:cacheprovider
```

若按OpenPI指南使用可选client target，先设置`PYTHONPATH="$PWD/.deps/openpi-client-runtime:$PWD/src"`，否则VLA相关测试模块会跳过。

完整测试需要固定资产、两份官方HDF5和可用EGL；CUDA项在CPU环境跳过。旧E5权重不分发，目前部分新版Checkpoint fixture也依赖它，缺权重时会一起跳过，不能视为已验证。基础入口检查可只运行`tests/test_evaluation.py tests/test_training_entry.py`。

历史报告、静态实验导出及未实施PRO调查见[归档索引](docs/archive/README.md)。数据、权重、实验结果、虚拟环境与缓存保留在本机，由`.gitignore`排除；小样本成功率不代表标准LIBERO成绩。

ROS四个源码模块、五个运行入口、Docker/controller及上游记录完整保留在本机`archive/ros2-phase7f-20261019`分支；当前分支不再维护ROS入口。需要恢复时使用独立目录：`git worktree add ../embodiedlab-ros2 archive/ros2-phase7f-20261019`，另行启用原隔离环境。当前没有真机部署、多任务训练或ACT/RL/World Model训练器。

## 许可

原创代码采用[MIT](LICENSE)，第三方代码、数据和资产遵守各自许可，见[来源声明](THIRD_PARTY_NOTICES.md)。
