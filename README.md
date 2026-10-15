# EmbodiedLab v0.1.0 — LIBERO Training, Inference & Evaluation Framework

一个简洁的 LIBERO 机器人学习研究框架。当前支持单任务 VisualBC 训练、Checkpoint恢复、原生闭环评测，以及 HDF5 轨迹记录和动作回放。任务、控制器、物理、奖励与成功判定采用固定版本的 LIBERO。

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

## 数据与代码接口

Observation包含两路原始HWC RGB `uint8`、8维`robot_state`、7维`joint_positions`和`language`。状态顺序为末端世界位置3维（m）、轴角3维（rad）、手指位置2维（m）。共用`preprocess_observation()`将两路图像转为CHW `float32 / 255`，保持像素方向；预览展示时的180度旋转不参与训练或推理。

动作是7维归一化OSC指令：`action[:3]`为世界平移增量（缩放0.05m），`action[3:6]`为轴角增量（缩放0.5rad），`action[6]`为夹爪`-1`开、`+1`关。训练采用前6维MSE和夹爪二分类CE。抽屉示范只包含`-1`夹爪标签，不能用其夹爪准确率证明开合切换能力。

已审核官方数据的观测在动作之后，监督对齐为`obs[t-1] → action[t]`，`t=1..T-1`；首个动作缺少完整输入，不能补造初始RGB。自录数据采用T个动作、T+1个观测，不做上述移位。新来源必须单独核查，不自动套用此规则。

| 位置 | 职责 |
|---|---|
| `src/embodiedlab/envs/` | 原生LIBERO适配 |
| `src/embodiedlab/policies/` | Policy、VisualBC、Checkpoint契约 |
| `src/embodiedlab/runners/`、`evaluation.py` | Rollout与评测组织 |
| `src/embodiedlab/data/` | HDF5、Torch样本、记录与回放 |
| `scripts/` | 下载、训练、评测、回放正式入口 |
| [tools/](tools/README.md) | 数据检查、历史诊断与CPU/GPU微基准 |

新增Policy只需实现`reset()`和`act(observation)`，Runner不依赖具体模型。当前范围是单任务、128px双RGB、8维状态和原生OSC控制；未实现多任务联合训练、ROS 2或真实机器人部署。

## 测试与实验记录

不依赖示范、权重或渲染的基础编排/训练入口检查：

```bash
$EL_PY -m pytest -q -p no:cacheprovider \
  tests/test_evaluation.py tests/test_training_entry.py
```

完整本地回归：

```bash
$EL_PY -m pytest -q -p no:cacheprovider
```

完整套件需要固定资产、两份官方HDF5和可用EGL。旧E5权重不分发：新克隆缺少它时，相关兼容性测试明确跳过并显示原因；其余真实数据和新格式测试仍可运行。保留E5的本地环境还可执行原模型回归。CUDA专属检查在CUDA不可用时跳过。

真实实验命令、结果与局限见[Phase 5C报告](docs/archive/PHASE5C_REPORT.md)，旧报告保存在[docs/archive/](docs/archive/)。少量固定初态的成功率和离线误差不等于标准Benchmark成绩。公开版本从经审核的v0.1.0快照开始；含本机路径和旧FR3资产的早期Git历史仅在维护者本机保留，未推送。数据、环境和本地实验输出保持独立，不随源码整理删除。

## 许可

EmbodiedLab原创代码采用[MIT许可证](LICENSE)。LIBERO代码、示范数据、仿真资产和依赖分别适用其自身条款，见[第三方声明](THIRD_PARTY_NOTICES.md)。本仓库不把第三方许可统一改为MIT。
