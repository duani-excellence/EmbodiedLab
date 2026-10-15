# EmbodiedLab：RTX 5090 GPU 训练环境修复报告

> 历史实验记录；本机路径已脱敏，诊断工具路径已更新。原始结果保留在本机，数据、权重和配图未随仓库发布。当前安装与入口以[项目 README](../../README.md)为准。

日期：2026-10-15。项目：`<project>`。

## 1. 结论

**GPU训练已真实启用，可以继续Phase 5B.4；本次没有启动在线BC推理或闭环评测。**

主机驱动原本正常，无需sudo、重装驱动、修改内核/Secure Boot或重启。原先的不可用现象有两个独立原因：**执行沙箱隐藏NVIDIA设备节点，以及项目使用CPU-only PyTorch**。建立独立CUDA环境后，还发现并修复了严格确定性模式下的池化反向兼容问题。

最终环境：RTX 5090 D v2 / 24GB / sm_120，驱动580.173.02，Python3.10.20，PyTorch2.9.1+cu128，torchvision0.24.1+cu128。真实VisualBC前向、反向、训练、checkpoint互载及原生LIBERO测试均通过。

## 2. 原因与实际诊断证据

| 检查项 | 实际结果 |
|---|---|
| hostname | `<redacted-hostname>` |
| OS / kernel | Ubuntu22.04.5 LTS；Linux6.8.0-138-generic，x86_64 |
| 宿主虚拟化 | `systemd-detect-virt=none`；DMI为Colorful CVN Z790D5 ARK FROZEN主板 |
| PCIe | `0000:01:00.0`，NVIDIA `10de:2b8c`；kernel driver in use: nvidia |
| 实际GPU名称 | **NVIDIA GeForce RTX 5090 D v2**，不是32GB版本；nvidia-smi总显存24455 MiB |
| 已加载驱动 | NVIDIA Open Kernel Module 580.173.02；nvidia、nvidia_modeset、nvidia_drm、nvidia_uvm均在 |
| 沙箱内 | `/dev/nvidia0`、`nvidiactl`、`nvidia-uvm`不存在；nvidia-smi退出码9，提示无法连接驱动 |
| 主机上下文 | 同一机器、普通用户权限；上述节点可读写，nvidia-smi成功，驱动580.173.02 |
| 环境屏蔽变量 | CUDA_VISIBLE_DEVICES / NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES均未设置 |
| 原Python环境 | `.venv-libero`，torch2.5.1+cpu，torch.version.cuda=None，CUDA不可用 |

因此，沙箱中的退出码9不能代表物理主机的驱动状态。此前检查范围只覆盖了受限执行环境，本次主机侧检查纠正了这一判断。没有发现需要更换驱动或配置Docker GPU透传的证据；沙箱外实际CUDA运算已成功。

在Codex中使用了逐命令的沙箱外执行，仍使用普通用户权限，**不是sudo/root**；没有修改全局沙箱设置。用户在能看到设备节点的普通主机终端可直接运行GPU命令。

诊断原件：`gpu_host_diagnosis.json`、`gpu_sandbox_diagnosis.json`、`gpu_cuda_probe.json`，均保留于本机研究产物目录（未随仓库发布）。

## 3. 独立环境与依赖固定

新增 `.venv-libero-gpu`，原 `.venv-libero`保留。GPU锁固定127个第三方包，连同项目共128个包，`uv pip check`全部兼容。

| 组件 | CPU环境，保持原样 | GPU环境 |
|---|---|---|
| Python | 3.10.20 | 3.10.20 |
| Torch | 2.5.1+cpu | **2.9.1+cu128** |
| torchvision | 0.20.1+cpu | **0.24.1+cu128** |
| SymPy | 1.13.1 | **1.14.0** |
| NumPy / h5py | 1.26.4 / 3.11.0 | 相同 |
| MuJoCo / robosuite | 3.3.2 / 1.4.0 | 相同 |
| hf-libero | 0.1.4，commit8561c60… | 相同固定commit |

选择的是正式发布的[PyTorch2.9.1 / torchvision0.24.1 / CUDA12.8组合](https://pytorch.org/get-started/previous-versions/)，没有使用nightly。PyTorch官方已在[2.7发布说明](https://pytorch.org/blog/pytorch-2-7/)宣布CUDA12.8 wheel的Blackwell支持；本次进一步实测2.9.1 wheel包含`sm_120`并成功执行，而不是仅依据版本名判断。

发现并处理的版本锁问题：

- 原`pyproject.toml`的data extra绑定torch==2.5.1+cpu；现改为兼容范围`torch>=2.5.1,<2.10`，具体CPU/CUDA版本由独立lock选择。
- 原CPU in/lock不修改，新增`requirements-libero-gpu.in/.lock`，精确锁定`+cu128`版本。
- Torch2.9.1要求SymPy>=1.13.3，不能原封沿用CPU锁的1.13.1；仅GPU环境改为1.14.0。
- GPU锁dry-run结果为“Would make no changes”；安装`.[data]`的dry-run只涉及项目editable元数据，没有Torch降级。

原CPU全部包版本及CPU锁SHA256均与安装前相同，见`gpu_cpu_preservation.json`。没有借用其他项目的site-packages或修改其他Python环境。

系统已有的nvcc12.6保持不动，没有安装完整系统CUDA Toolkit。`nvidia-smi`显示驱动支持的CUDA版本13.0，Torch wheel使用的runtime是12.8，两者含义不同。[NVIDIA文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)说明该版本字段表示驱动支持的最新CUDA版本。

LIBERO资产复用原固定revision目录；新环境只建立自己的assets符号链接，并使用`LIBERO_CONFIG_PATH=<project>/.libero-gpu`，避免复用指向CPU包目录的配置。原`.libero`和资产不修改。

## 4. 代码兼容修复与GPU验证

### 严格确定性的实际报错

首次真实GPU训练在反向传播报错：

```text
adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation,
but you set torch.use_deterministic_algorithms(True)
```

模型固定128px输入，第三层特征固定16×16，原AdaptiveAvgPool到4×4可等价替换为**kernel=4、stride=4的普通AvgPool**。保留严格确定性，不改变参数数量、权重键、特征维度或动作语义，也没有训练新架构。

用原E5 checkpoint与真实Batch4在原CPU环境比较，替换前后motion/logits最大差均为 **0**；参数量仍为194,920。新增自动化测试覆盖该等价性与严格确定性CUDA反向。完整报错及验证见`gpu_train_initial.log`、`gpu_pool_compatibility.json`。

### 实际验证结果

- Torch识别设备：RTX5090 D v2；compute capability **(12,0)**；compiled arch包含 **sm_120**。
- CUDA张量及128×128 FP32矩阵乘在cuda:0执行，数值有限。矩阵乘与CPU的最大绝对差为7.63e-5，未假定浮点结果逐位相同。
- 真实DataLoader的两路RGB、状态、动作均迁移到cuda:0；网络两个输出头也位于cuda:0。
- VisualBC真实Batch32的CPU/GPU梯度均有限、总体非零；严格确定性GPU训练20步，Loss **0.779921→0.209470**，耗时3.99 s（含统计/初始化/验证，非纯kernel时间）。
- 默认auto实际选择cuda；显式`--device cpu`在同一GPU环境中仍使用CPU，仅执行CPU梯度检查。两种入口另各跑1步验证。

### 原E5 checkpoint的跨环境恢复

实际加载原`training_results/phase5b3_train_seed42/best.pt`到GPU，并保存CUDA tensor checkpoint后再由原CPU Torch2.5.1环境加载：

| 比较 | motion最大差 | logits最大差 |
|---|---:|---:|
| 新环境CPU→CUDA预测 | 1.64e-7 | 4.77e-7 |
| GPU保存→原CPU环境预测 | 1.86e-7 | 4.77e-7 |

GPU保存再CPU加载的所有权重/buffer与原E5**逐元素相同**，state_dict严格加载成功。比较容差atol=1e-5、rtol=1e-4；没有将非零浮点差异写成0。临时跨设备checkpoint验证后删除，原E5文件未修改。

见`gpu_checkpoint_validation.json`、`gpu_checkpoint_cpu_roundtrip.json`。这证明旧训练结果可用于新设备，不要求为换设备重新训练。

## 5. 最小真实训练Benchmark

`tools/benchmark_bc.py`在同一个Torch2.9.1+cu128进程中比较CPU/GPU，均使用原生VisualBC、真实LiberoTorchDataset、相同初始state_dict、seed42、训练集归一化统计、Adam lr=0.001。batch32、CPU4线程、num_workers0、FP32，关闭AMP/TF32，开启严格确定性。

每组先10步warmup；固定真实Batch计时100步，DataLoader端到端计时30个Batch。每个阶段显式CUDA同步；不依靠异步提交时间计算虚高吞吐。梯度检查和checkpoint验证在计时区外，标准训练入口另有指标/检查开销，因此这些数值不是完整20轮run的耗时预测。

| 场景 | CPU平均 / 中位数 ms | GPU平均 / 中位数 ms | CPU / GPU samples/s | 平均加速比 |
|---|---:|---:|---:|---:|
| 固定Batch，不重新读数据，含设备传输 | 28.64 / 26.96 | 4.78 / 4.52 | 1117 / 6695 | **5.99×** |
| DataLoader→传输→训练端到端 | 40.43 / 38.10 | 19.44 / 18.88 | 791 / 1646 | **2.08×** |

分段平均耗时（ms/Batch）：

| 场景/设备 | 数据加载 | 传输 | Forward+Loss | Backward+Adam |
|---|---:|---:|---:|---:|
| 固定Batch CPU | ≈0 | 0.009 | 13.439 | 15.193 |
| 固定Batch GPU | ≈0 | 1.573 | 1.050 | 2.157 |
| 端到端 CPU | 13.296 | 0.041 | 13.024 | 14.073 |
| 端到端 GPU | 14.071 | 1.493 | 1.237 | 2.644 |

固定Batch经过10次预热+100次计时更新后，GPU Loss从0.779921降至 **0.006500**，CPU为0.006548。端到端30Batch后在固定参考Batch评估，GPU Loss为0.147657，CPU为0.148589；这只是设备/优化器检查，不是验证集泛化指标。

显存与利用率：

- Torch GPU最大allocated **136.7 MiB**；固定Batch最大reserved **154.0 MiB**。
- GPU阶段nvidia-smi共10次全卡采样：利用率8～32%，平均15.3%；全卡显存1894～2718 MiB。
- 全卡统计包含桌面进程、CUDA初始化/预热和checkpoint操作，不能当作模型独占利用率或长期稳态峰值；Torch allocator也不包含所有驱动/库分配。
- GPU端到端约72%时间花在数据加载，解释了为何实际流水线加速小于固定Batch。当前小CNN与num_workers0管线没有饱和GPU，不是设备不可用。本阶段不增加数据加载框架或进行Batch/参数搜索。

CPU先测、GPU后测，HDF5和OS缓存已预热。这是当前本机同步实现的少量测量，不代表冷盘或其他模型/Batch的上限。完整逐段数据与采样见`gpu_bc_benchmark.json`。

## 6. 测试结果与文件改动

| 环境 | 完整测试实际结果 |
|---|---|
| 原CPU `.venv-libero` | **101 passed, 2 skipped, 3 warnings，73.37 s**；两项CUDA测试因CPU-only构建跳过 |
| 新GPU `.venv-libero-gpu` | **103 passed, 4 warnings，28.90 s**；CUDA项目实际执行，无skip |

包括原有LIBERO初始化、真实RGB、数据、录制/回放、原生结束条件及新增BC测试。GPU环境4条warning来自robosuite的3条旧日志API和首次编译logo字符串的无效转义提示；无失败。GPU回归使用自己的原生包/固定资产配置和实际EGL渲染，未修改任务、奖励或成功判定。

本次新增/修改的项目文件：

- `requirements-libero-gpu.in/.lock`：独立CUDA依赖与完整版本固定。
- `pyproject.toml`：移除package extra对CPU build的硬绑定；原CPU锁不改。
- `src/embodiedlab/policies/visual_bc.py`：固定分辨率下的等价确定性池化修复。
- `scripts/train_bc.py`：打印实际设备/版本、清楚报CUDA不可用原因、CPU选项不跑GPU梯度检查、显式关闭TF32并保持确定性。
- `tools/benchmark_bc.py`：小型真实数据性能/内存/checkpoint验证入口。
- `tests/test_bc.py`：新增等价池化与CUDA严格确定性测试。
- `README.md`、`.gitignore`、`GPU_SETUP_REPORT.md`：运行说明、GPU配置目录与本报告。

没有修改官方HDF5、旧checkpoint、上游源码、物理模型、控制器或系统驱动。GPU smoke目录的模型只用于设备验证，不能替代正式E5训练结果。

## 7. 安装与运行命令

本机环境已配置完成，直接从能访问GPU的普通主机终端使用：

```bash
cd <project>
export LIBERO_CONFIG_PATH="$PWD/.libero-gpu"
export MPLCONFIGDIR=/tmp/embodiedlab-libero-matplotlib

# 默认auto；显式指定也可以。未提供output-dir时生成唯一目录。
.venv-libero-gpu/bin/python scripts/train_bc.py --mode overfit --steps 20 --device cuda
.venv-libero-gpu/bin/python scripts/train_bc.py --mode overfit --steps 20 --device cpu

.venv-libero-gpu/bin/python tools/benchmark_bc.py \
  --steps 100 --pipeline-batches 30 --batch-size 32 --threads 4 \
  --output training_results/gpu_benchmark_new.json

env -u DISPLAY -u WAYLAND_DISPLAY MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_CONFIG_PATH="$PWD/.libero-gpu" NUMBA_CACHE_DIR=/tmp/embodiedlab-gpu-numba \
  MPLCONFIGDIR=/tmp/embodiedlab-libero-matplotlib OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venv-libero-gpu/bin/python -m pytest -q -p no:cacheprovider
```

实际benchmark输出为本报告目录`gpu_bc_benchmark.json`；命令示例使用新的输出名以免覆盖已有结果。原环境测试将解释器换为`.venv-libero/bin/python`并不设置`.libero-gpu`配置。

从零建立GPU虚拟环境的完整命令在项目README的“RTX5090 GPU环境”节：使用GPU锁及官方cu128 index，再`--no-deps -e .`，建立固定资产链接。实际安装首先装CUDA两包，然后安装保留CPU其他版本的基础依赖并升级GPU SymPy，最后freeze独立GPU锁；GPU锁安装dry-run已核验无变化。安装日志`gpu_torch_install.log`、`gpu_base_install.log`保留在本机研究产物目录，无需重复下载现有资产。

## 8. 下一阶段是否就绪

**可以继续Phase 5B.4。** 设备访问、sm_120算子、真实训练、旧权重互载和原生仿真兼容性均已有实际证据。应继续使用原E5模型与既定预处理、官方reset和原生成功规则；Phase5B.2的场景一致性风险、Phase5B.3的过拟合/rx与夹爪切换问题仍需在未来闭环评测中观察。

本次不以GPU benchmark的Loss宣称任务已成功，不重训完整模型，不开发新的在线Policy，也不开始Phase 5B.4。

