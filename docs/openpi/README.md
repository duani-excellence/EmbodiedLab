# OpenPI π₀.₅：可选服务端策略

这里连接真正的官方 `pi05_libero` JAX Checkpoint，不训练模型。EmbodiedLab 的 LIBERO、PyTorch CPU/GPU 环境保持原样；OpenPI 模型运行在独立 Python 3.11 环境，通过官方 WebSocket 客户端通信。

源码和每个权重对象的 generation、大小、SHA256 固定在根目录 `openpi-upstream.json`。本地权重为 16 个 JAX Orbax OCDBT 文件，共 12,439,085,481 bytes；不包含 PyTorch safetensors，也不随源码发布。

## 安装一次

先完成根 README 的 LIBERO GPU 环境和资产安装。下面从仓库根目录执行；已有对应环境/源码时直接复用，不覆盖它们。

```bash
export OPENPI_SRC="$PWD/.deps/openpi-215abfb217dbac7d5f1273282331b9b1866c0479"
git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_SRC"
git -C "$OPENPI_SRC" checkout --detach 215abfb217dbac7d5f1273282331b9b1866c0479
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-openpi"
export UV_CACHE_DIR="$PWD/.deps/openpi-uv-cache"
export UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1
(cd "$OPENPI_SRC" && uv sync --locked --no-dev --python 3.11)
```

使用上游原始 `uv.lock`，其 SHA256 也在 pin 中；本机实际 Python 3.11.15、JAX/jaxlib 0.5.3。无需获取 LIBERO 子模块或为 OpenPI 修改原仿真环境。若本机 Git 配置了失效代理，应仅为安装命令设置有效代理，避免改全局配置。

客户端安装到可选 target，不改原 `.venv-libero-gpu` 的 site-packages：

```bash
uv pip install --python .venv-libero-gpu/bin/python --no-deps \
  --target .deps/openpi-client-runtime -r requirements-openpi-client.lock \
  "$OPENPI_SRC/packages/openpi-client"
.venv-libero-gpu/bin/python scripts/download_openpi.py
.venv-libero-gpu/bin/python scripts/download_openpi.py --verify-only
```

客户端其余依赖沿用原 LIBERO 锁中的 NumPy/Pillow/attrs/absl/typing-extensions。服务端依赖完全使用独立环境的上游锁。下载器按 GCS generation 下载并检查 SHA256；不匹配的已有文件会报错并保留，续传仅写 `.partial`。首次模型加载还会取得官方 PaliGemma tokenizer，保存到同一个忽略的缓存。所有缓存和环境都不进入 Git。

## 终端一：模型服务

本机 RTX 5090 D v2 的实际验证命令如下。`ptxas` 必须优先使用独立环境中 12.9.41；本次上游 JAX 0.5.3 的 Triton GEMM 候选编译触发 assembler 崩溃，明确关闭这一条编译路径后运行成功。没有修改驱动、系统 CUDA、模型结构或权重。

```bash
export PATH="$PWD/.venv-openpi/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$PWD/.deps/openpi-jax-cache"
export OPENPI_DATA_HOME="$PWD/.deps/openpi-cache" HF_HOME="$PWD/.deps/openpi-hf-cache"
export WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv-openpi/bin/python tools/serve_openpi.py --port 8006 \
  --manifest evaluation_results/openpi_server/run01.json
```

服务只监听本机。入口检查源码 commit、tracked diff、上游锁、全部权重 SHA256 和实际 GPU，然后调用官方 loader/server。manifest 同时通过 WebSocket 握手发送；评测会比较 launch identity 和已审核模型指纹，拒绝旧 manifest、错端口或其他 Checkpoint。manifest 路径必须不存在；重启服务要换新文件名。

## 终端二：闭环评测

沿用根 README 的 EGL、LIBERO_CONFIG_PATH 和缓存配置，再加可选客户端：

```bash
export PYTHONPATH="$PWD/.deps/openpi-client-runtime${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
.venv-libero-gpu/bin/python scripts/evaluate_openpi.py \
  --suite libero_goal --task-id 7 --init-state-indices 0 1 2 3 4 --seed 42 \
  --port 8006 --native-action-bounds \
  --server-manifest evaluation_results/openpi_server/run01.json \
  --output-dir evaluation_results/openpi_stove
.venv-libero-gpu/bin/python scripts/evaluate_openpi.py \
  --suite libero_goal --task-id 0 --init-state-indices 0 1 2 3 4 5 6 7 --seed 42 \
  --port 8006 --native-action-bounds \
  --server-manifest evaluation_results/openpi_server/run01.json \
  --output-dir evaluation_results/openpi_drawer
```

`--native-action-bounds` 是显式、可审计的接口转换。官方逆归一化夹爪值会略超出单位区间；原生 robosuite 对手臂输入 clip、对 Panda 夹爪取 sign。84 步真实动作在原生环境中，原值/限幅输入的全仿真状态、RGB、奖励和终止状态完全一致，证据见 [native_bounds_equivalence.json](native_bounds_equivalence.json)。适配器不额外缩放、不按分类阈值重写夹爪，并保存所有 raw chunk/issued action/越界维度与幅度。未指定此参数时遇到执行前缀越界会明确中断，不静默裁剪。

输出为独立运行目录：`protocol.json`、`results.json`、逐 Episode JSON、请求/动作/接触诊断、少量预览及 `records/` 下 HDF5。HDF5 记录真正传给环境的动作，T 动作对应 T+1 观测。异常标记 incomplete、success 为 null，不算任务失败，也不计入成功率分母；保留前面已完成的结果。

回放沿用原入口，将路径设为该次 `results.json` 中的 `record_path`：

```bash
.venv-libero-gpu/bin/python scripts/replay_libero.py --episode \
  evaluation_results/phase6a_stove/20261016T075146Z_8280fdb1/records/episode_000/episode_000.hdf5
```

这个具体路径是本机本阶段已执行的第一条炉灶轨迹，新实验要换成其实际路径。Replay 只重执行记录动作，不再次调用 VLA，不注入历史状态。

## 模型契约与两种协议

- 原始双路 RGB：256×256 HWC uint8；两轴反转一次，再调用官方 PIL `resize_with_pad` 到 224×224。两张相机及 8 维状态与固定版本官方代码在真实观测上逐元素完全一致，见 [input_parity.json](input_parity.json)。不复用 VisualBC 的 128px 预处理。
- 状态：末端世界 xyz（m）、轴角（rad）、两指 qpos（m）；prompt 使用原生任务语言。
- 服务端使用 Checkpoint 的 quantile 归一化/逆变换；返回 10×7 LIBERO OSC 增量动作，前六维世界平移/旋转增量，末维夹爪 -1 开/+1 关。客户端没有二次归一化。
- 每次执行前 5 个动作，尾部丢弃，然后读取新观测；Episode reset 清空队列。上游 client.reset 不重置服务器采样 RNG，服务 key(0) 随请求连续推进。要复现实验需重启服务并保持任务/请求顺序，env seed 不等于模型采样 seed。
- Official Reference：实际官方 helper/client/循环，seed7、10 dummy、goal 300步；本机复用当前 hf-libero 原生环境。原官方 Docker 固定的旧 LIBERO/MuJoCo/robosuite 与本机版本不同，因此这里不声称复现论文软件栈或论文成绩。
- EmbodiedLab Unified：原生官方初态，seed42按 Episode 顺序递增，10步 settling，goal300；正式实验先炉灶5个、再抽屉8个初态。`--evaluation-profile openpi_horizon` 只选择上游时限（spatial220 vs现有280），不代表完整官方评测协议。

本阶段真实结果：炉灶5/5，抽屉8/8；同初态协议重跑旧 BC 为3/5、4/8。模型、预训练数据、输入分辨率和 chunk 机制不同，结果是小样本工程验证，不能称严格公平算法比较或全 LIBERO Benchmark。

## 测试与限制

```bash
PYTHONPATH="$PWD/.deps/openpi-client-runtime" \
  .venv-libero-gpu/bin/python -m pytest -q -p no:cacheprovider
```

未安装可选客户端时，两个 OpenPI 专属测试模块明确跳过，核心 BC 流程仍可用。真实采样 chunk 的额外顺序回归可通过 `EMBODIEDLAB_OPENPI_REFERENCE_CHUNK` 指向本机 `first_real_chunk.npy` 启用；缺少该文件会明确跳过这一项，不伪造模型结果。

实际同步采样约0.383s/请求，另有首次编译和通信开销。同步仿真可以等待推理，不代表满足真机20Hz实时控制。同步采样计时包含 dispatch+GPU执行，并非纯 kernel profiler 时间。GPU峰值是离散NVML采样值，未必覆盖瞬时峰值。完整实验、错误定位、回归与限制见[Phase6A报告](../archive/PHASE6A_OPENPI_REPORT.md)。OpenPI代码/Gemma条款见根目录第三方声明，权重不再分发。
