# Phase 6B：固定六任务、多模型挑战集

从仓库根目录运行。先完成根README的LIBERO GPU环境、固定资产和 [OpenPI安装](../openpi/README.md)。已有环境与下载原件直接复用。模型服务与仿真分进程；**两个大模型顺序运行**。

## 1. OpenVLA-OFT独立环境

模型源代码保留在忽略目录，不复制进EmbodiedLab源码。以下commit必须与根 `openvla-upstream.json` 一致：

```bash
git clone https://github.com/moojink/openvla-oft.git \
  .deps/openvla-oft-e4287e94541f459edc4feabc4e181f537cd569a8
git -C .deps/openvla-oft-e4287e94541f459edc4feabc4e181f537cd569a8 \
  checkout --detach e4287e94541f459edc4feabc4e181f537cd569a8
git clone https://github.com/moojink/transformers-openvla-oft.git \
  .deps/transformers-openvla-oft-bc339d9ad707454c0c115970db43c260067c61ab
git -C .deps/transformers-openvla-oft-bc339d9ad707454c0c115970db43c260067c61ab \
  checkout --detach bc339d9ad707454c0c115970db43c260067c61ab
git clone https://github.com/moojink/dlimp_openvla.git \
  .deps/dlimp-openvla-040105d256bd28866cc6620621a3d5f7b6b91b46
git -C .deps/dlimp-openvla-040105d256bd28866cc6620621a3d5f7b6b91b46 \
  checkout --detach 040105d256bd28866cc6620621a3d5f7b6b91b46
uv venv --python 3.10 .venv-openvla
uv pip install --python .venv-openvla/bin/python --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match -r requirements-openvla.lock
```

锁中的OpenPI客户端路径来自前一步OpenPI源码，仅用于NumPy/msgpack通信；OFT模型没有使用OpenPI预处理或采样器。不要把此锁装入原LIBERO或OpenPI环境。

本机实际Python3.10.20、Torch2.9.1+cu128、torchvision0.24.1+cu128、官方fork Transformers4.40.1、TF **CPU** 2.15.0。官方旧Torch2.2/A100环境与RTX5090不同，故显式覆盖Torch版本并使用 `--no-deps`，不能让上游元数据重新装回旧Torch。官方import链需要部分TFDS/dlimp/Graphics模块，但不安装训练、音频、OpenEXR等完整可选能力。`uv pip check`仍会报告旧Torch声明及`tensorflow`与`tensorflow-cpu`发行名等未满足项；这是一份经过实际推理验证的隔离推理环境，不声称支持上游完整训练依赖。无需安装系统CUDA Toolkit或FlashAttention；实际使用官方fork的双向SDPA。

获取固定联合Checkpoint并写独立校验报告：

```bash
.venv-libero-gpu/bin/python scripts/download_openvla.py \
  --verification-file .deps/openvla_checkpoint_verified.json
```

下载25个固定对象，15,939,168,050 bytes；每个大小/SHA256均由固定上游身份校验，已有不匹配文件会保留并报错。复核时使用 `--verify-only` 和**新的** `--verification-file` 路径。权重默认保存 `.deps/openvla-cache/638918f3d1c2e43a39a8a20772bdb8b91835e4b7/`，不下载专家示范，不使用量化或重新合并LoRA。

## 2. 固定协议与顺序

`benchmark_manifest.json` 在正式实验前冻结，SHA256为 `7fab88133ff8816d4867389114c80da1b5f935c1aadf2e135deeff9f4ac0b105`。不要按成绩修改任务、初态或种子。顺序为10/2、10/3、10/4、10/9、Spatial/0、Spatial/1；每任务正式init0–9、seed42–51。Smoke使用init49/seed7，另存目录。

共同原生协议：reset→官方初态→10个开夹爪dummy动作；20Hz，10任务520步、Spatial220步。模型输入双RGB原始256px、raw8 state及原生语言。模型差异必须保留：

| 项目 | OpenPI π₀.₅ | OpenVLA-OFT |
|---|---|---|
| 图像 | 180°一次，官方PIL pad resize224 | 180°一次，官方TF JPEG/Lanczos224、90%面积中心crop、官方Processor |
| 归一化 | Checkpoint quantile | 当前Suite的`*_no_noops` q01/q99 |
| 输出/执行 | 10动作，执行前5个 | 8动作，执行全部8个 |
| 夹爪 | 官方逆变换，显式原生等价饱和 | 官方 `-sign(2*g-1)`，客户端不再变换 |
| 模型随机流 | key(0)按每请求推进，跨集连续 | seed42、L1/eval/do_sample=False，记录请求序号 |

查看任务可沿用 `scripts/evaluate_libero.py --suite libero_10 --list-tasks`。单任务OFT入口为 `scripts/evaluate_openvla.py`，参数与OpenPI编排入口一致；其独立默认采用模型官方时限，旧OpenPI默认仍保留原配置，挑战集始终由manifest覆盖时限。

## 3. 重跑挑战集

以下使用新的 `phase6b_repeat_01` 路径，避免覆盖原实验。已有这些路径时必须更换后缀。原正式PI/OFT结果分别保留在 `evaluation_results/phase6b_formal_openpi/` 与 `phase6b_formal_openvla/`。

先按OpenPI说明启动**新的**PI服务manifest，保持官方key0，然后在另一个终端：

```bash
export PYTHONPATH="$PWD/.deps/openpi-client-runtime${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PWD/.libero-gpu" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/embodiedlab-gpu-numba MPLCONFIGDIR=/tmp/embodiedlab-libero-matplotlib
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset DISPLAY WAYLAND_DISPLAY
.venv-libero-gpu/bin/python scripts/run_challenge.py --model openpi \
  --manifest benchmark_manifest.json \
  --server-manifest evaluation_results/openpi_server/run01.json \
  --output-dir evaluation_results/phase6b_repeat_01_openpi
```

Smoke在正式运行前用 `--phase smoke` 和独立输出目录执行；正式运行前重启服务，避免Smoke消费PI随机流。不要在正式任务之间重启/插入其他请求。

PI完成后停止其服务，再启动OFT：

```bash
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv-openvla/bin/python tools/serve_openvla.py \
  --checkpoint .deps/openvla-cache/638918f3d1c2e43a39a8a20772bdb8b91835e4b7 \
  --verification .deps/openvla_checkpoint_verified.json \
  --runtime-dir .deps/openvla-runtime-repeat_01 \
  --manifest evaluation_results/phase6b_repeat_01_oft_server/manifest.json \
  --port 8007 --seed 42
```

服务在完整校验并加载后才创建manifest。官方local loader会同步少量模型Python/config，因此本入口复制小配置、将大权重链接至校验原件，避免改写原始下载。OFT服务仅监听loopback，TF仅做CPU图像处理。

另一个终端沿用前面的EGL/客户端环境：

```bash
.venv-libero-gpu/bin/python scripts/run_challenge.py --model openvla_oft \
  --manifest benchmark_manifest.json \
  --server-manifest evaluation_results/phase6b_repeat_01_oft_server/manifest.json \
  --output-dir evaluation_results/phase6b_repeat_01_openvla
```

任务成功与超时均保留；异常立即停止，partial HDF标记incomplete、success=null，不计成功率分母。不得筛选成功初态重跑覆盖结果。

## 4. 分析、视频和回放

本阶段已实际执行的分析入口：

```bash
.venv-libero-gpu/bin/python tools/analyze_challenge.py \
  --manifest benchmark_manifest.json \
  --openpi-dir evaluation_results/phase6b_formal_openpi \
  --openvla-dir evaluation_results/phase6b_formal_openvla \
  --output-dir evaluation_results/phase6b_analysis_repeat_01 --videos
```

`--output-dir`必须不存在。JSON、Markdown、paired CSV明确列出每任务分母、Wilson95%、首次成功步及请求/延迟。视频从HDF逐帧读取，原像素不变，展示180°旋转。`results.json`中的 `record_path` 可直接传给：

```bash
.venv-libero-gpu/bin/python scripts/replay_libero.py --episode \
  evaluation_results/phase6b_formal_openvla/libero_10/task_2/20261016T143152Z_4317c5e6/records/episode_000/episode_000.hdf5
```

实际目录时间戳应以本机 `run_index.json` / `results.json` 为准，不要猜测路径。Replay重执行记录动作，不调用模型、不向仿真注入历史状态。正式120条原数据、详细分析和13个视频均留在忽略的本地 `evaluation_results/`。

回归命令：

```bash
export EMBODIEDLAB_OPENPI_REFERENCE_CHUNK="$PWD/evaluation_results/phase6a_reference_stove_no_triton_seed7/first_real_chunk.npy"
PYTHONPATH="$PWD/.deps/openpi-client-runtime" \
  .venv-libero-gpu/bin/python -m pytest -q -p no:cacheprovider
PYTHONPATH="$PWD/.deps/openpi-client-runtime" \
  .venv-libero/bin/python -m pytest -q -p no:cacheprovider
```

本机最终GPU 308通过，CPU 304通过/4项CUDA检查跳过。新克隆没有历史真实chunk文件时，取消上述export，相关可选测试会明确跳过；不得伪造该采样文件。

范围：六个固定任务、每模型每任务10初态。两个Checkpoint都针对LIBERO fine-tuned，不能将其成绩称为未见任务泛化或论文完整Benchmark。模型同步RPC延迟也不等于真机20Hz实时能力。LIBERO-PRO只完成独立可行性审查，尚未通过真实扰动闭环验收，不能混入本表。
