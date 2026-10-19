# 第三方来源与许可

EmbodiedLab原创代码的许可见[LICENSE](LICENSE)。本声明区分第三方代码、示范数据、仿真资产和安装依赖；这些材料不因被本项目使用而改变许可。仓库不打包官方HDF5、仿真资产、已安装环境、Checkpoint或用户实验结果。

## LIBERO

- 项目与作者：Lifelong Robot Learning；Bo Liu、Yifeng Zhu、Chongkai Gao、Yihao Feng、Qiang Liu、Yuke Zhu、Peter Stone，*LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*，2023。
- 使用的维护分支：[huggingface/LIBERO](https://github.com/huggingface/LIBERO/tree/8561c60eea2fb93096146f240194649df73d8b1e)，commit `8561c60eea2fb93096146f240194649df73d8b1e`，包`hf-libero 0.1.4`。
- **代码：MIT**，copyright 2023 Lifelong Robot Learning。固定源码的[LICENSE](https://github.com/huggingface/LIBERO/blob/8561c60eea2fb93096146f240194649df73d8b1e/LICENSE)提供原声明；再分发其代码或实质片段时应保留相应版权和许可声明。
- **官方示范数据：CC BY 4.0**。这是固定源码[README许可表](https://github.com/huggingface/LIBERO/blob/8561c60eea2fb93096146f240194649df73d8b1e/README.md#license)对Datasets的单独声明，不能将示范数据称为MIT。原始来源为[yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/147fa5eb607bb696dd922c3518084a90954cd2f3)，revision `147fa5eb607bb696dd922c3518084a90954cd2f3`。

本项目按任务下载原始示范，不改写原HDF5。公开的[数据审核证据](docs/data_sources)是对来源指纹、约定和数值诊断的整理，去除了本机路径；其中单步诊断不代表完整专家轨迹复现。数据使用或再分发应按[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)保留归属、许可信息并说明改动。

## 仿真资产

固定资产来自[lerobot/libero-assets](https://huggingface.co/datasets/lerobot/libero-assets/tree/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6)，revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`。本地核查的快照未提供独立的LICENSE、README、COPYING或NOTICE，不能据LIBERO代码的MIT声明推定全部网格和纹理也为MIT。

本仓库仅保存来源pin和下载代码，**不再分发该资产快照**。资产的使用与再分发范围应依据其原始来源及各资产适用条款确认。

维护者本机历史`pre-libero-migration`中的Franka FR3模型来自MuJoCo Menagerie commit `0059d4335f8156206f63a35662313385f7ad6d74`，原目录带有Apache-2.0及附加BSD条款和来源记录；该历史与资产未随公开版本推送。

## OpenPI（可选）

- 官方源码：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)，commit `215abfb217dbac7d5f1273282331b9b1866c0479`。代码主许可为[Apache-2.0](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE)，包括本项目调用的官方客户端及图像工具；源码保存在本地忽略的依赖目录，不复制模型实现到本项目。
- 模型使用 Gemma/PaliGemma 组件；上游另附 [LICENSE_GEMMA.txt](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE_GEMMA.txt)。模型及衍生物应遵守其单独条款和使用限制，不能据代码的Apache许可推定所有权重许可相同。
- 官方 `gs://openpi-assets/checkpoints/pi05_libero` 的16个对象、generation及SHA256保存在 `openpi-upstream.json`。本仓库只提供来源、校验和可选集成代码，不再分发权重、tokenizer、JAX环境或CUDA二进制；不为这些第三方材料另行声明MIT许可。
- 独立推理环境按上游 `uv.lock` 安装，仍适用各发行包及CUDA运行库的原始条款。相关命令和本机兼容措施见 [OpenPI说明](docs/openpi/README.md)。

## OpenVLA-OFT（可选）

- 官方代码：[moojink/openvla-oft](https://github.com/moojink/openvla-oft/tree/e4287e94541f459edc4feabc4e181f537cd569a8)，固定commit `e4287e94541f459edc4feabc4e181f537cd569a8`。源码[MIT许可证](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/LICENSE)归属2025 Moo Jin Kim、Chelsea Finn、Percy Liang。
- 官方Transformers fork固定 `bc339d9ad707454c0c115970db43c260067c61ab`，保留上游Transformers及其修改的适用声明；不能以普通HF包替代其双向SDPA语义。dlimp来源与版本也记录在 `openvla-upstream.json`。
- 联合Checkpoint：[固定HF revision](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/tree/638918f3d1c2e43a39a8a20772bdb8b91835e4b7)，模型卡标注MIT。基础语言/视觉组件及其衍生物仍适用各自条款；不将第三方模型统一重新许可为本项目MIT。
- 仓库只提供模型调用、独立环境锁和每对象校验身份，不分发25个权重/配置对象、LoRA、tokenizer或CUDA二进制。调用官方加载器时使用独立运行视图，保留下载原件不变。
- LIBERO-PRO只做固定版本可行性审查，来源链接见 `docs/archive/libero_pro_feasibility.md`；没有把其资产复制进标准LIBERO安装，也没有分发其初始化文件。

## Isaac Lab Factory（可选）

- 官方 [Isaac Lab v2.3.0](https://github.com/isaac-sim/IsaacLab/tree/3c6e67bb5c7ada942a6d1884ab69338f57596f77) 采用 BSD-3-Clause；[NVIDIA RL-Games fork](https://github.com/isaac-sim/rl_games/tree/6b3534f29568158e9e29ec8bf83cc88fce5f0cae) 采用 MIT。源码仅安装在忽略的依赖目录，本项目调用其原生 Factory 环境和 PPO。
- Isaac Sim 5.1、USD资产、官方PPO Checkpoint及CUDA组件分别遵循原始发行条款，不能统一视为本项目MIT。仓库不分发这些资产、权重或仿真器二进制。独立环境及来源指纹见 [Factory锁文件](requirements-factory.lock)。

## 主要安装依赖

归档分支`archive/ros2-phase7f-20261019`中的ROS 2环境使用[官方ROS Docker镜像](https://hub.docker.com/_/ros)、[franka_ros2 2.7.1](https://github.com/frankarobotics/franka_ros2/tree/3b50164eabbe2d461baca007bbbbe03c024387bd)、[franka_description 2.8.1](https://github.com/frankarobotics/franka_description/tree/02afaae282d4a8e10d7d2f781b23b3515c303ce5)和libfranka 0.20.4。三个Franka项目的主许可证为Apache-2.0；原始声明保留在忽略的源码缓存和安装包中。ROS及控制器依赖各自的原始许可证仍适用。来源、镜像与版本见该归档分支的`upstream-ros2.json`。该归档仅提供构建与调用代码，不打包这些源码、资产或容器二进制。

笛卡尔FK/IK另外调用[Pinocchio](https://github.com/stack-of-tasks/pinocchio) 4.0.0（BSD-2-Clause）和[SciPy](https://scipy.org/) 1.8.0（主BSD-3-Clause及其附带组件声明）。它们仅安装在ROS容器中，版本在Dockerfile显式锁定，许可证保留在原始安装包中。

下表依据本地已安装发行包中的LICENSE、NOTICE与元数据核查。完整依赖版本见[CPU锁](requirements-libero.lock)和[GPU锁](requirements-libero-gpu.lock)；各发行包自带的原始声明及第三方声明仍适用。

| 依赖 | 主许可证及说明 |
|---|---|
| MuJoCo 3.3.2 | Apache-2.0；另有`LICENSES_THIRD_PARTY.md` |
| robosuite 1.4.0 | MIT；其LICENSE还包含部分MuJoCo实现的Apache说明 |
| robomimic 0.2.0、bddl 1.0.1 | MIT |
| PyTorch 2.5.1 / 2.9.1 | BSD-3-Clause；附LICENSE/NOTICE及捆绑组件声明 |
| torchvision 0.20.1 / 0.24.1 | BSD |
| NumPy 1.26.4、h5py 3.11.0 | 主BSD-3-Clause；NumPy二进制所含OpenBLAS、GCC runtime等另有条款 |
| OpenCV、Transformers、HuggingFace Hub | Apache系列；OpenCV另有第三方组件声明 |
| Hydra、pytest | MIT |

GPU环境还安装NVIDIA CUDA运行库。CUDA runtime、cuDNN、cuBLAS、cuSPARSE等发行包包含NVIDIA专有许可；NCCL另有BSD-3-Clause许可。不能把整套GPU环境统称为MIT或全部开源。本项目通过锁文件引用官方发行包，不重新打包或再许可这些二进制。

如另行分发第三方源码、资产、数据或环境，应携带适用的原始许可与声明；本文件是来源索引，不能代替各组件的完整条款。
