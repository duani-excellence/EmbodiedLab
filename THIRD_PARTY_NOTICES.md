# 第三方来源与许可

EmbodiedLab原创代码的许可见[LICENSE](LICENSE)。本声明区分第三方代码、示范数据、仿真资产和安装依赖；这些材料不因被本项目使用而改变许可。仓库不打包官方HDF5、仿真资产、已安装环境、Checkpoint或用户实验结果。

## LIBERO

- 项目与作者：Lifelong Robot Learning；Bo Liu、Yifeng Zhu、Chongkai Gao、Yihao Feng、Qiang Liu、Yuke Zhu、Peter Stone，*LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*，2023。
- 使用的维护分支：[huggingface/LIBERO](https://github.com/huggingface/LIBERO/tree/8561c60eea2fb93096146f240194649df73d8b1e)，commit `8561c60eea2fb93096146f240194649df73d8b1e`，包`hf-libero 0.1.4`。
- **代码：MIT**，copyright 2023 Lifelong Robot Learning。固定源码的[LICENSE](https://github.com/huggingface/LIBERO/blob/8561c60eea2fb93096146f240194649df73d8b1e/LICENSE)提供原声明；再分发其代码或实质片段时应保留相应版权和许可声明。
- **官方示范数据：CC BY 4.0**。这是固定源码[README许可表](https://github.com/huggingface/LIBERO/blob/8561c60eea2fb93096146f240194649df73d8b1e/README.md#license)对Datasets的单独声明，不能将示范数据称为MIT。原始来源为[yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/147fa5eb607bb696dd922c3518084a90954cd2f3)，revision `147fa5eb607bb696dd922c3518084a90954cd2f3`。

本项目按任务下载原始示范，不改写原HDF5。公开的[数据审核证据](docs/data_sources/)是对来源指纹、约定和数值诊断的整理，去除了本机路径；其中单步诊断不代表完整专家轨迹复现。数据使用或再分发应按[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)保留归属、许可信息并说明改动。

## 仿真资产

固定资产来自[lerobot/libero-assets](https://huggingface.co/datasets/lerobot/libero-assets/tree/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6)，revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`。本地核查的快照未提供独立的LICENSE、README、COPYING或NOTICE，不能据LIBERO代码的MIT声明推定全部网格和纹理也为MIT。

本仓库仅保存来源pin和下载代码，**不再分发该资产快照**。资产的使用与再分发范围应依据其原始来源及各资产适用条款确认。

维护者本机历史`pre-libero-migration`中的Franka FR3模型来自MuJoCo Menagerie commit `0059d4335f8156206f63a35662313385f7ad6d74`，原目录带有Apache-2.0及附加BSD条款和来源记录；该历史与资产未随公开版本推送。

## 主要安装依赖

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
