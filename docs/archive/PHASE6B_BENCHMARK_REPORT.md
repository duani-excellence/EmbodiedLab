# Phase 6B：Research Benchmark Expansion

日期：2026-10-16。验收结论：**PASS**。

原始OpenPI π₀.₅与官方联合OpenVLA-OFT已在两个Suite、六个固定任务上完成 **120/120条真实闭环**。OpenPI **59/60**，OFT **60/60**，六任务宏平均分别 **98.33% / 100%**。正式实验无异常、无漏跑，唯一失败为原生520步超时。另完成两模型各六任务Smoke、OFT官方函数直接循环与Adapter等价验证、真实断连处理及旧BC回归。

没有训练、微调、量化、修改原始权重、LIBERO动力学/控制器/成功判定或添加手工控制。没有Push GitHub。LIBERO-PRO只完成可行性审查，**没有PRO闭环成绩**。

这些是每任务10初态的小样本、固定六任务挑战集结果。两个模型均已在LIBERO上fine-tuned；它们不是未见任务泛化实验，也不是LIBERO-130或论文成绩复现。

## 1. 固定任务与实验协议

任务ID、顺序及语言来自固定hf-libero `8561c60eea2fb93096146f240194649df73d8b1e`，与用户计划一致，没有替换任务。每任务均有50个官方初态。

| Suite / ID | 完整原生Task.language | 策略时限 |
|---|---|---:|
| libero_10 / 2 | turn on the stove and put the moka pot on it | 520 |
| libero_10 / 3 | put the black bowl in the bottom drawer of the cabinet and close it | 520 |
| libero_10 / 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 520 |
| libero_10 / 9 | put the yellow and white mug in the microwave and close it | 520 |
| libero_spatial / 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 220 |
| libero_spatial / 1 | pick up the black bowl next to the ramekin and place it on the plate | 220 |

前四项覆盖多子目标、放置后关门等长时序操作；后两项检验相近场景中的语言空间条件。没有人为添加扰动。Spatial的BDDL内部文字与Task.language存在命名差异；Prompt严格使用官方Suite的Task.language。

`benchmark_manifest.json` 在正式实验前固定，SHA256：

```text
7fab88133ff8816d4867389114c80da1b5f935c1aadf2e135deeff9f4ac0b105
```

- 每模型每任务正式初态 **0–9**、对应环境seed **42–51**；任务及初态顺序固定。
- Smoke单独使用初态49、seed7，不计入正式分母；正式开始前重启模型服务。
- 原生hard reset→官方set_init_state→10次`[0,0,0,0,0,0,-1]` settling；不注入专家历史状态。
- 仿真控制20Hz、每动作0.05s；settling不计策略Episode长度。520/220步与两模型固定官方评测循环一致；LIBERO任务本身没有一个独立于评测实现的唯一时限。旧BC/环境Spatial默认280步保留，本挑战集显式使用220。
- 两模型共用相同环境、初态、seed、时限、原生成功判定、Recorder与统计代码。原生成功终止，超时截断；异常标记incomplete、success=null并单列，不伪装任务失败。
- PI官方key(0)按请求连续推进，不因Episode reset重置；OFT使用seed42、L1/eval/do_sample=False。两者随机流不同，均记录服务身份与全局请求序号。
- 不依据表现更换初态。正式120条中，完成120、异常0、未执行0。

Manifest还保存任务BDDL/init及前10条状态指纹、软件版本、环境锁SHA、图像/动作/chunk协议和代码提交。冻结时协议commit为`229a3ff8c47a154fe64b734e1b62b86b8d376ff2`；OFT正式服务另记录实际wrapper SHA和`b2468acadae8411e4c2e90acfe48e6529e414133`。后续文档、独立入口默认值和防漂移检查未改动manifest或已运行结果。

## 2. 模型、来源与实际运行环境

| 项目 | OpenPI π₀.₅ | OpenVLA-OFT |
|---|---|---|
| 官方源码commit | `215abfb217dbac7d5f1273282331b9b1866c0479` | `e4287e94541f459edc4feabc4e181f537cd569a8` |
| Checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` |
| HF revision | 不适用；GCS generation逐对象固定 | `638918f3d1c2e43a39a8a20772bdb8b91835e4b7` |
| 原件 | 16文件，12,439,085,481 B | 25文件，15,939,168,050 B |
| 独立环境 | Python3.11.15 / JAX0.5.3 | Python3.10.20 / Torch2.9.1+cu128 |
| 精度/后端 | 官方JAX Orbax | 官方BF16连续L1 head、无量化、官方fork双向SDPA |

原件内容指纹：

```text
OpenPI: d9104dfdea46eca2fadf05ec7fc478b19d39b19aa8dfda0e6adedcd6d6b6efac
OFT:    a7a086ac19e0f16b7d8afd2feca1246f6f563a3ced306ad5e196f11458b9216e
```

OFT独立环境还固定torchvision0.24.1+cu128、timm0.9.10、TF CPU2.15.0、tokenizers0.19.1、peft0.11.1。官方Transformers fork为`bc339d9ad707454c0c115970db43c260067c61ab`，dlimp为`040105d256bd28866cc6620621a3d5f7b6b91b46`。普通HF Transformers4.40.1不能替代：官方fork的SDPA使用双向注意力，实际加载时断言`attention=sdpa`。

主机RTX5090 D v2、24,455MiB、驱动580.173.02；实际CUDA能力12.0。仿真环境仍为Python3.10.20、MuJoCo3.3.2、robosuite1.4.0、hf-libero0.1.4，资产revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`。GPU/CPU原环境未改动。

**兼容性差异明确保留**：官方OFT论文软件栈为Torch2.2/A100，本机为支持Blackwell的Torch2.9.1/cu128；没有声称逐位复现论文。OpenPI沿用Phase6A的独立ptxas12.9.41与显式`--xla_gpu_enable_triton_gemm=false`，不改模型权重。TF CPU避免占用模型显存。没有安装系统CUDA Toolkit或修改驱动。

OFT首次安装遇到上游旧Torch约束、过宽解析及网络下载问题，全部在新环境修复。实际CUDA12.8数值包曾从已验证GPU环境**复制普通文件**并逐RECORD核验，没有硬链接或改写旧环境；稳定后移除新环境中无用的三个CUDA13元包/绑定。独立锁保留实测版本、使用`--no-deps`，不会自动回装旧Torch。`uv pip check`仍报告11项上游完整训练元数据缺口/版本声明：旧Torch/Torchvision/Torchaudio、`tensorflow`与`tensorflow-cpu`发行名、TF Graphics/Probability可选依赖。实际官方推理import、BF16 SDPA及67条完整OFT原生/Adapter Episode均正常；此环境只保证本次推理，不宣称支持其训练与完整图形导入链。

## 3. 协议正确性与真实推理证据

模型共同输入为两张原生256×256 HWC RGB uint8、raw8机器人状态及任务语言。state顺序：末端世界xyz米、轴角弧度、两指qpos米。七维动作是OSC位姿增量和夹爪，不是七个关节角。两模型各自处理：

| 环节 | OpenPI | OFT |
|---|---|---|
| 相机 | 180°一次→官方PIL pad resize224 | 180°一次→官方TF JPEG encode/decode及Lanczos3 resize224→90%**面积**中心crop→官方DINO/SigLIP Processor |
| 状态/动作统计 | 原Checkpoint quantile | 当前Suite的`*_no_noops` q01/q99，不能混用Spatial与10 |
| 文本 | 原生Task.language | `In: What action should the robot take to {language.lower()}?\nOut:` |
| Action chunk | 10×7，执行前5个，尾部丢弃 | 8×7，执行完整8个再观察 |
| 夹爪 | 原官方逆变换；延用显式、已验证原生等价饱和 | 官方normalize/binarize再invert，即`-sign(2*g-1)`；客户端不二次缩放/裁剪 |

PI的`--native-action-bounds`沿用Phase6A已实测等价转换，raw chunk与每次投影完整记录；没有增加启发式。OFT调用官方夹爪函数，原始夹爪略超[0,1]的实际输出没有在客户端先行裁剪。本次全部OFT最终动作满足环境单位区间。

OFT真实加载使用四份已合并safetensors、官方action head/proprio projector，不使用下载仓库中的LoRA重新合并。参数数目：base **7,541,237,184**，action head **151,117,831**，proprio projector **16,818,176**。

官方local loader会同步config/model Python；因此创建小文件副本、大权重symlink的独立运行视图，原下载目录每文件复核。HF模型代码与固定源码的差异仅在未使用diffusion分支的两行scheduler逻辑，本次L1路径相同。

实证链条：

1. 真实native观测逐项对照官方helper：resize/crop像素、state、Prompt完全一致；官方Processor单相机输出`(1,6,224,224)` float32。
2. 同一真实观测通过直接模型请求与Adapter的两次实际前向，全部8×7 raw/final动作及处理指纹最大差为 **0**。
3. 不经PolicyAdapter的官方函数直接循环：Spatial0/init49/seed7真实 **73步成功**。独立重置后的Adapter Smoke同为73步，全部动作最大差 **0**。没有修改成功标准或恢复历史状态。
4. 六任务OFT Smoke全部成功，长度258/217/231/258/73/101；PI六任务Smoke也全部成功。随后各自新服务正式运行。
5. 独立审查120条：PI **2390**请求序号连续0–2389，OFT **1450**次连续0–1449；每集请求时刻精确符合`range(0,T,5/8)`。所有请求语言正确，raw8 state逐值等于相应HDF **pre-action** state。所有 **23,250**个实际动作逐值符合各自官方解码与缓存顺序。

原始证据保存在chat `work/phase6b/`的`oft_preprocessing_parity.json`、`oft_live_adapter_parity.json`、`oft_closed_loop_parity.json`、`independent_request_action_audit.json`，以及本地evaluation_results下的official_reference、smoke与formal目录。完整源码定位见[OFT协议审计](../benchmark/openvla_protocol.md)。

## 4. 逐任务正式结果

首次成功均使用原生判定；表内首次成功均值只对成功Episode计算，长度均值包含超时。Completion Time是策略长度×0.05秒，不包含settling或推理等待。

| 模型 | Suite/ID | 成功/完成 | 成功率 | Wilson95% | 平均长度 | 成功集平均首次成功步 | 平均仿真秒 | 请求数 |
|---|---|---:|---:|---|---:|---:|---:|---:|
| PI | 10/2 | 9/10 | 90% | 59.59–98.21% | 272.6 | 245.11 | 13.630 | 548 |
| PI | 10/3 | 10/10 | 100% | 72.25–100% | 231.7 | 231.70 | 11.585 | 467 |
| PI | 10/4 | 10/10 | 100% | 72.25–100% | 235.2 | 235.20 | 11.760 | 475 |
| PI | 10/9 | 10/10 | 100% | 72.25–100% | 260.2 | 260.20 | 13.010 | 524 |
| PI | Spatial/0 | 10/10 | 100% | 72.25–100% | 78.9 | 78.90 | 3.945 | 162 |
| PI | Spatial/1 | 10/10 | 100% | 72.25–100% | 105.4 | 105.40 | 5.270 | 214 |
| OFT | 10/2 | 10/10 | 100% | 72.25–100% | 244.0 | 244.00 | 12.200 | 307 |
| OFT | 10/3 | 10/10 | 100% | 72.25–100% | 233.8 | 233.80 | 11.690 | 297 |
| OFT | 10/4 | 10/10 | 100% | 72.25–100% | 223.9 | 223.90 | 11.195 | 284 |
| OFT | 10/9 | 10/10 | 100% | 72.25–100% | 256.2 | 256.20 | 12.810 | 325 |
| OFT | Spatial/0 | 10/10 | 100% | 72.25–100% | 79.0 | 79.00 | 3.950 | 103 |
| OFT | Spatial/1 | 10/10 | 100% | 72.25–100% | 104.1 | 104.10 | 5.205 | 134 |

PI总动作11,840、OFT 11,410。六任务宏平均 **98.33% / 100%**，每任务分母均10；相同分母使本次宏平均恰好等于合并比例，但不将其称为完整LIBERO成功率。逐初态数据见[paired CSV](phase6b/paired_initial_states.csv)，精确数组/延迟见[results.json](phase6b/results.json)。

60对同初态中：共同成功59对、仅OFT成功1对、仅PI成功0、共同超时0。59对共同成功的OFT−PI长度差均值−2.525步、中位−1步，范围−110至100。只作描述，不做显著性或总体算法优越性结论。

## 5. 失败与子目标分析

唯一超时：**PI / libero_10 task2 / init7 / seed49 / 520步**。原生`turnon flat_stove_1`第78步首次为true并保持；`on moka_pot_1 flat_stove_1_cook_region`在所记录的所有策略步从未true。可确认失败在最终壶放置条件未完成，不能仅凭视频断言感知、抓取、姿态或动作预测哪个是根因。同初态OFT在239步原生成功。

其余可验证差异：PI双杯任务第一杯`on porcelain_mug_1 plate_1`有 **4/10**集曾满足后丢失，随后恢复，最终全部成功；OFT对应 **0/10**。这描述谓词保持性，不把短暂丢失自动归因为碰撞或策略错误。

抽屉/微波炉任务分别记录容器`in`及`close`原生谓词的首次满足。不能虚构“已经抓牢”阶段：原生`in`主要验证几何包含，接触检查并不能证明稳定抓取；本次formal的close首次true均在最终成功步。完整20行逐初态分布见[native_predicate_progress.md](phase6b/native_predicate_progress.md)。接触数/力仅是策略控制步末端采样，不能称为物理子步峰值。

生成 **13个视频**：每模型每任务首个成功，共12个；另加唯一PI超时。全部通过FFmpeg首尾解码、512×256双相机、20fps、T+1帧和180°展示方向检查。视频是动作时间回放，包含初始帧，不表达RPC真实等待时间。路径在 `evaluation_results/phase6b_analysis_final/videos/`，验证见同目录`video_validation.json`。原HDF像素没有翻转或改写。

## 6. 延迟、运行时间和资源

| 模型 | RPC均值/P50/P95 ms | 同步sample均值/P95 ms | 每步Policy.act均值 ms | native step均值 ms | 正式六任务wall秒 | NVML观测峰值 |
|---|---|---|---:|---:|---:|---:|
| PI | 405.82 / 404.68 / 407.88 | 399.29 / 401.26 | 82.23 | 7.87 | 1174.33 | 8990MiB |
| OFT | 97.94 / 97.50 / 99.61 | 87.60 / 88.76 | 12.48 | 7.80 | 335.90 | 16150MiB |

PI同步sample计时为JAX sampler dispatch+实际device执行；OFT为同步官方`model.predict_action`，含相关head/decode/CPU逻辑。不是相同范围的纯kernel profiler。RPC包含序列化/传输与服务端处理，OFT另保存完整官方pipeline计时。Policy.act多数步骤取缓存，不能将其低P50当成每次模型推理时间。

正式计时包含按任务构建环境、写HDF与诊断；不包含此前下载、模型hash校验、原生reference和Smoke。PI使用已有编译缓存，正式首RPC1304.84ms；OFT首RPC423.20ms。PI期间部分回归测试短暂共享主机，时延是本次实测系统负载下的观测值，不是隔离硬件吞吐Benchmark。

NVML分别975/557个有效样本，PI观测从正式运行中途开始，未覆盖完整冷启动。两服务观测范围内显存均稳定，离散采样不保证捕获瞬时峰值；OFT正式加载后Torch allocated14748.98MiB。大模型顺序运行，服务结束后两个PID均从NVML及`/proc`消失，其他用户进程未终止。没有声称整张GPU空闲或已经完成长期内存泄漏证明。同步仿真等待RPC，不代表真机20Hz实时控制。

## 7. 代码结构与运行入口

主要新增/修改：

| 文件 | 当前实际用途 |
|---|---|
| `src/embodiedlab/policies/openvla_oft.py` | OFT原始输入、独立官方动作协议、8步缓存、超时与close |
| `tools/serve_openvla.py` | 隔离官方模型进程、只读原件/运行视图、GPU与来源核验、实际同步计时 |
| `scripts/download_openvla.py` | 固定HF revision下载/续传与逐对象验证，不下载数据集 |
| `scripts/evaluate_openpi.py`、`evaluate_openvla.py` | 复用现有评测编排，两个明确模型分支，独立协议验证 |
| `scripts/run_challenge.py` | 按冻结顺序执行任务、分开Smoke/formal、保存run_index、异常停止 |
| `src/embodiedlab/benchmark.py` | 原生谓词进展、逐任务分母、Wilson区间，异常不计任务失败 |
| `tools/analyze_challenge.py` | 只读真实结果/记录，逐任务及配对分析、流式生成视频 |
| `src/embodiedlab/evaluation.py` | 增加原生goal原子诊断，不将privileged state输入Policy |
| `tests/test_openvla_policy.py`、`test_benchmark_summary.py`、`test_challenge_analysis.py` | 协议、缓存、异常、统计与损坏记录检查 |
| `benchmark_manifest.json`、`openvla-upstream.json`、`requirements-openvla.lock` | 冻结实验与来源/运行时身份 |
| `docs/benchmark/`、根README/第三方声明 | 实际命令、数值结果、协议与PRO审查 |

没有改VisualBC网络、训练逻辑、Environment动力学、Runner、Recorder、Dataset或Replay核心。没有新增Registry/通用Trainer。复用少量NumPy/msgpack传输，并不复用PI的模型处理。

执行链：固定manifest→顺序任务CLI→模型专属Policy→现有Evaluator/RolloutRunner→原生LiberoEnv→Recorder/HDF；新观测驱动下一次模型请求。离线分析只读原始文件，Replay重新执行记录动作，不再次调用模型。

详细安装与重跑命令见[使用说明](../benchmark/README.md)。本次真实formal命令为：

```bash
# 已配置README中的EGL、LIBERO_CONFIG_PATH及可选客户端PYTHONPATH。
.venv-libero-gpu/bin/python scripts/run_challenge.py --model openpi \
  --manifest benchmark_manifest.json \
  --server-manifest evaluation_results/phase6b_openpi_formal_server/manifest.json \
  --output-dir evaluation_results/phase6b_formal_openpi
.venv-libero-gpu/bin/python scripts/run_challenge.py --model openvla_oft \
  --manifest benchmark_manifest.json \
  --server-manifest evaluation_results/phase6b_oft_formal_server/manifest.json \
  --output-dir evaluation_results/phase6b_formal_openvla
.venv-libero-gpu/bin/python tools/analyze_challenge.py \
  --manifest benchmark_manifest.json \
  --openpi-dir evaluation_results/phase6b_formal_openpi \
  --openvla-dir evaluation_results/phase6b_formal_openvla \
  --output-dir evaluation_results/phase6b_analysis_final --videos
```

这些正式输出已存在，重跑必须使用全新输出、运行视图及服务manifest路径，且重启模型随机流；不要直接覆盖已有成果。

## 8. 回归、回放和数据保护

最终完整回归：**GPU 308 passed，28.49s；CPU 304 passed、4 skipped，28.09s**。CPU跳过项均为CUDA专属检查，没有测试失败；保留上游弃用/private macro等14/13条警告。可选真实PI chunk已通过显式路径启用。

实际命令（沿用前述EGL/cache环境变量）：

```bash
export PYTHONPATH="$PWD/.deps/openpi-client-runtime"
export EMBODIEDLAB_OPENPI_REFERENCE_CHUNK="$PWD/evaluation_results/phase6a_reference_stove_no_triton_seed7/first_real_chunk.npy"
.venv-libero-gpu/bin/python -m pytest -q -p no:cacheprovider
.venv-libero/bin/python -m pytest -q -p no:cacheprovider
```

新增防漂移检查在首次reset前核对实际BDDL/init字节、前10条初态SHA、任务与环境版本；OFT握手要求seed42/BF16/非量化/官方SDPA，PI核对key(0)协议。六个真实环境仅构造、不reset/step的资产核对全部通过；保存的真实握手通过。该后置工程加固没有伪装成重新运行120条正式实验。九个新旧命令行入口的`--help`全部正常。

固定OFT下载入口既实际完成运行后全量`--verify-only`，也通过真实网络重新下载21字节官方`added_tokens.json`验证fetch路径；其余大对象只读链接核验，没有为测试重复下载16GB。主要验收数值和证据日志指纹汇总在[validation_summary.json](phase6b/validation_summary.json)。

- 旧BC真实原生闭环重新执行13条：炉灶E5 **3/5**，抽屉 **4/8**；与原实验一致，两Checkpoint SHA不变。
- PI回放3条：长成功247步、长超时520步、Spatial成功109步；OFT回放2条：长成功245步、Spatial成功75步。五次Replay的robot state/joint/RGB/reward/time最大偏差均 **0**，长度/语言/terminated/truncated/success一致。这只证明本次同软件栈自录轨迹的实际回放，未假定浮点普遍完全一致。
- 真实OFT断连：一次模型请求、执行8动作后关闭实际socket；下一请求报ConnectionClosedOK。保存partial HDF、complete=false、success=null，标准EpisodeDataset明确拒绝其作为完整数据，env/policy/recorder均释放。此诊断不计入正式分母。
- 120集HDF全部T动作/T+1观测，通过seed/init/时限/manifest及诊断交叉核验。387个原JSON/HDF/manifest、共3,536,270,267字节，分析前后流式SHA全部相同。
- 两模型原件在运行后全部重新SHA验证；未改原始下载。旧数据/训练/评测成果共462个文件重新hash，原虚拟环境/旧缓存共82,389条原有记录核对，issues=[]。没有删除旧数据、模型、环境或项目外文件。
- 本地`phase6a-openpi`与恢复tag `phase6a-verified-20261016`保留commit300aba2；当前独立`phase6b-benchmark`。main/v0.1.0及远端仍为4a67c64，没有推送。权重、HDF、视频、环境和缓存均在忽略目录，不纳入源码Git。

## 9. LIBERO-PRO可行性：本次未运行扰动闭环

审查固定Git `eafdb809426b13153aa1e4c42d6601844217dfec` 与HF revision `c86fc3b8293185a6f373677018ff3e37f8391602`，检查100对旧x/y BDDL/init和少量新格式样本，未安装PRO完整运行环境或覆盖标准资产。

确认的工程问题：

1. `use_swap`是初始区域关系交换，不能当成只平移目标。Git旧x/y与HF新版07配置不是同一资产集合。
2. 旧x0.1实际目标平移约 **0.07m**；较高档部分其他区域移出 **10m**，还有区域宽度变化，不能只凭目录名称为合理单变量扰动。
3. 100份旧init中99份为(50,110)，一份y0.5 cream-cheese为空(0,)。
4. 同初态编号的原始/扰动机器人qpos及其他物体也会变化，前7qpos最大差约0.0646rad，不能称严格反事实配对。
5. 当前固定PRO Git parser实际忽略HF新`:perturbation_config`，只运行后会有“扰动未生效”的风险。静态验证通过不能代替运行时核验。

原生goal仍存在，但当前组合下的可达性、实际扰动、生效初态与成功判定尚未通过仿真确认。因此本阶段**不强行接入或给出PRO分数**。可行方案是独立环境/配置，先选择非空温和档或明确的swap，核验实际状态/原生goal，单独manifest与分母；不改当前LIBERO资产。两模型原始权重理论上可复用，但必须先实际验证对应输入/控制接口，不能当成已验证事实。证据和最小方案见[可行性报告](libero_pro_feasibility.md)。

## 10. 结论、边界与下一步

**工程验收PASS**：两个原始大模型、两个Suite、六个固定任务真实推理/闭环/记录/分析/回放跑通；旧BC与数据链路保留，失败与异常有明确分母语义。

**研究解释边界**：119/120的本次成功率说明这组固定初态对两个LIBERO fine-tuned模型总体偏容易；每格10条的Wilson区间很宽，且只出现一个成对差异，不能支撑总体模型排名或跨分布稳健性结论。阶段谓词可定位未完成目标，不能独自定位感知/控制根因。

下一阶段优先解决一个问题：**取得并独立验证物理合理、确实生效且身份清楚的分布变化评测协议**，再做原始/扰动配对分析。此项尚未执行；不自动扩展完整LIBERO-130、不训练模型、不Push远端。
