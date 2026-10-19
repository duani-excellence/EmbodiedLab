# Phase 6A — 官方 OpenPI π₀.₅ 集成与真实 LIBERO 闭环验收

日期：2026-10-16。**验收：PASS（本阶段工程集成范围）**。官方权重在 RTX 5090 D v2 上真实加载与推理；固定炉灶初态5/5、抽屉8/8成功；Recorder/Replay正常，旧BC同协议回归结果和逐集步数不变。原生任务、控制器、奖励与成功判定没有修改。

这是13个预先指定初态的小规模验证，不是完整LIBERO Benchmark或论文成绩。当前模拟器与官方参考Docker的软件版本不同；首次编译需要明确的XLA兼容开关；动作边界需要显式、经过诊断的接口转换。这些限制在下文记录。

## 1. 来源、版本与环境隔离

- 官方仓库：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)。固定commit：`215abfb217dbac7d5f1273282331b9b1866c0479`。
- 官方模型：`gs://openpi-assets/checkpoints/pi05_libero`，配置`pi05_libero`；实际为 **JAX Orbax OCDBT**，不是PyTorch safetensors。
- 16个对象、12,439,085,481 bytes（约11.58GiB）。逐对象大小、GCS generation、CRC32C及SHA256均已实际校验；含MD5的对象也检查MD5。未发现多余文件。完整对象pin见`openpi-upstream.json`。
- 内容清单指纹：`d9104dfdea46eca2fadf05ec7fc478b19d39b19aa8dfda0e6adedcd6d6b6efac`。这是按路径排序的`{path,size,sha256}`列表规范JSON的SHA256，不是单一权重文件哈希。
- 来源清单指纹：`d648396a09ba9648267eeb933810ae6bd51a1b26d376966e53175e574c625848`；归一化文件SHA256：`b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`。

| 部分 | 实际配置 |
|---|---|
| 主机 | Ubuntu22.04.5，RTX5090 D v2，24455MiB，驱动580.173.02 |
| 原GPU环境 | `.venv-libero-gpu`，Python3.10.20，Torch2.9.1+cu128 |
| 原CPU环境 | `.venv-libero`，Torch2.5.1+cpu |
| 仿真 | hf-libero0.1.4，commit8561c60e…，MuJoCo3.3.2，robosuite1.4.0，EGL |
| 新模型环境 | `.venv-openpi`，Python3.11.15；原始上游`uv.lock`，SHA256 793488b5… |
| 实际推理 | JAX/jaxlib/CUDA plugin0.5.3，flax0.10.2，orbax0.11.13；官方模型BF16/FP32路径，无量化 |
| 独立CUDA依赖 | 上游锁内runtime12.6.77、cuBLAS12.6.4.1、cuDNN9.5.1.17；隔离环境ptxas12.9.41 |
| 客户端 | 固定commit的openpi-client0.1.0，websockets15.0.1、msgpack1.1.2、dm-tree0.1.9、wrapt1.17.3 |

服务端按上游锁还安装Torch2.7.1等依赖，但本次模型实际使用JAX CUDA后端。客户端附加包位于`.deps/openpi-client-runtime`，通过PYTHONPATH启用；没有向原CPU/GPU环境重装依赖。没有安装系统CUDA Toolkit、修改驱动或升级原MuJoCo。

### 实际遇到的编译问题

1. 系统ptxas12.6.85被优先选择，报告不支持CC12.0并回退。仅将独立环境的ptxas12.9.41放到该进程PATH前面；矩阵乘和卷积实际通过。
2. 完整模型第一次推理在Triton GEMM候选编译期间出现`ptxas exit139`，服务器SIGABRT；当时进程采样显存约8716MiB，证据不支持OOM解释。
3. 单独增加`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`后，相同权重和上游依赖成功完成真实推理与所有闭环。没有继续升级JAX或调整物理。

这个开关仅绕开Triton GEMM路径，并不关闭全部Triton；[对应JAX版本固定的XLA编译器源码](https://github.com/openxla/xla/blob/df971129bd82e381954da0185b534220e21798a4/xla/service/gpu/gpu_compiler.cc#L1557)显示常规GEMM路径仍保留。已定位触发路径，但尚不能区分PTX生成器与NVIDIA assembler的具体缺陷责任，也不声称替代内核与无法运行的默认内核逐位等价。

## 2. 实际代码变动

| 文件 | 用途 |
|---|---|
| `src/embodiedlab/policies/openpi.py`（新增） | 官方客户端、输入映射、有限连接/接收等待、chunk队列、动作验证、原始输出与时延日志 |
| `scripts/evaluate_openpi.py`（新增） | 配置任务/初态/seed；核验服务身份；复用Evaluator/Runner/Recorder，保存真实结果、诊断及预览 |
| `tools/serve_openpi.py`（新增） | 校验固定源码/锁/16个权重对象，在loopback调用官方loader/server；同步采样计时 |
| `scripts/download_openpi.py`（新增） | generation固定的官方权重下载/续传和SHA256校验；已有不匹配文件保留并报错 |
| `openpi-upstream.json`、`requirements-openpi-client.lock`（新增） | 来源、内容指纹、协议及可选客户端附加依赖 |
| `tests/test_openpi_policy.py`、`tests/test_openpi_entry.py`（新增） | 输入/动作/缓存/超时/异常/身份与记分测试；fixture测试明确不算模型实验 |
| `src/embodiedlab/evaluation.py`（修改） | 集中原有接触诊断与六帧预览函数，供BC/OpenPI共用 |
| `scripts/evaluate_bc.py`（修改） | 导入上述原函数，去掉重复实现；其评测逻辑不变 |
| `pyproject.toml`（修改） | 只增加可选openpi extra，原CPU/GPU锁不变 |
| `README.md`、`tools/README.md`、`THIRD_PARTY_NOTICES.md`（修改） | 可选入口、范围及第三方许可说明 |
| `docs/openpi/`（新增） | 安装与复现实验说明、真实输入/边界证据和精简机器可读结果 |

Environment、Runner、Recorder、Dataset、Replay、VisualBC网络和训练入口均没有改动。不复制OpenPI模型代码，不引入注册系统或Trainer抽象。

## 3. 输入、动作和Chunk一致性

依据固定源码[官方LIBERO主循环](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py#L113)、[图像工具](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/image_tools.py)、[策略转换](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/libero_policy.py)逐项核查，并使用真实原始观测验证：

| 项目 | 实际结果 |
|---|---|
| 两路相机 | raw256×256 HWC RGB uint8 → 两轴反转一次 → 官方PIL双线性resize_with_pad224×224；两幅图与官方函数逐像素相同，最大差0 |
| 机器人状态 | xyz世界坐标3（m）+轴角3（rad）+两指qpos2（m），float64；与官方构造逐元素相同，最大差0 |
| prompt | 原生任务语言，完全相同 |
| 模型输出 | 官方逆归一化之后的10×7 OSC增量，客户端不再次归一化，不加当前state |
| chunk执行 | 按原序执行前5个，尾部5个丢弃，队列耗尽才请求最新观测；Episode reset清空队列 |
| 动作类型 | float64 `(7,)`，前3平移、后3旋转增量、末维夹爪；不是关节角 |

真实输入证据：`docs/openpi/input_parity.json`。没有把privileged object/contact诊断送入模型。

服务端官方JAX默认key(0)，每次infer推进随机流；官方client.reset是no-op。环境seed仅控制环境，不能声称它重置模型噪声。正式实验重启服务后，按炉灶→抽屉固定顺序连续请求，服务request index为0…278。

另外，最终服务工具重启后，以相同真实初始观测和第一请求重跑炉灶：prompt、两图SHA、state、10×7首chunk，以及完整86步执行动作均与正式实验逐值相同。捕获的官方参考chunk也通过适配器顺序测试；没有拿两次随机状态不同的请求强求相等。

### 动作越界：先诊断，再显式转换

官方quantile逆归一化不保证数值严格落在单位区间。官方参考真实84步中，有45个夹爪值越界，范围约[-1.0073561,1.0112328]；前六维没有越界。原生官方循环不裁剪，直接把它交给robosuite。

已核对本机`robosuite/controllers/base_controller.py:104–123`的输入clip，以及`models/grippers/panda_gripper.py:43–58`的Panda夹爪sign逻辑。两套独立原生环境使用相同官方reset/seed/初态，分别执行这84步原始输出及单位区间投影：**全仿真状态、RGB、reward最大差均0，done与success一致，均84步成功**。没有恢复历史状态来匹配轨迹。

因此适配器默认仍严格拒绝越界；仅显式`--native-action-bounds`时，执行`clip`这一与原生控制器等效的ABI转换。所有raw chunk、raw/issued action、越界维度与幅度都保留。正式13集共1366步，越界仅发生在夹爪维：炉灶237步、抽屉626步，最大幅度0.00989455；没有平移/旋转越界，没有二次缩放、夹爪阈值或额外控制器。NaN/Inf和错误维度始终报错。

## 4. 官方参考与统一评测协议

| 项目 | Official Reference流程（本机） | EmbodiedLab Unified正式实验 |
|---|---|---|
| 客户端/循环 | 实际上游helper、官方client与同样chunk循环 | 最小Adapter + 现有Runner/Evaluator |
| 环境 | 本机固定hf-libero原生实现 | 相同环境 |
| seed/初态 | seed7、炉灶初态0 | 炉灶0–4/seed42–46；抽屉0–7/seed42–49 |
| 稳定初始化 | 官方reset→set_init_state→10步dummy | 原有相同机制，10步settling不计policy步数 |
| 相机/chunk | 256→224，两轴反转，10预测/5执行 | 相同 |
| 终止/上限 | 原生成功，goal300步 | 相同；无历史状态注入 |
| 动作接口 | 原始模型值直接给原生控制器 | 显式等效投影以满足原框架合法动作契约 |

上游参考Docker固定LIBERO子模块f78abd68…、MuJoCo3.2.3、robosuite1.4.1；本机采用已验证hf-libero/MuJoCo3.3.2/robosuite1.4.0。因此本次“官方参考”验证的是官方推理与控制循环在当前原生环境可运行，不是完整旧Docker栈复现。两个goal任务都是300步，无spatial时限混用。额外`openpi_horizon`只提供220步spatial时限选择，本阶段没有跑spatial。

**官方参考实测：84步原生成功、17次推理、墙钟18.319s**。首次未修复编译问题的尝试执行0个动作、success=null，明确保留为工程异常，不算失败Episode。首次集成Smoke为88步成功；最终工具/身份核验入口复验为86步成功。它们均单列，不混入正式13集统计。

## 5. 正式闭环结果与BC对照

每一步均为真实Observation→官方π₀.₅→Action→原生仿真→新Observation。表中时间为policy步骤对应的仿真时间，不含初始化0.5s；成功步即首次成功步。

| Task | 初态 | Seed | 结果 | 步数 | 仿真秒 | 请求数 |
|---|---:|---:|---|---:|---:|---:|
| stove7 | 0 | 42 | success | 86 | 4.30 | 18 |
| stove7 | 1 | 43 | success | 72 | 3.60 | 15 |
| stove7 | 2 | 44 | success | 66 | 3.30 | 14 |
| stove7 | 3 | 45 | success | 89 | 4.45 | 18 |
| stove7 | 4 | 46 | success | 76 | 3.80 | 16 |
| drawer0 | 0 | 42 | success | 122 | 6.10 | 25 |
| drawer0 | 1 | 43 | success | 128 | 6.40 | 26 |
| drawer0 | 2 | 44 | success | 113 | 5.65 | 23 |
| drawer0 | 3 | 45 | success | 119 | 5.95 | 24 |
| drawer0 | 4 | 46 | success | 113 | 5.65 | 23 |
| drawer0 | 5 | 47 | success | 119 | 5.95 | 24 |
| drawer0 | 6 | 48 | success | 124 | 6.20 | 25 |
| drawer0 | 7 | 49 | success | 139 | 6.95 | 28 |

炉灶5/5、抽屉8/8；无超时或不完整集。炉灶共389步/81请求、墙钟43.087s；抽屉977步/198请求、墙钟98.622s。正式实验前指定全部初态，没有挑选成功样本。

原生对象诊断也显示实际交互：炉灶初态0的button从0到约0.502813，抽屉初态0的middle joint从0到约-0.141439m；均伴随机器人/目标接触，原生success=True。接触采样位于控制步端点，不能当成物理子步峰值或真机安全验证。

| 同一环境/初态/seed/300步协议 | 旧VisualBC（本阶段重新实跑） | 官方π₀.₅ |
|---|---:|---:|
| 炉灶goal7 | 3/5 | 5/5 |
| 抽屉goal0 | 4/8 | 8/8 |

BC炉灶逐集225/300/300/113/109步，抽屉300/300/293/242/208/300/214/300步，与历史结果完全一致，两份权重SHA未改变。BC为128px、单步动作；OpenPI为256→224px、5步重规划，模型和预训练数据也不同。此表证明接口可承载两类模型，不足以作严格公平比较、泛化或统计显著性结论。

## 6. 延迟与显存

| 实际计时 | 炉灶 | 抽屉 |
|---|---:|---:|
| RPC首次 / 均值ms | 1312.05 / 395.89 | 376.26 / 391.54 |
| RPC P50 / P95 ms | 383.59 / 402.94 | 387.67 / 419.80 |
| 同步模型采样均值ms | 383.15 | 383.25 |
| 同步采样 P50 / P95 ms | 375.33 / 394.62 | 379.12 / 412.36 |
| act均值ms（含缓存动作） | 82.78 | 79.69 |
| act P50 ms（通常读缓存） | 0.0284 | 0.0292 |
| env.step均值 / P95 ms | 9.93 / 23.59 | 9.76 / 22.90 |
| 该任务运行总墙钟s | 43.09 | 98.62 |

- 真正首次参考请求包括编译：RPC9043.46ms，同步采样8761.88ms。正式评测服务重启，但磁盘编译缓存已由参考实验预热；其首请求1.312s不能称冷启动性能。
- 同步采样用`jax.block_until_ready`围住原始compiled sampler，包含dispatch和设备执行；不是CUDA kernel profiler的纯计算时间。RPC还含服务器前/后处理、同步、序列化和传输；act另含客户端预处理，缓存步不调用模型。
- 正式模型服务PID的NVML离散采样峰值 **8814MiB（约8.61GiB）**，不是整卡其他进程占用，也不保证捕捉瞬时峰值。713个有效样本，退出后1个不可用样本没有当作0使用；资源补充JSON位于两组运行目录。
- 10步预测只执行5步，即0.25s模拟时间约需0.39s请求等待；同步模拟可以等待。不能据此声称满足真机20Hz或实时控制。

## 7. 记录、回放、异常与回归

- HDF5继续采用T动作/T+1观测，raw256 RGB、状态、语言、实际issued action与时间戳完整保留。原始模型chunk和投影记录保存为对应诊断JSON，不把模型预测误当成底层力矩。
- 炉灶第一条86步、抽屉第一条122步用原Replay真实重执行：状态、关节、RGB、reward、timestamp最大差均0；长度、语言、success、terminated/truncated全部一致。没有注入状态。
- 真实断连实验：第一次实际模型请求后关闭本阶段WebSocket，缓存5步执行完毕，在下一请求抛`ConnectionClosedOK`；环境、连接和HDF写句柄均释放。文件complete=False，Dataset明确拒绝将其作为完整Episode加载。success=null，排除正式统计。
- CLI异常测试另覆盖reset失败、第二集RPC失败、前集结果保存、停止后续、stale manifest/伪模型身份拒绝；这些fixture测试明确不充当模型实验。
- 服务工具检查源码HEAD和tracked diff、锁、权重；评测在创建环境前比较manifest与握手的run_id、commit、config、checkpoint、完整指纹、norm stats、后端与chunk长度，拒绝连错服务。
- 正式13集使用先经完整权重校验的诊断启动器；它保存commit/config/norm hash和实际PID，但尚无最终工具新增的run_id/content hash字段。随后最终工具和严格身份校验实跑1整集成功，首chunk及完整86步与正式首集逐值一致；没有事后伪造旧握手元数据。

完整测试实际命令：在README的EGL/LIBERO配置下，分别使用原两个解释器，启用可选target和真实捕获chunk：

```bash
export PYTHONPATH="$PWD/.deps/openpi-client-runtime"
export EMBODIEDLAB_OPENPI_REFERENCE_CHUNK="$PWD/evaluation_results/phase6a_reference_stove_no_triton_seed7/first_real_chunk.npy"
.venv-libero-gpu/bin/python -m pytest -q -p no:cacheprovider
.venv-libero/bin/python -m pytest -q -p no:cacheprovider
```

**GPU：192 passed，27.68s；CPU：188 passed、4 skipped（CUDA专属），26.94s**。新增30项与原有162项GPU测试均通过。warning为上游宏配置/库弃用等已记录提示，没有测试错误。

对清理前保留清单再核查：462个原数据/训练/评测文件重新SHA256，无缺失或改动；原依赖缓存与三套旧环境共检查82389个原条目（大小/软链接），无差异。环境检查不是逐字节哈希声明，核心环境锁无改动。v0.1.0/main仍指向4a67c6490c12b94efae4cbcf7433a044ba4c01e9，开发位于独立phase6a-openpi分支，没有推送远端或覆盖旧成果。

模型服务已结束，并核验其PID不再出现在NVML进程列表；没有停止其他用户进程。

## 8. 实际运行入口与结果位置

完整安装、服务启动和双终端命令见[docs/openpi/README.md](../openpi/README.md)。以下为正式实验参数及最终入口；服务manifest必须使用对应这一次启动生成的文件，不可拿旧文件冒充新服务：

```bash
.venv-libero-gpu/bin/python scripts/download_openpi.py --verify-only
# 服务进程按使用文档设置PATH/XLA/JAX缓存后启动：
.venv-openpi/bin/python tools/serve_openpi.py --port 8006 \
  --manifest evaluation_results/openpi_server/run01.json
# 客户端进程按文档设置PYTHONPATH/EGL/LIBERO_CONFIG_PATH：
.venv-libero-gpu/bin/python scripts/evaluate_openpi.py --suite libero_goal --task-id 7 \
  --init-state-indices 0 1 2 3 4 --seed 42 --port 8006 --native-action-bounds \
  --server-manifest evaluation_results/openpi_server/run01.json \
  --output-dir evaluation_results/openpi_stove
.venv-libero-gpu/bin/python scripts/evaluate_openpi.py --suite libero_goal --task-id 0 \
  --init-state-indices 0 1 2 3 4 5 6 7 --seed 42 --port 8006 --native-action-bounds \
  --server-manifest evaluation_results/openpi_server/run01.json \
  --output-dir evaluation_results/openpi_drawer
```

本阶段正式运行的已存在目录：

- `evaluation_results/phase6a_reference_stove_no_triton_seed7/`：官方参考结果、视频、首真实请求和chunk；先前编译失败的独立目录保留，不算Episode失败。
- `evaluation_results/phase6a_smoke/20261016T074835Z_2feeaeb8/`：首次集成Smoke。
- `evaluation_results/phase6a_stove/20261016T075146Z_8280fdb1/`：正式炉灶5集。
- `evaluation_results/phase6a_drawer/20261016T075229Z_c5df7a9f/`：正式抽屉8集。
- `evaluation_results/phase6a_final_entry/20261016T080532Z_bc2f9df0/`：最终服务/身份检查/入口整集复验。
- `evaluation_results/phase6a_bc_regression_{stove,drawer}/`：旧BC真实回归。
- `evaluation_results/phase6a_disconnect/`：真实断连的明确不完整轨迹。
- `docs/archive/phase6a_results.json`：便携数值汇总；完整本机日志、上游审查行号、权重校验、GPU采样及诊断脚本保留在本次Codex会话的`work/phase6a/`。

例如，已实际执行的Replay命令：

```bash
.venv-libero-gpu/bin/python scripts/replay_libero.py --episode \
  evaluation_results/phase6a_stove/20261016T075146Z_8280fdb1/records/episode_000/episode_000.hdf5
.venv-libero-gpu/bin/python scripts/replay_libero.py --episode \
  evaluation_results/phase6a_drawer/20261016T075229Z_c5df7a9f/records/episode_000/episode_000.hdf5
```

正式运行均生成少量六帧双相机预览；没有失败任务Episode，因此没有挑选性保留失败视频。官方参考有真实视频。本机预览示例：

![炉灶第一条真实闭环](../../evaluation_results/phase6a_stove/20261016T075146Z_8280fdb1/episode_000_preview.png)

![抽屉第一条真实闭环](../../evaluation_results/phase6a_drawer/20261016T075229Z_c5df7a9f/episode_000_preview.png)

## 9. 局限与结论

**工程正确性：通过。** 真正官方模型与GPU后端、官方预处理、真实chunk、原生控制、成功判定、轨迹与Replay串联；异常明确且不伪记成绩，原BC和全部测试无回归。

**算法有效性：在这13个固定初态上确实成功。** 没有测试其他suite、未见任务、更多随机化或完整50初态协议。与BC存在模型和原生输入/动作块差异，不能扩展为总体优越性结论。

当前只验收固定OpenPI commit、JAX GPU后端与这台5090主机；依赖较旧，依靠明确的ptxas/XLA措施运行，不能泛称所有Blackwell环境兼容。上游Docker栈差异、跨请求RNG、编译缓存、NVML采样误差均需随新实验继续记录。WebSocket限本机、单客户端串行执行，不实现并发服务调度或真机部署。

后续优先问题是：在不改变策略的前提下，扩大预先固定的独立初态样本并记录失败类型，判断当前小样本成功是否稳定。**本阶段到此停止：未训练或微调π₀.₅，未加入其他算法、ROS2或新的控制器。**
