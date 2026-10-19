# Phase 7F — LIBERO / ROS 2 / OpenPI 在线闭环

> 历史资料：对应源码、脚本、配置和上游记录完整保留在本机 `archive/ros2-phase7f-20261019` 分支。下列ROS命令需在该分支的独立工作目录执行，当前仿真分支不包含这些模块。

2026-10-19。**完成真实在线闭环：ROS路线2/2成功，直接调用对照2/2成功，步数及全部实际动作一致。** 没有连接FR3、训练或Push。这是两个预先指定初态的工程验证，不是完整LIBERO成绩。

## 最小改动与真实执行路径

- 只增加一个入口：`scripts/run_ros_libero.py`，支持`--mode direct|ros`。复用LiberoEnv、原OpenPIPolicy、Evaluator、Runner、Recorder及预览函数。
- `src/embodiedlab/ros2_vision.py`增加共享`publish_observation()`和`Ros2OpenPIClient`。后者作为原Policy的client，保留原动作缓存、解码和边界审计逻辑。
- 原`run_ros_vision.py`改用共享发布函数，删除重复消息构造；README与`upstream-ros2.json`补充入口。没有新控制器、Registry、模型或数据格式。

循环：运行中环境的当前RGB/状态 → 六条实际ROS消息 → 订阅端重建原始Observation → 原OpenPI预处理 → 官方GPU模型 → ROS动作块回复 → 原Policy消费前5步 → 环境真实执行并产生新观测 → 再推理。

消息为双Image、双CameraInfo、Panda JointState和世界EEF PoseStamped；动作通过`/embodiedlab/libero_online/action_chunk`上的标准UInt8MultiArray承载既有msgpack-numpy格式。两个ROS节点在同一Python进程，实际经过ROS发布/订阅；模型服务在独立进程。模型只读取ROS解码的观测，环境只消费ROS回复的动作；没有本地旁路或离线观测输入。33组模型图像输入均不同，来自实际推进的仿真。

图像方向、256→224预处理、8维状态顺序、轴角分支和世界系OSC语义保持原协议。每次预测10个动作，最多执行前5个，沿用[固定上游主循环](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py)的消费方式；继续使用明确的`native_action_bounds=True`原生ABI转换，没有增加滤波或缩放。CameraInfo保持未提供标定状态；模型不使用这些参数，不将相机标定假设用于动作转换。

## 固定对照协议与结果

任务：`libero_goal / task7 / turn on the stove`；官方初态0、1，seed42、43；原生reset与10步settling、300步时限、20Hz仿真控制、原生成功判定。

两条路线分别重启官方pi05_libero服务，从相同key(0)开始；没有预先消耗模型请求。两边正式请求顺序均为0～32，Episode reset不重置模型RNG。权重及固定源码沿用`openpi-upstream.json`，服务启动重新校验全部SHA256；RTX5090 D v2 / JAX0.5.3，未改权重或依赖。

| 初态 / seed | Direct | ROS在线 | 两边步数 | 仿真完成时间 | 推理请求数 |
|---|---|---|---:|---:|---:|
| 0 / 42 | 成功 | 成功 | 86 / 86 | 4.30s | 18 |
| 1 / 43 | 成功 | 成功 | 72 / 72 | 3.60s | 15 |
| 合计 | **2/2** | **2/2** | **158 / 158** | 7.90s | 33 |

逐项对照结果：

- 全部33个原始`(10,7)`动作块、158步实际执行动作：逐元素完全一致，最大误差0。
- 所有记录帧的双路RGB、robot_state：最大误差0；模型输入图像哈希完全一致。
- ROS内部四元数/轴角往返最大误差4.44e-16；模型float32状态输入一致。
- 每次请求与回复的仿真步号完全相同，源消息时间差为0；没有观测错位、重复动作消费、迟到回复被执行或异常结束。

总共预测330个动作，实际执行158个；每块后5步共165个按原协议丢弃，两个Episode结束时还分别丢弃未消费前缀4步和3步。没有在成功后继续执行缓存动作。

## 延迟与控制时序

以下稳态统计排除两边各自第一个模型请求，样本数32：

| 指标 | 平均 |
|---|---:|
| Direct同步GPU采样 | 397.15ms |
| ROS同步GPU采样 | 396.43ms |
| Direct完整推理往返 | 403.48ms |
| ROS完整推理往返 | 419.89ms |
| ROS观测构造、传输及预处理核对 | 16.58ms |
| ROS动作回复传输 | 0.64ms |
| **ROS路径额外开销：总往返减实际模型RPC** | **17.22ms** |
| 额外开销P95 / 最大 | 17.62 / 18.29ms |
| Direct / ROS单次环境step（含渲染） | 7.90 / 8.06ms |

17.22ms包含Python打包、序列化、同步、额外预处理核对与动作返回，不是纯网络传输延迟。直接调用与ROS稳态往返均值差16.42ms，GPU采样本身也有约0.72ms波动。

首次Direct GPU采样2002.54ms，首次ROS为1006.65ms；首次ROS总往返1710.89ms，其中观测构造/传输/校验段416.02ms。未逐项剖析首次导入、初始化和编译/缓存开销，不能将首次或整个Episode墙钟差异解读为ROS更快。

ROS header记录本次快照的发布时刻，仿真采样时刻由`env_step × 0.05s`标识；它不是硬件相机曝光时间。六路消息在接收时距发布平均15.25ms。正常回复时距发布通常约419ms，首请求约1.31s；期间仿真没有推进，因此通过**源时间戳、请求身份、未改变的仿真步号和5秒请求截止时间**判断是否仍可消费，不刷新原时间戳冒充新状态。所有33次请求均验证了仿真步号不变。接收端原有250ms新消息检查仍保留。

环境仍按每步0.05秒推进虚拟时间，每消费5步后重新观测；推理等待增加墙钟时间。**没有达到或宣称墙钟20Hz控制，也没有连续异步推进仿真。** 这种锁步语义适用于可暂停的仿真，不能照搬到持续运动的真实机器人。

## 异常停止与回归

实际将请求截止设为1ms，首条请求在构造/发布后约25.2ms检测到超时，未调用模型、未执行任何策略动作，仿真步号为0；HDF5保留0个策略动作且`complete=false`。reset阶段原生10步settling仍正常执行，不计为策略动作。超时结果单独存放，不计入2/2分母。

截止时间用于拒绝消费，不保证能中断正在运行的Python或GPU计算。异常会使ROS client不可复用，Runner退出，记录器标记未完成并关闭仿真/ROS/WebSocket。回复绑定运行、Episode、请求序号、源时间戳和仿真步号；代码会拒绝错误、重复或迟到的回复。本次正常实验没有收到此类坏回复，没有额外伪造动作进行测试。

旧Phase7E回放入口实际运行通过，3次真实模型前向，服务请求33～35，位于正式在线实验之后。新ROS轨迹也用原Replay入口实际重执行：86→86步，状态、关节、RGB、奖励和时间戳偏差均0，终止与成功一致。原关键自动化测试 **107 passed，14 warnings，5.52s**，覆盖真实LIBERO录制回放、历史BC CPU/CUDA与VLA Adapter。未新增Mock测试。两个模型服务均已主动停止，ROS节点和环境均已关闭。

## 运行环境与入口

为保持同一LIBERO/EGL环境，未在ROS镜像中重装仿真依赖，也未向GPU虚拟环境安装ROS。只从既有固定镜像导出私有运行时到`.deps/ros2-humble-runtime`：`/opt/ros/humble`及三个缺少的动态库libspdlog.so.1、libfmt.so.8、libtinyxml2.so.9。来源镜像ID：`sha256:2d11bf0913e5bbe09a95a23e7cd8d83f494437be5a5436e912dd5675153832fd`。没有系统安装、修改虚拟环境或导出Franka控制资源。

模型启动沿用[现有OpenPI说明](../openpi/README.md)，两轮分别使用新manifest `server_direct_01.json`、`server_ros_01.json`，必须重启服务以恢复相同随机流。下面是已执行的核心入口；复跑要换未存在的输出目录和服务manifest。

```bash
cd <project>
# Direct环境沿用既有LIBERO/EGL配置与OpenPI客户端PYTHONPATH
.venv-libero-gpu/bin/python scripts/run_ros_libero.py --mode direct \
  --suite libero_goal --task-id 7 --init-state-indices 0 1 --seed 42 \
  --server-manifest evaluation_results/phase7f_online/server_direct_01.json \
  --output-dir evaluation_results/phase7f_online/direct_01
```

ROS模式仅给当前进程添加私有运行时，不要source已导出前缀内带原绝对路径的setup脚本：

```bash
export EMBODIEDLAB_ROS_PREFIX="$PWD/.deps/ros2-humble-runtime/prefix"
export PYTHONPATH="$EMBODIEDLAB_ROS_PREFIX/lib/python3.10/site-packages:$EMBODIEDLAB_ROS_PREFIX/local/lib/python3.10/dist-packages:$PWD/.deps/openpi-client-runtime:$PWD/src"
export LD_LIBRARY_PATH="$PWD/.deps/ros2-humble-runtime/lib:$EMBODIEDLAB_ROS_PREFIX/lib/x86_64-linux-gnu:$EMBODIEDLAB_ROS_PREFIX/lib"
export AMENT_PREFIX_PATH="$EMBODIEDLAB_ROS_PREFIX"
export ROS_VERSION=2 ROS_PYTHON_VERSION=3 ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=79
.venv-libero-gpu/bin/python scripts/run_ros_libero.py --mode ros \
  --suite libero_goal --task-id 7 --init-state-indices 0 1 --seed 42 \
  --server-manifest evaluation_results/phase7f_online/server_ros_01.json \
  --output-dir evaluation_results/phase7f_online/ros_01
```

所有证据在`evaluation_results/phase7f_online/`：`direct_01/`、`ros_01/`含结果、逐请求与逐步诊断、HDF5及预览；`comparison.json`为完整逐项对照；`summary.json`为汇总；`timeout_check/`与`phase7e_regression/`为单独检查，另保留实际运行日志。

轨迹可使用原入口重执行动作（不再次调用模型）：

```bash
.venv-libero-gpu/bin/python scripts/replay_libero.py \
  --episode evaluation_results/phase7f_online/ros_01/records/episode_000.hdf5
```

范围仅为同机锁步LIBERO视觉闭环；跨进程ROS推理节点、持续墙钟仿真、外部相机、FR3控制均未验证或实现。动作信封使用的[UInt8MultiArray](https://github.com/ros2/common_interfaces/blob/humble/std_msgs/msg/UInt8MultiArray.msg)在Humble仍可用，但上游已将其标为弃用的示例消息；当前只用于最小传输验证，没有把它定义为长期硬件动作API。原始数据、权重与历史实验保持不变，本阶段到此停止。
