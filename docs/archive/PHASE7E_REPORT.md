# Phase 7E — ROS 2 观测与真实 OpenPI 推理

> 历史资料：对应源码、脚本、配置和上游记录完整保留在本机 `archive/ros2-phase7f-20261019` 分支。下列ROS命令需在该分支的独立工作目录执行，当前仿真分支不包含这些模块。

2026-10-19。**完成回放观测 → 标准ROS消息 → 真实π₀.₅ GPU前向 → 原始动作记录。** 没有连接机器人、执行动作、训练或Push。当前是推理链路验证，不是机器人视觉控制或任务成功率实验。

## 代码与观测链路

只增加两个执行文件：

- `src/embodiedlab/ros2_vision.py`：读取Image、CameraInfo、JointState、PoseStamped。每个输入只保留最新消息；要求源时间戳递增、观测年龄≤250ms、六个输入时间偏差≤20ms，检查相机frame、尺寸与Panda关节名称。
- `scripts/run_ros_vision.py`：`prepare`复用EpisodeDataset惰性读取少量帧；`infer`用两个真实ROS节点发布/接收消息，复用`prepare_openpi_observation()`和既有有界WebSocket客户端。每组观测真正请求一次新前向，不使用动作缓存或裁剪。

原Environment、Runner、ROS控制Bridge、VisualBC、OpenPI模型及LIBERO评测入口均未修改。推理不需要Environment.step，因此没有启动Fake Hardware，也没有创建FJT客户端。README及`upstream-ros2.json`补充入口与协议。

本机未发现`/dev/video*`、`/dev/v4l/by-id`可用相机接口。使用已存在的完整真实仿真记录：

`evaluation_results/phase6a_final_entry/20261016T080532Z_bc2f9df0/records/episode_000/episode_000.hdf5`

来源为`libero_goal / task7 / turn on the stove`，seed42、官方初态0；86个动作、87帧观测。原文件13,197,142字节，SHA256为`5bd7658897f801947841a383f6a9b40798698ab44cd28d6c5b8d34dfbee647f5`，实验后再次校验不变。仅抽取0、30、60帧，原始仿真时刻0、1.5、3.0秒；没有读取整套图像进内存或修改HDF5。

| 消息 | 本次内容 |
|---|---|
| Image ×2 | 256×256 RGB uint8；agentview与wrist；相应`libero_*_optical` frame |
| CameraInfo ×2 | 同图像尺寸和frame；原记录没有标定，D/K/R/P保持零、K[0]=0 |
| JointState | 同一记录的Panda七关节rad、两指位置m；保留第二指原始负号 |
| PoseStamped | 同帧原始EEF位置m和xyzw四元数；`libero_world` |

EEF位置不由FR3 FK替代，图像也不与FR3 Fake状态拼接。接收后组成原有8维状态：世界xyz3 + 轴角3 + 两指qpos2；关节角另外保留作检查。语言来自同一条记录。

实际topic均在`/embodiedlab/libero_replay/`下，后缀为`agentview`、`agentview_info`、`wrist`、`wrist_info`、`joint_states`、`eef_pose`，不发布到FR3控制链路的状态或命令topic。

原始robosuite数组行原点在底部，发布前只反转行使ROS图像符合top-left约定；接收后恢复原始数组，再调用原有OpenPI“180°旋转 + 224px resize”预处理。两路模型输入哈希与直接从HDF5预处理得到的哈希完全一致。ROS规范依据：[Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg)、[CameraInfo](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/CameraInfo.msg)。未标定CameraInfo不提供可用内参，也不宣称存在相机到世界/机器人基座的标定TF。

必须保留四元数符号：原始状态存在大于π的轴角，若转换成通常的最短旋转向量会改变模型输入。采用固定OpenPI官方示例的四元数→轴角公式，三帧往返最大状态误差 **4.44e-16**，两路原始RGB和七关节位置逐元素不变。

## 实际模型输出和延迟

服务沿用原始`pi05_libero` JAX Checkpoint，固定源码`215abfb217dbac7d5f1273282331b9b1866c0479`，内容指纹`d9104dfdea46eca2fadf05ec7fc478b19d39b19aa8dfda0e6adedcd6d6b6efac`。启动入口重新核验全部权重SHA256；RTX5090 D v2、JAX0.5.3、既定XLA开关，未改模型。服务manifest在`evaluation_results/phase7e_vision/server_01.json`，模型加载4.385s；一次推理后观察到该服务进程显存8786MiB，不将此称为峰值。

完整成功证据：`evaluation_results/phase7e_vision/ros_run_02/result.json`。共30条实际ROS消息：2组无效输入及3组有效输入，每组六条；所有有效输入时间偏差为0。过期500ms及相机/状态错位100ms的输入均未进入模型。

| 原记录帧 | 服务请求序号 | 输入重发布年龄 | 同步GPU采样 | RPC | 重发布到回复 |
|---|---:|---:|---:|---:|---:|
| 0 | 3 | 36.01ms | 382.32ms | 399.56ms | 441.91ms |
| 30 | 4 | 34.63ms | 381.42ms | 387.73ms | 428.12ms |
| 60 | 5 | 20.63ms | 379.54ms | 388.86ms | 415.35ms |
| 平均 | — | 30.43ms | **381.09ms** | **392.05ms** | **428.46ms** |

每次返回有限的原始 **(10,7)** 动作块。第一组的首动作：

```text
[0.014664, 0.049233, 0.010798,
 0.003803, 0.027962, -0.007650, -0.998027]
```

这是官方逆归一化后的LIBERO OSC增量：前三维按0.05m缩放，旋转三维按0.5rad缩放并左乘世界姿态，末维为夹爪语义；不是FR3七关节目标。第二组首动作夹爪值为-1.001645，按原始值保存，没有为满足执行范围而裁剪。所有三组完整输出都在result.json。**发送到硬件的动作数为0。**

首次运行也完成了三次GPU请求（序号0～2），但最终JSON保存遇到NumPy bool序列化错误；修复为Python bool并在创建文件前完成序列化，重新完整运行通过。首轮日志及部分失败文件保留，不能作为完整结果读取。首轮第一次RPC为1392ms、端到端1435ms，含服务首次请求开销；上表是随后三次稳态测量。官方key(0)随机流随请求推进，没有重置或筛选输出，因此完整报告请求序号为3～5。

本次是**锁步回放**：每帧用当前ROS时钟重新发布，同组消息共用时间戳；原始仿真时间单独保留。端到端指标是本机消息重发布到模型回复，包含构造/传输、同步、预处理和RPC，**不包含真实相机曝光、采集或驱动延迟**。不冒充20Hz实时采集或视觉闭环。

## 运行入口与回归

实际数据准备命令：

```bash
cd <project>
PYTHONPATH="$PWD/.deps/openpi-client-runtime" \
.venv-libero-gpu/bin/python scripts/run_ros_vision.py prepare \
  --recording evaluation_results/phase6a_final_entry/20261016T080532Z_bc2f9df0/records/episode_000/episode_000.hdf5 \
  --frames 0 30 60 --output-dir evaluation_results/phase7e_vision/replay_01
```

模型启动参数沿用[已验证OpenPI说明](../openpi/README.md)，本次加`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，服务manifest路径改为`evaluation_results/phase7e_vision/server_01.json`。只监听127.0.0.1:8006；每次重启要用新manifest名。

实际ROS容器运行命令如下。复跑时需使用未存在的输出目录，并启动匹配manifest的模型服务：

```bash
docker run --rm --init --network host --user 1000:1000 \
  --cap-drop ALL --security-opt no-new-privileges \
  -e HOME=/tmp -e ROS_LOG_DIR=/results/ros_logs \
  -e ROS_LOCALHOST_ONLY=1 -e ROS_DOMAIN_ID=78 \
  -e PYTHONPATH=/embodiedlab/src:/client:/vision-deps \
  -v "$PWD/src:/embodiedlab/src:ro" \
  -v "$PWD/scripts:/embodiedlab/scripts:ro" \
  -v "$PWD/openpi-upstream.json:/embodiedlab/openpi-upstream.json:ro" \
  -v "$PWD/.deps/openpi-client-runtime:/client:ro" \
  -v "$PWD/.venv-libero-gpu/lib/python3.10/site-packages/PIL:/vision-deps/PIL:ro" \
  -v "$PWD/.venv-libero-gpu/lib/python3.10/site-packages/pillow.libs:/vision-deps/pillow.libs:ro" \
  -v "$PWD/.venv-libero-gpu/lib/python3.10/site-packages/typing_extensions.py:/vision-deps/typing_extensions.py:ro" \
  -v "$PWD/evaluation_results/phase7e_vision/replay_01:/replay:ro" \
  -v "$PWD/evaluation_results/phase7e_vision/server_01.json:/server.json:ro" \
  -v "$PWD/evaluation_results/phase7e_vision:/results" \
  embodiedlab-ros2:humble-fr3-cartesian-v1 bash -lc \
  'source /opt/ros/humble/setup.bash && python3 scripts/run_ros_vision.py infer --bundle /replay --server-manifest /server.json --port 8006 --output-dir /results/ros_repro_01'
```

ROS仍使用既有镜像`sha256:2d11bf0913e5bbe09a95a23e7cd8d83f494437be5a5436e912dd5675153832fd`；模型在独立宿主Python环境。容器只读挂载已存在的轻量客户端/Pillow，不安装或修改任何环境，不映射设备。host网络用于访问本机loopback模型服务；ROS限制localhost和独立domain78，没有启动机器人控制进程。

既有关键自动化回归 **107 passed，14 warnings，8.38s**：真实LIBERO无界面录制回放、历史BC CPU/CUDA及OpenPI/OpenVLA Adapter；完整命令与Phase7D相同，缓存路径改为phase7e，日志`evaluation_results/phase7e_vision/regression_gpu.log`。本次未增加Mock测试。模型服务已主动停止、ROS容器已退出、8006端口释放。

## 后续部署边界

1. 真实相机需要内参/畸变K、D、分辨率及RGB方向验证；固定相机需要相机到base标定，腕相机需要hand-eye标定，且要获取与图像时刻匹配的机器人姿态。
2. 需要明确FR3 base、法兰、真实工具TCP的变换及工具偏置，统一世界平移、旋转左乘和动作单位。不能把当前LIBERO世界或Panda EEF当作FR3 TCP。
3. 需要真实采集时钟同步和延迟预算。当前约428ms明显超过Phase7D的150ms动作有效期，不能直接把回复交给现有执行器，也不能通过静默更新时间或放宽动作边界绕过检查。
4. 当前Reader要求每帧都有更新的CameraInfo；真实驱动可能只发布静态标定，需要按实际驱动做最小调整。当前模型训练域、机器人本体状态及夹爪语义也尚未验证适合真实FR3；标定完成不等于模型已经可直接部署。

本阶段到此停止，只完成推理与记录。
