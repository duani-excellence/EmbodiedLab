# Phase 7D — 在线 ROS 2 控制

> 历史资料：对应源码、脚本、配置和上游记录完整保留在本机 `archive/ros2-phase7f-20261019` 分支。下列ROS命令需在该分支的独立工作目录执行，当前仿真分支不包含这些模块。

2026-10-19。**已完成 Fake Hardware 在线闭环**，未接真机、视觉或夹爪，未 Push。

## 实际实现

- `src/embodiedlab/ros2_bridge.py`：直接读取硬件 JointState，订阅深度为1；保留源ROS时间戳、接收/观测单调时间及序号，拒绝旧、重复和过度超前的源时间。等待FJT执行时检查反馈失联；取消结果增加请求与确认时间。
- `src/embodiedlab/ros2_actions.py`：`submit(action, observation_monotonic_s=...)`增加可选时效约束，在IK前和实际FJT发送前检查。过期返回明确拒绝，不刷新时间后重试、不剪裁、不排队。旧调用方式仍可使用。
- `scripts/run_ros_online.py`：一个简单目标跟踪Policy和一个具体TrackingEnv，直接复用原`run_episode()`、FK/IK及Executor。Policy只根据Observation中的实测位姿和当前目标计算增量，不读取Bridge内部状态。原Runner和LIBERO链路没有修改。
- README和`upstream-ros2.json`补充入口与时间约定；未增加依赖、控制器或测试框架。

执行顺序：`reset()`读取当前关节反馈 → FK → `policy.act(obs)`计算小幅误差修正 → `step()`检查动作年龄 → IK → 发送一条短FJT → 等待结果或取消 → 读取更晚源时间戳的反馈 → 下一次决策。reset不修改硬件状态。诊断JSON保留每次策略输入、原始动作、实际发送动作、目标、FJT反馈和结束状态，不将ROS记录冒充LIBERO训练数据。

时间约定：ROS源时间只与同一ROS时钟比较；本地截止时间使用monotonic。源反馈最大年龄100ms，允许源时钟超前20ms；动作从对应观测源时刻起有效150ms，示例命令等待上限1.5s。源年龄映射为本地单调时间用于动作截止检查。这些是本次Fake实验参数，不是实机安全认证。

## 真实运行结果

环境沿用Humble、官方franka_ros2 2.7.1、franka_description 2.8.1、JTC 2.54.1、Pinocchio 4.0.0。镜像`embodiedlab-ros2:humble-fr3-cartesian-v1`，ID `sha256:2d11bf0913e5bbe09a95a23e7cd8d83f494437be5a5436e912dd5675153832fd`；完整上游版本见`upstream-ros2.json`。断网、无设备映射、无特权，以普通用户运行。

正式证据：`evaluation_results/phase7d_online/run_02/result.json`，汇总`summary.json`，轨迹图`tracking.png`。首次`run_01`已保留：在线运动通过，但诊断调用了镜像未安装的`ros2 control`命令而失败；随后改为已有的SwitchController服务，没有安装额外依赖。

四个目标相对起点位置为(6,0,0)、(6,6,0)、(0,6,3)、(0,0,0)mm，同时执行小幅旋转；策略每步最多1.5mm、0.003rad。最大实测空间偏移8.485mm。每次策略反馈源序号和源时间戳均严格增加，129条原生FJT运动反馈已保存。

| 指标 | 实测 |
|---|---:|
| 在线决策 / 到达目标 | 19次 / 4个全部到达 |
| 主跟踪段用时 / 完成动作频率 | 7.553s / 2.515Hz |
| 决策间隔 最小 / 平均 / 最大 | 0.307 / 0.405 / 0.459s |
| 规划FJT段时长 | 0.241～0.400s |
| 策略输入源反馈年龄 平均 / 最大 | 0.457 / 0.843ms |
| 观测源时刻到动作发送 平均 / 最大 | 5.684 / 8.052ms |
| 每段终点FK位置误差 RMS / 最大 | 7.02e-10 / 9.64e-10m |
| 每段终点FK旋转误差 RMS / 最大 | 1.44e-8 / 2.13e-8rad |
| 原生FJT运动过程最大关节跟踪误差 | 2.49e-5rad |
| 四个最终目标的位置误差最大值 | 3.61e-9m |
| 返回起点位置 / 旋转误差 | 2.83e-10m / 7.01e-9rad |

接近阶段的“剩余目标距离”本来就有数毫米，不能当作插补误差。上述极小FK终点误差表示GenericSystem反馈与同一URDF的数值一致性；**不代表真实机械臂具有该精度**。Fake没有验证动力学、接触或物理制动。FJT关节插补也不是笛卡尔直线路径或碰撞规划。

原安全边界保持：每动作平移10mm、旋转0.05rad、关节变化0.05rad；关节速度0.05rad/s、加速度0.5rad/s²、jerk 5rad/s³。本次规划峰值分别为0.02493rad/s、0.19186rad/s²、4.99998rad/s³，最大单步关节变化0.005317rad。没有放宽约束。

## 异常与恢复实测

- 故意延迟决策180ms：150ms动作时效到期，IK前拒绝，关节漂移0。另在Bridge最后发送边界验证过期拒绝，没有发送FJT。
- 已有动作在途：第二条动作明确拒绝，无积压。
- 停用真实joint_state_broadcaster：旧状态没有被当作新反馈，读取超时；重新激活后恢复。
- 运动中客户端等待设为1ms：发生真实超时后发送取消，17.2ms内确认原生`CANCELED`；已运动0.103mm，随后观察0.5s保持漂移0rad。
- 运动中停用真实反馈源：约99.2ms后检测失联，51.9ms内确认原生`CANCELED`，已运动0.179mm。诊断定时器在300ms后恢复广播，然后通过真实反馈检查0.5s保持漂移0rad。恢复源用于验证保持，不用于伪造取消结果。
- 停止后不重置状态：在线Policy根据实际停止位置重新计算，1条动作回到起点。

超时不是停止确认。执行路径在异常后调用`stop()`，必须取得原生取消状态，并用新反馈确认保持；若反馈一直未恢复，保持无法确认，现有实现会报告`stop_unconfirmed`并退出。ACK超时仍保留pending请求，由关闭流程尝试解析并取消，绝不声称“超时所以未执行”。看门狗仅在客户端执行/等待期间运行，不是独立进程级或硬件急停。

## 回归与复现

原关节入口、Cartesian入口、Phase7C历史VLA动作入口均实际运行通过，分别7.576s、17.662s、11.094s，结果在`joint_regression/`、`cartesian_regression/`、`actions_regression/`。三个入口及在线运行均正常关闭，launch退出码0，无残留EmbodiedLab容器。

既有关键自动化测试 **107 passed，8.38s**；包括真实LIBERO无界面录制回放、历史E5 CPU/CUDA推理及两种VLA Adapter。VLA回归使用既有真实推理记录，没有重新加载大模型；本阶段未训练。日志`evaluation_results/phase7d_online/regression_gpu.log`。

实际测试命令：

```bash
PYTHONPATH="$PWD/.deps/openpi-client-runtime" \
LIBERO_CONFIG_PATH="$PWD/.libero-gpu" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
NUMBA_CACHE_DIR=/tmp/phase7d-numba MPLCONFIGDIR=/tmp/phase7d-mpl \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
EMBODIEDLAB_OPENPI_REFERENCE_CHUNK="$PWD/evaluation_results/phase6a_reference_stove_no_triton_seed7/first_real_chunk.npy" \
.venv-libero-gpu/bin/python -m pytest -q -rs -p no:cacheprovider \
  tests/test_evaluation.py tests/test_data.py tests/test_bc_policy.py \
  tests/test_openpi_policy.py tests/test_openvla_policy.py
```

在线入口按以下方式运行；`repro_01`必须不存在，防止覆盖历史结果：

```bash
cd <project>
mkdir -p evaluation_results/phase7d_online
docker run --rm --init --network none --user 1000:1000 \
  --cap-drop ALL --security-opt no-new-privileges \
  -e HOME=/tmp -e ROS_LOG_DIR=/results/ros_logs \
  -v "$PWD/src:/embodiedlab/src:ro" \
  -v "$PWD/scripts:/embodiedlab/scripts:ro" \
  -v "$PWD/ros2:/embodiedlab/ros2:ro" \
  -v "$PWD/evaluation_results/phase7d_online:/results" \
  embodiedlab-ros2:humble-fr3-cartesian-v1 bash -lc \
  'source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && python3 scripts/run_ros_online.py --output-dir /results/repro_01'
```

## FJT是否适合后续VLA

适合目前低速、分步、可监控的在线执行验证。**不能满足LIBERO式20Hz连续增量控制**：本次只有约2.5Hz，且每段端点速度和加速度均为零，存在停顿；controller_manager的1000Hz不等于策略控制频率。慢模型还可能超过150ms动作时效，需要明确设计后续推理与执行时序，不能通过扩大动作或静默更新时间掩盖延迟。

最小替代方向：在Fake中评估原生ros2_control `JointGroupPositionController`，继续复用现有局部IK，以固定周期消费最新有效目标，并补齐有界插补、速度/加速度/jerk约束、独立失联停止机制。该控制器只转发关节目标，不自带这些安全能力，也不保证Franka真机适用；先验证再决定。当前未实现新控制器。官方接口说明见[位置控制器](https://control.ros.org/humble/doc/ros2_controllers/position_controllers/doc/userdoc.html)；现有FJT的监控、取消与插补语义见[官方JTC说明](https://control.ros.org/humble/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)及[轨迹插补](https://control.ros.org/humble/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html)。

后续真机仍需坐标/TCP标定、碰撞与工作空间约束、实际Franka接口验证、物理停止与故障恢复验证。本阶段到此停止。
