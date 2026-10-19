# Phase 7 — ROS 2 / FR3 Fake Hardware

> 历史资料：对应源码、脚本、配置和上游记录完整保留在本机 `archive/ros2-phase7f-20261019` 分支。下列ROS命令需在该分支的独立工作目录执行，当前仿真分支不包含这些模块。

已完成ROS接口验收：真实运行官方ROS节点，读取FR3七关节状态，通过原生FollowJointTrajectory执行小幅往返轨迹，验证反馈、结果和运动中取消。未连接真机，未Push。当前分支`phase7-ros2`，恢复标签`pre-phase7-ros2-20261019`。

## 实测环境

| 项目 | 实际配置 |
|---|---|
| 宿主机 | Ubuntu 22.04.5，Linux 6.8.0-138-generic |
| 隔离方式 | 独立Docker镜像，未在宿主机安装ROS，未修改已有Python环境 |
| ROS / Python / NumPy | Humble / 3.10.12 / 1.21.5（ROS容器内） |
| franka_ros2 | 官方Humble 2.7.1，commit `3b50164eabbe2d461baca007bbbbe03c024387bd` |
| franka_description | 2.8.1，commit `02afaae282d4a8e10d7d2f781b23b3515c303ce5` |
| libfranka | 0.20.4，ROS官方Debian包 |
| controller_manager / hardware_interface | 2.54.3 |
| JointTrajectoryController | 2.54.1，position command、position/velocity state |
| 控制 / 反馈 | manager配置1000Hz；controller state 50Hz、action monitor 20Hz |
| 镜像 | `embodiedlab-ros2:humble-fr3-v2.7.1`，ID `8208ae46daa9ce8a4e14bcecde3b9265c56a8164eafca4fba2759f76eb7ec2f9` |

来源、基础镜像digest、源码压缩包SHA256和关键Debian完整版本见`upstream-ros2.json`（ROS归档分支）。实际全部安装包清单保存在`evaluation_results/phase7_ros2/installed-packages.txt`。基础镜像和源码固定；未来软件源若撤下锁定Debian版本，构建将报错，需要另行审核版本，不能默默升级。

软件源没有可用的franka_bringup二进制，因此编译了官方description、msgs、hardware、gripper、bringup五个包。仅限定colcon发现范围，未修改上游代码；未安装或启动MoveIt、Gazebo、移动底盘。gripper包满足原生launch引用，但本次不启动夹爪。

## 核心代码

- `ros2_bridge.py`（ROS归档分支）：运行中URDF核验、按名称重排状态、限位检查、FJT动作客户端、反馈、取消和资源释放。
- `run_fr3_fake.py`（ROS归档分支）：启动官方bringup和标准JTC，验证到位、返回、运动中取消，保存JSON和日志，关闭ROS进程。
- `controllers.yaml`（ROS归档分支）：七关节position模式的原生JTC配置。
- `Dockerfile`（ROS归档分支）：独立构建ROS环境；README与第三方来源声明作了对应更新。

执行链：官方FR3 Xacro → `mock_components/GenericSystem` → ros2_control → joint_state_broadcaster → Bridge；关节目标 → FollowJointTrajectory → 原生JointTrajectoryController → Fake Hardware → 状态和Action反馈。

Bridge读取`/franka/joint_states`硬件广播器，避免中间聚合节点持续重发旧状态；通过运行中`/controller_manager/get_parameters`核验URDF中的硬件插件。非GenericSystem直接拒绝。关节顺序固定为`fr3_joint1`至`fr3_joint7`，位置限位从该URDF读取，状态为rad和rad/s。

API为`read_observation()`、`send_joint_positions(..., command_semantics='joint_position_rad')`、`wait_result()`、`cancel_and_check_hold()`、`close()`。返回观测沿用现有字典/NumPy风格；仅关节位置Policy可以使用这个显式命令接口。既有BC/OpenPI/OpenVLA的七维OSC动作没有接到这里，原Policy、Environment、Runner、Recorder和Replay无需修改。

当前命令边界只用于小幅Fake实验：位移不超过0.05rad，时长至少1秒，零端点速度/加速度，通过原生五次插值计算的峰值速度不超过0.05rad/s。这不是完整真机安全系统。

## 真实运行结果

| 运行 | 往返结果 | 最大终点误差 | 去程反馈数 | 取消结果 | 额外1秒保持采样 / 漂移 | 总耗时 |
|---|---|---:|---:|---|---|---:|
| smoke_02 | SUCCEEDED / FJT 0 | 0rad | 39 | CANCELED（5） | 101条 / 0rad | 7.595s |
| smoke_03 | SUCCEEDED / FJT 0 | 0rad | 39 | CANCELED（5） | 1000条 / 0rad | 7.528s |
| final | SUCCEEDED / FJT 0 | 0rad | 40 | CANCELED（5） | 1001条 / 0rad | 7.607s |

smoke_03和final使用硬件直接话题；smoke_02使用聚合话题，因此采样率不同。初态为官方配置`[0,-π/4,0,-3π/4,0,π/2,π/4]`。去程joint1增加0.02rad、joint7增加0.01rad，2秒到位，再用2秒返回。

取消实验发送8秒、joint1增加0.04rad的目标，在约1秒后、已经观察到约0.00064rad位移时取消。核对取消响应中的Goal UUID与终态CANCELED；不能因为取消结果的FJT error_code恰好为0，就将其当作成功完成。先观察原生保持0.5秒，再额外观察1秒，位置均保持不变。

五类错误命令均在发送前被拒绝：OSC语义、维度错误、NaN、过大位移、URDF越界。最终运行正常关闭controller、hardware、publisher，launch退出码0；同一容器内检查剩余ROS工作进程为`[]`。

首轮smoke_01因Bridge使用独立Context却调用默认全局Executor而失败，已修复为自有SingleThreadedExecutor。失败JSON和日志原样保留。随后修正了对整个进程组重复发送SIGINT的问题，最终节点均正常退出；未修改原生控制器。

原有关键回归在GPU/EGL可用权限下实际执行：**107 passed，0 skipped，5.26s**。范围包括Runner/Evaluator、官方数据读取、真实LIBERO记录与动作Replay、旧E5在CPU/CUDA上的一致性、OpenPI与OpenVLA Adapter。OpenPI测试使用已有真实预测Chunk；本阶段未重新加载两套大VLA模型或重训。未新增Mock测试体系。

结果及完整反馈：`evaluation_results/phase7_ros2/{smoke_01,smoke_02,smoke_03,final}/result.json`，对应目录内有`bringup.log`。最终源码和运行证据另有SHA256清单，均保留本机。

## 运行命令

在项目根目录执行。镜像本机已构建；输出目录必须使用新名称，重复名称会明确报错，保护已有结果。

```bash
mkdir -p evaluation_results/phase7_ros2
docker run --rm --init --network none --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges \
  -e HOME=/tmp -e ROS_LOG_DIR=/results/ros_logs \
  -v "$PWD/src:/embodiedlab/src:ro" \
  -v "$PWD/scripts:/embodiedlab/scripts:ro" \
  -v "$PWD/ros2:/embodiedlab/ros2:ro" \
  -v "$PWD/evaluation_results/phase7_ros2:/results" \
  embodiedlab-ros2:humble-fr3-v2.7.1 bash -lc \
  'source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash &&
   python3 scripts/run_fr3_fake.py --output-dir /results/manual_01'
```

容器无外部网络、无机器人设备映射，DDS限制本机、domain 77。官方launch显式开启Fake Hardware、关闭夹爪；`robot_ip=127.0.0.1`为未使用的占位值。不得改为host网络来连接真机。

从固定源码重新构建（需可访问GitHub和ROS软件源）：

```bash
mkdir -p .deps/ros2
curl -fL --retry 2 -o .deps/ros2/franka_ros2-3b50164.tar.gz \
  https://codeload.github.com/frankarobotics/franka_ros2/tar.gz/3b50164eabbe2d461baca007bbbbe03c024387bd
curl -fL --retry 2 -o .deps/ros2/franka_description-02afaae.tar.gz \
  https://codeload.github.com/frankarobotics/franka_description/tar.gz/02afaae282d4a8e10d7d2f781b23b3515c303ce5
sha256sum .deps/ros2/*.tar.gz
# 与upstream-ros2.json中的两个SHA256逐项核对后解压：
tar -xzf .deps/ros2/franka_ros2-3b50164.tar.gz -C .deps/ros2
tar -xzf .deps/ros2/franka_description-02afaae.tar.gz -C .deps/ros2
docker build --progress plain -t embodiedlab-ros2:humble-fr3-v2.7.1 \
  -f ros2/Dockerfile .deps/ros2
```

## 验收边界与后续工作

Fake Hardware镜像命令到状态，所以零误差符合其设计，不能代表真实FR3精度、动力学、接触或硬件制动能力。[官方Mock Components说明](https://control.ros.org/humble/doc/ros2_control/hardware_interface/doc/mock_components_userdoc.html)明确其用途是离线验证接口管线。容器日志存在无FIFO实时调度权限的提示；本次Fake实验正常，未授予实时权限或改宿主机内核。

连接真实FR3前仍需确认FCI授权、机器人固件与libfranka兼容性、专用网络及实时调度；建立坐标系、标定、IK和轨迹限速/限加速度/碰撞约束；验证硬件急停、通信中断和看门狗停止。VLA的位姿增量和夹爪语义必须经过这些转换，不能直接写入关节轨迹。本阶段不实现上述真机功能，不打开真实硬件通路。
