# Phase 7B — FR3 Cartesian Control

> 历史资料：对应源码、脚本、配置和上游记录完整保留在本机 `archive/ros2-phase7f-20261019` 分支。下列ROS命令需在该分支的独立工作目录执行，当前仿真分支不包含这些模块。

已在官方FR3 Fake Hardware完成两轮笛卡尔目标执行。没有连接真机、夹爪或VLA，没有Push。分支`phase7b-cartesian`；Phase7恢复标签为`pre-phase7b-cartesian-20261019`。

## 实现

只新增`kinematics.py`（ROS归档分支），扩展`现有Bridge`（ROS归档分支）和`原运行入口`（ROS归档分支），不新增控制框架。Dockerfile显式锁定容器原已安装的Pinocchio 4.0.0、SciPy 1.8.0；旧ROS镜像和全部模型环境、权重、数据保持不变。

- Base：`fr3_link0`；EE：无夹爪法兰`fr3_link8`。官方Link7→Link8的0.107m固定偏移已包含，未假设额外TCP。
- 位姿：`T_base_ee`，平移单位m，旋转为SO(3)矩阵；`pose_matrix()`支持ROS顺序的单位四元数`[x,y,z,w]`。基坐标系与URDF根之间的固定变换显式处理。
- FK直接加载运行中`robot_description`，按关节名称映射索引；额外用官方robot_state_publisher的TF核对。
- IK采用Pinocchio LOCAL SE(3)残差/解析Jacobian及SciPy有界TRF，以当前反馈关节为初值。限制URDF范围和相对初值0.05rad窗口，并优先最小关节变化，避免冗余方向漂移。
- 只有独立FK核验达到`1e-5m`位置误差、`1e-4rad`姿态误差后，才将关节目标交给原Bridge和原生FJT。错误坐标系、非法旋转、未收敛目标不会发送轨迹。

预检发现不带关节变化偏好的欠定优化会产生约0.0496rad的多余位移；加入当前状态偏好后，相同Ry小旋转只需约0.0106rad。解析Jacobian与中心有限差分最大差异为`1.17e-9`。没有调整机器人参数或修改关节状态。

## 实际执行

最终运行7/7目标成功，每段2秒；XYZ及旋转轴均用Base坐标表达，旋转左乘且保持当前末端位置。误差来自实际关节反馈的FK，而不是目标关节的FK。

| 目标 | 位置误差m | 姿态误差rad | 最大关节变化rad |
|---|---:|---:|---:|
| X +2mm | 1.25e-9 | 5.14e-9 | 0.00630 |
| Y +2mm | 6.43e-10 | 1.49e-8 | 0.00368 |
| Z +2mm | 1.24e-9 | 3.28e-8 | 0.00598 |
| Rx +0.01rad | 1.31e-10 | 2.94e-8 | 0.00766 |
| Ry +0.01rad | 1.66e-9 | 5.95e-8 | 0.01060 |
| Rz +0.01rad | 7.37e-10 | 3.40e-8 | 0.00849 |
| 返回初始位姿 | 3.65e-9 | 1.06e-7 | 0.01418 |

每段收到39～40条Action反馈，均为`SUCCEEDED`、FJT error_code 0。FK/TF对照均小于`1e-7m / 1e-7rad`；TF与FK共用URDF，只用于验证坐标和实现一致性，不是外部测量精度。

取消实验：发送X+3mm、8秒目标，约1秒后已移动0.048mm时取消；状态为`CANCELED`（5），停止后仍距目标2.952mm，0.5秒保持观察中的关节漂移为0rad。另对一个8秒目标实际触发10ms客户端等待超时，随后确认取消和零漂移保持；未将超时当作成功到位。非法旋转、错误frame和5m远目标均被拒绝，拒绝前后关节未移动。局部IK失败不等于证明目标在全局不可达。

最终笛卡尔验证17.512秒，原关节模式回归7.665秒，ROS节点正常退出，残留工作进程`[]`。现有LIBERO/数据/记录回放/BC CPU和CUDA/OpenPI/OpenVLA关键回归：**107 passed，0 skipped，8.39秒**；未增加Mock测试体系或重训模型。

原始结果和反馈：`evaluation_results/phase7b_cartesian/cartesian_final/result.json`；旧关节回归在同级`joint_regression/`；各目录有`bringup.log`。首轮结果`cartesian_01/`也保留。

## 使用

项目根目录运行；新镜像`embodiedlab-ros2:humble-fr3-cartesian-v1`已经构建。镜像ID为`2d11bf0913e5bbe09a95a23e7cd8d83f494437be5a5436e912dd5675153832fd`，完整依赖清单保存在上述结果根目录。输出目录不能重名。

```bash
mkdir -p evaluation_results/phase7b_cartesian
docker run --rm --init --network none --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges \
  -e HOME=/tmp -e ROS_LOG_DIR=/results/ros_logs \
  -v "$PWD/src:/embodiedlab/src:ro" -v "$PWD/scripts:/embodiedlab/scripts:ro" \
  -v "$PWD/ros2:/embodiedlab/ros2:ro" \
  -v "$PWD/evaluation_results/phase7b_cartesian:/results" \
  embodiedlab-ros2:humble-fr3-cartesian-v1 bash -lc \
  'source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash &&
   python3 scripts/run_fr3_fake.py --mode cartesian --output-dir /results/manual_01'
```

官方Fake节点已经启动时，Python API为：

```python
from embodiedlab.ros2_bridge import Ros2JointBridge

bridge = Ros2JointBridge()
try:
    target = bridge.read_cartesian_pose()
    target[0, 3] += 0.002
    bridge.send_cartesian_pose(target, frame_id="fr3_link0", duration_s=2.0)
    result = bridge.wait_result(timeout=5.0)
    measured_pose = bridge.read_cartesian_pose()
finally:
    bridge.close()
```

`pose_matrix(position_m, quaternion_xyzw)`可构造绝对目标；不自动猜测坐标系、单位或旋转顺序。取消仍使用`cancel_and_check_hold()`。镜像可按Phase7同样方式从固定源码重建，只需将目标镜像tag改为本阶段tag。

## 连接VLA前还缺什么

当前是“局部目标IK→关节轨迹→反馈FK”，不是笛卡尔连续伺服或直线路径规划。Fake Hardware理想跟随，纳米级模型误差不代表真机精度，也未验证碰撞、接触和物理制动。

VLA接入前首先要明确每个模型的增量/绝对动作、坐标系、平移缩放、旋转组合、动作Chunk和控制周期；完成相机外参与工具TCP标定、FR3本体状态适配。之后仍需限速/限加速度、碰撞约束、失联停止和真机安全监督。现有七维LIBERO OSC动作不能直接当作本接口的七关节或末端位姿输入。本阶段不实施这些扩展。

数学依据：[Pinocchio官方IK示例](https://github.com/stack-of-tasks/pinocchio/blob/devel/examples/inverse-kinematics.py)、[SciPy有界最小二乘](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)；实际调用API和版本已在容器内运行验证，未照搬上游模型或控制器。
