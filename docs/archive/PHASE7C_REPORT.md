# Phase 7C — Policy Action → ROS 2

> 历史资料：对应源码、脚本、配置和上游记录完整保留在本机 `archive/ros2-phase7f-20261019` 分支。下列ROS命令需在该分支的独立工作目录执行，当前仿真分支不包含这些模块。

已完成Fake Hardware策略动作执行链路。未连接真机、夹爪或在线视觉模型，未Push。分支`phase7c-policy-actions`，恢复标签`pre-phase7c-policy-actions-20261019`。复用原ROS镜像，未安装新依赖或修改原LIBERO评测链路。

## 最小接口与动作约定

新增`ros2_actions.py`（ROS归档分支）和`run_ros_actions.py`（ROS归档分支）。原`Bridge`（ROS归档分支）增加统一短轨迹时长约束；原Fake运行入口抽出共用的启动/清理函数，没有重复一套ROS启动器。

```python
from embodiedlab.ros2_actions import Ros2PolicyExecutor

executor = Ros2PolicyExecutor(bridge, world_to_base_rotation=R_base_from_world)
audit = executor.execute(action)  # canonical LIBERO float array, shape (7,)
# 非阻塞提交/反馈/停止：submit(action), finish(), stop()
```

`R_base_from_world`必须显式给出。`Δp_base=R·(0.05·action[:3])`，`ΔR_base=R·Exp(0.5·action[3:6])·R.T`，`R_target=ΔR_base·R_current`；位置单位m，旋转向量单位rad。全零旋转保留前一已提交姿态目标，与固定版robosuite一致；停止后重置这份目标记忆。输入越界直接拒绝，不clip、不缩小动作、不按新时长重标定动作幅度。

第七维负值请求打开、正值请求关闭、0保留此前夹爪目标；它不是绝对开度。本阶段始终`gripper_command_sent=null`，未启动夹爪节点。控制点为FR3裸法兰`fr3_link8`，Base为`fr3_link0`。Fake主实验明确假设策略world与Base轴对齐；另用90°旋转映射实际验证world X→base Y及旋转左乘。没有声称完成物理标定或LIBERO grip_site→FR3 TCP迁移。

## 时序与边界

原生LIBERO每50ms接收增量目标，内部持续运行OSC反馈；这里每次等FJT结果并用新关节反馈FK核验后才接收下一条。只允许一条在途，忙时拒绝，不积压Chunk。提交动作幅值不变，时间协议明确不同，不能当作原始轨迹复现或VLA任务成功。

固定Fake约束：每动作平移范数≤10mm、旋转范数≤0.05rad；URDF限位、局部IK关节变化≤0.05rad；规划峰值速度≤0.05rad/s、加速度≤0.5rad/s²、jerk≤5rad/s³。零端点速度/加速度的原生五次插值，其峰值分别为`1.875·Δq/T`、`(10/√3)·Δq/T²`、`60·Δq/T³`。按这些约束向上取整到1ms，最短0.2秒；需要超过0.8秒的动作拒绝，不悄悄拉长或缩小。

完成动作的规划段长为0.200～0.556秒，包含准备、ROS反馈的实际逐条耗时0.256～0.605秒，约对应1.6～3.9Hz的阻塞执行节奏，**没有达到或宣称20Hz**。规划峰值实算最大为0.0482rad/s、0.2668rad/s²、4.996rad/s³；这是轨迹计算值，不是实测物理力矩或制动能力。

## 真实结果

正式验证前固定两份Phase6B真实记录：首个挑战任务`libero_10/task2`、init0、seed42，各取前8条已执行动作，不按ROS可达性筛选。限制和来源SHA256先写入`protocol.json`，再开始执行。

| 输入 | 已完成 | 拒绝 | 说明 |
|---|---:|---:|---|
| 小幅七维动作 | 4/4 | 0 | 平移、旋转、混合、零动作；夹爪三种请求均只记录 |
| 旋转坐标映射 | 1/1 | 0 | 明确90°world/Base映射，实际FK核对 |
| OpenPI历史前缀 | 0/8 | 8 | 平移19.3～26.7mm，均超过固定10mm边界 |
| OpenVLA-OFT历史前缀 | 2/8 | 6 | 原样完成前两条6.39mm、9.11mm动作，其余超过10mm |

这反映当前保守执行边界，不比较两个模型优劣。拒绝后继续审计固定前缀的后续动作，也不构成完整原始动作序列复现。

7条完成动作均有4～11条原生反馈，返回`SUCCEEDED/FJT 0`。最大FK位置误差`3.58e-9m`、旋转误差`6.14e-8rad`。全部14条真实记录拒绝动作都未发送FJT，前后关节漂移0rad。

运动中停止返回`CANCELED`，保持0.5秒关节漂移0rad；另实际触发10ms客户端等待超时，随后确认取消和保持。忙时提交被拒绝。越界原始值、过大旋转以及需要1.063秒的9mm目标分别被明确拒绝。取消失败会标记`stop_unconfirmed`并抛出，不能记作约束拒绝或已停止；本次未注入取消服务故障。

每条审计保留原模型输出、原记录canonical动作、请求/Chunk索引；`commanded_arm_action`是原样提交的前六维，实际位移/旋转另由反馈FK记录，部分执行的取消不会标成完成。

OpenPI这8条中6条原始夹爪值曾在旧LIBERO适配链路投影到边界；OpenVLA-OFT raw夹爪经官方normalize/binarize/invert后才成为canonical（此样本+1→-1）。来源加载器只校验这些已有记录与steps.action一致，**没有在ROS侧重新clip或套用错误的raw动作语义**；选中前六维均与原始预测相同。

来源文件（均只读挂载）：

- `evaluation_results/phase6b_formal_openpi/libero_10/task_2/20261016T134735Z_c9978137/episode_000_diagnostics.json`，SHA256 `975c335d2ee8a5b2ed082fa21107b04870c5e5142a67cd1115d4f0bf865e1de9`。
- `evaluation_results/phase6b_formal_openvla/libero_10/task_2/20261016T143152Z_4317c5e6/episode_000_diagnostics.json`，SHA256 `c5748c66cf1f2a328ba225f7ab81d16bd26035099975bd07c4be637d75089966`。

结果在`evaluation_results/phase7c_actions/verified/{result,protocol}.json`，同目录有bringup日志。动作验证11.123秒；原关节模式回归7.629秒、笛卡尔模式回归17.571秒，均通过且ROS工作进程最终为`[]`。既有LIBERO、记录回放、BC CPU/CUDA、OpenPI/OpenVLA关键回归：**107 passed，0 skipped，8.42秒**。未添加Mock测试体系，模型权重和数据未改。

## 复现

项目根目录运行，输出目录使用新名称：

```bash
ROS_PI_RECORD="$PWD/evaluation_results/phase6b_formal_openpi/libero_10/task_2/20261016T134735Z_c9978137/episode_000_diagnostics.json"
ROS_OFT_RECORD="$PWD/evaluation_results/phase6b_formal_openvla/libero_10/task_2/20261016T143152Z_4317c5e6/episode_000_diagnostics.json"
mkdir -p evaluation_results/phase7c_actions
docker run --rm --init --network none --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges -e HOME=/tmp -e ROS_LOG_DIR=/results/ros_logs \
  -v "$PWD/src:/embodiedlab/src:ro" -v "$PWD/scripts:/embodiedlab/scripts:ro" \
  -v "$PWD/ros2:/embodiedlab/ros2:ro" -v "$PWD/evaluation_results/phase7c_actions:/results" \
  -v "$ROS_PI_RECORD:/records/openpi.json:ro" -v "$ROS_OFT_RECORD:/records/openvla.json:ro" \
  embodiedlab-ros2:humble-fr3-cartesian-v1 bash -lc \
  'source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash &&
   python3 scripts/run_ros_actions.py --openpi-record /records/openpi.json \
   --openvla-record /records/openvla.json --count 8 \
   --policy-world-aligned-with-base --output-dir /results/manual_01'
```

## 真机/VLA仍需解决

首先需要验证能维持模型期望节奏的在线反馈与重规划机制，包括延迟、陈旧命令丢弃、Chunk消费和失联停止；不能直接把本次阻塞短FJT当作20Hz OSC。还需相机/world/Base标定、工具TCP与本体状态适配、碰撞/工作空间约束，以及FCI、实时网络、硬件急停和实机制动验证。Fake理想跟随产生的小误差不代表真机精度。本阶段到此停止。

动作语义依据：固定版robosuite的[OSC goal](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/osc.py#L234)、[位姿增量](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/utils/control_utils.py#L114)、[夹爪](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/models/grippers/panda_gripper.py#L43)。源代码位置和当前已安装1.4.0实现已逐项核对。
