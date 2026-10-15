# Phase 5B.2：LIBERO 专家数据与仿真环境一致性诊断

> 历史实验记录；本机路径已脱敏，诊断工具路径已更新。原始结果保留在本机，数据、权重和配图未随仓库发布。当前安装与入口以[项目 README](../../README.md)为准。

日期：2026-10-15（Asia/Ulaanbaatar）  
项目：`<project>`
诊断前代码：`90ec906fc13749e96ac9fcb6969cae8363984216`

## 1. 结论与适用范围

**结论：CONDITIONAL GO，可以进入单任务视觉 BC 的受控实验；本阶段没有训练模型。**

当前证据支持数据与环境的动作接口兼容。没有发现需要先更换数据源、重新采集示范或迁移物理版本的明确错误。历史轨迹无法精确重现，主要涉及两类已实验证实的问题：

1. 恢复 `time/qpos/qvel` 后，OSC 的缓存和控制目标没有随之恢复。仅刷新缓存，9个代表性单步点的末端误差中位数由 **5.363 mm 降至 0.545 mm**。
2. 固定夹具的位置属于模型，而不属于 `qpos`。仅恢复 demo_0 XML 中记录的三个固定夹具初始位置，保持当前质量、惯性、控制器、求解器和碰撞参数不变，多步最大末端偏差由 **15.156 mm 降至 1.303 mm**。

三个炉灶专家 Episode（80、96、89个动作）在当前环境的官方初始化方案中均触发原生成功判定；三个历史起点方案也均成功。成功不代表轨迹完全相同，更不能据此宣称整个 LIBERO 的兼容性或学习后的成功率。

范围：仅 `libero_goal / task_id=7 / turn_on_the_stove`；单步选择 demo_0、demo_1、demo_2，进行28次动作执行；多步共14个有明确标签的运行、1220个动作，只有3条不同专家示范。所有历史状态注入、XML恢复、夹具恢复和控制器检查均为隔离诊断，**没有改变正式环境、控制器、奖励、成功判定或评测协议**。

## 2. 固定配置与证据文件

| 项目 | 实际配置 |
|---|---|
| Python / NumPy / h5py | 3.10.20 / 1.26.4 / 3.11.0 |
| hf-libero / robosuite / MuJoCo | 0.1.4 / 1.4.0 / 3.3.2 |
| LIBERO 源码提交 | `8561c60eea2fb93096146f240194649df73d8b1e` |
| 资产来源 | `lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |
| 专家来源 | `yifengzhu-hf/LIBERO-datasets@147fa5eb607bb696dd922c3518084a90954cd2f3` |
| HDF5 | `data/libero/libero_goal/turn_on_the_stove_demo.hdf5`，447509922 bytes，50条示范 |
| SHA256，诊断前后相同 | `387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5` |
| 种子 / 官方初态索引 | seed=0；demo_0/1/2 分别选官方索引0/1/2，不声称这些索引对应原始示范 |
| 时间 | 物理步长0.002 s；控制周期0.05 s；原生诊断 horizon=300 |
| 渲染 | EGL headless；多步图像32×32，模型审计128×128；没有改相机外参 |

证据保存在 `<local-artifacts>/`：

- `phase5b2_audit.json`：实际模型参数、源码位置、元数据及恢复前后快照。
- `phase5b2_alignment.json`：重新扫描全部50条示范的时间对齐、动作范围和实际训练样本。
- `phase5b2_single.json`：11个时间点、28次单步实验及控制器/物理状态快照。
- `phase5b2_frame_correction.json`：hand body 与 grip_site 坐标系核验。
- `phase5b2_replays.json`：14次多步运行的每一步误差、对象状态、原生成功和接触数据。
- `phase5b2_summary.json`：均值、中位数、P95、最大值、接触前后统计。
- `phase5b2_error_curves.png`、`phase5b2_run_log.txt`：曲线和实际终端结果。
- `phase5b2_scripts/`：本次诊断脚本快照，便于复核；中间 JSON 已合并，原件保留在 `work/phase5b2/intermediate/`。

## 3. 实验 A：动作、时间与状态语义

### 3.1 动作定义一致

原 HDF5 `env_args` 与当前控制器均为 Panda 的 `OSC_POSE` 增量控制：

| 维度 | 语义 |
|---|---|
| action[0:3] | 世界坐标系下末端位置增量；归一化输入[-1,1]，每轴缩放到±0.05 m |
| action[3:6] | 旋转向量增量；每轴缩放到±0.5 rad，`R_goal = R_delta @ R_current` |
| action[6] | 夹爪，-1打开、+1闭合；0保持内部命令累加器 |

它们不是七个关节角。控制器 `kp=150`、`damping_ratio=1`、固定阻抗、无插值、无位置/姿态限幅，均与数据元数据一致。当前模型9个执行器不等于7维上层动作：机械臂经 OSC 转换成力矩，夹爪走原生控制逻辑。未额外缩放、交换维度或改变符号。

源采集配置 `ignore_done=True/reward_shaping=True`，当前正式配置为 False/False；它们有退出/奖励语义差异，不等于动作变换差异。本诊断逐步调用当前原生 `check_success()`，没有把 HDF5 的奖励当成当前环境输出。官方发布文件的 `rewards/dones` 由整理脚本置零并把最后一项置1，**不是逐步原生成功事件记录**。

### 3.2 两种“状态”的时间位置不同

对本固定文件重新扫描50条、4460个动作：

- `states[t]` 为 `action[t]` 执行前的物理状态。
- `obs[t]` 为该动作执行后的观测。
- 全50条中，`obs/joint_states[:-1]` 与 `states[1:]` 的对应关节位置完全相等；夹爪同样完全相等，最大数值差均为0。
- 相邻状态时间差0.05 s，最大浮点偏差约 `5.51e-15 s`。
- 因而监督配对仍为 **`obs[t-1] → action[t]`，t=1…T-1**。共有4410对；舍弃 action[0] 是因为没有对应的动作前 RGB 观测。最后一个动作可以保留，最后一个完整 post-action 物理状态则没有发布。
- 另抽查4个实际 PyTorch 样本，跨 Episode 边界及首尾索引：动作、8维状态、两路 CHW RGB 与源数组经既定 float32 转换后逐元素一致。图像保持原 OpenGL 方向，uint8 RGB→CHW float32/[0,1]；状态顺序仍为 eef xyz、hand body 旋转向量、两指 qpos。

源码解释：采集 wrapper 先存初态再存每步结果；合并时删去最后一个状态，因此原始 `states` 对齐 pre-action。发布整理脚本执行动作后保存观测，并跳过最初5个动作的记录，但这5步确实执行了。保留的 `states[0]` 时间为0.25 s；遗漏的动作与控制器历史无法从发布数据完整重建。此结论只针对已固定和实测的炉灶文件。

### 3.3 79维物理状态不是完整检查点

当前模型 `nq=41, nv=37, na=0`，存储向量为：

```text
states[t] = [time(1), qpos(41), qvel(37)]
```

包含关节/自由物体位置和速度；本模型没有非空 actuator activation。未保存 `ctrl`、求解器 warmstart、外力历史、OSC 缓存和 `new_update`、`goal_pos/goal_ori`、nullspace `initial_joint`、夹爪命令累加器、环境计数器/RNG、固定夹具 `body_pos` 及模型参数。

原生 `set_init_state()` 只写 time/qpos/qvel，再 `sim.forward()` 和更新观测，没有刷新或恢复控制器。`forward()` 会重算派生量，不能说所有未保存的物理数组都保持不变。审计实测旧控制目标和夹爪命令被保留，warmstart 被重算。**以下全部历史恢复都标记为“部分状态恢复诊断”，不能证明相同完整状态下动力学不同。**

### 3.4 可定位的源码证据

路径前缀：`P=<project>`，`R=P/.venv-libero/lib/python3.10/site-packages/robosuite`，`L=P/.deps/LIBERO-8561c60eea2fb93096146f240194649df73d8b1e`。完整原始路径仅存于本机审计 JSON，未随仓库发布。

| 源文件与行号 | 证据 |
|---|---|
| `R/utils/binding_utils.py:213–241,1140–1169` | flatten/restore 仅 time/qpos/qvel，且要求 na=0 |
| `L/libero/libero/envs/env_wrapper.py:127–145` | 状态恢复→forward→观测，不恢复控制器 |
| `R/controllers/base_controller.py:125–161,175–188` | `new_update/force` 缓存条件，初始关节参考 |
| `R/controllers/osc.py:234–266,347–383` | 非零旋转更新目标；零旋转保留旧目标；nullspace及reset_goal |
| `R/utils/control_utils.py:150–177` | 增量旋转左乘 |
| `R/models/grippers/panda_gripper.py:43–66`；`R/robots/single_arm.py:243–259` | sign(action)驱动夹爪累加器，每个物理控制子步更新 |
| `R/wrappers/data_collection_wrapper.py:69–107,156–170`；`L/scripts/collect_demonstration.py:152–179` | 初态+post-state采集，合并删除最后状态 |
| `L/scripts/create_dataset.py:123,150–154,175–227,260–267` | 执行后观测、裁剪前5步、整理reward/done |
| `P/src/embodiedlab/envs/libero_env.py:112–133` | 官方初态后10步 settling，产生新的控制历史 |
| `R/robots/single_arm.py:123,309` | 控制使用 grip_site，观测四元数来自 hand body |

## 4. 实验 B：单步一致性与控制器因果检查

每条选 t=0、T//2、末段炉灶关节变化最大的一个动作，覆盖早期、中期及接触操作；另选两个零旋转动作。所有比较都是原动作执行后的观测对存储 `obs[t]`。关节指标为7关节最大绝对差；末端位置为欧氏距离；姿态为同一 hand body 的 SO(3) 测地角。

`raw`：只恢复物理状态；`sync`：在相同恢复后仅调用一次 `controller.update(force=True)`，不额外 step、不修改目标或物理参数。

| Episode / t（0起始） | 阶段 | raw→sync 关节最大差(rad) | raw→sync 末端差(mm) |
|---|---|---:|---:|
| demo_0 / 0 | 早期 | 0.004526→0.001281 | 2.651→0.246 |
| demo_0 / 40 | 中期 | 0.016524→0.004715 | 4.071→0.749 |
| demo_0 / 64 | 炉灶运动 | 0.085665→0.075298 | 5.363→6.604 |
| demo_1 / 0 | 早期 | 0.004863→0.000557 | 1.802→0.062 |
| demo_1 / 48 | 中期 | 0.025537→0.002440 | 13.923→0.545 |
| demo_1 / 83 | 炉灶运动 | 0.021748→0.014029 | 12.595→1.527 |
| demo_2 / 0 | 早期 | 0.002673→0.000413 | 0.993→0.054 |
| demo_2 / 44 | 中期 | 0.033659→0.000507 | 13.767→0.270 |
| demo_2 / 83 | 炉灶运动 | 0.031564→0.004938 | 5.740→2.481 |

9个主采样点的分布（P95为这9点的描述统计，不是置信区间）：

| 指标 | raw 中位数 / P95 / 最大 | sync 中位数 / P95 / 最大 |
|---|---:|---:|
| 关节最大差(rad) | 0.021748 / 0.064863 / 0.085665 | 0.002440 / 0.050790 / 0.075298 |
| 末端位置差(mm) | 5.363 / 13.861 / 13.923 | 0.545 / 4.955 / 6.604 |
| 末端姿态差(rad) | 0.016481 / 0.050536 / 0.056874 | 0.000976 / 0.024580 / 0.032761 |
| 夹爪位置差(mm) | 5.148 / 21.141 / 24.740 | 5.124 / 19.872 / 22.472 |

刷新缓存使9/9关节指标、8/9末端位置指标改善，说明缓存是实际因素，不能解释全部接触误差。接触点同步后炉灶角误差分别为0.018281、0.002278、0.000297 rad；原生成功分别为False、False、True，与参考下一状态炉灶角0.379782、0.480419、0.733232 rad相容（原生门槛0.5 rad）。夹爪累加器没有恢复，夹爪误差不应归因于动作符号错误。

### 零旋转目标与物理历史隔离

缓存刷新后，零旋转动作仍保留旧 `goal_ori`。仅改变目标恢复方式，得到：

| Episode/t | sync 关节差 | sync后reset_goal | sync后重建上次非零旋转目标 | sync后清空物理数据重置（独立方案） |
|---|---:|---:|---:|---:|
| demo_1/51 | 0.016529 | 0.001426 | 0.001639 | 0.015434 |
| demo_2/16 | 0.004748 | 0.001465 | 0.001308 | 0.004667 |

单位均rad。重建目标仅利用此前保留的非零旋转动作和对应历史状态做 scratch FK，假设同一OSC定义；没有恢复 nullspace、夹爪或完整求解器历史。`reset_goal` 只是一种对照，不等于恢复真正历史目标。清空物理数据仅在两个非接触点做过，改善有限，不能全局排除 warmstart 的影响。

**坐标系纠错：** 初次审计将 OSC 的 grip_site 姿态与观测的 hand body 姿态直接比较，产生约π/2的假差。已用3个现场状态核验固定变换，并重算56条缓存审计记录；同步后同坐标系缓存角差为0。原28次物理动作和 raw-vs-stored 误差未改。独立证据及前后非角度字段校验和见 `phase5b2_frame_correction.json`。这里没有用π/2假差作为控制器不一致证据。

## 5. 实验 C：多步专家动作回放

四个方案每条均从起点连续执行完整动作序列，途中没有覆盖状态：

- **official**：当前正式 reset、官方初态索引0/1/2、10个原有 settling 动作，再执行保留的专家动作。该初态不等于原示范起点；遗漏的前5个动作未知，settling后的时间0.5 s也不同于裁剪边界0.25 s。
- **history_raw**：当前原生模型，恢复历史 `states[0]`，不刷新缓存。
- **history_sync**：在 history_raw 上仅刷新缓存一次。
- **stored_xml_sync**：历史 XML 仅重定位资产文件路径，然后恢复 `states[0]` 并刷新缓存。这恢复了更多场景信息，也改变了模型，属于诊断，不是正式评测方案。

为获得完整误差曲线，诊断在首次成功后继续执行剩余动作；所有序列均小于原生300步上限。正式 Evaluator 仍按原协议结束，没有修改。

### 5.1 误差与原生成功

表内为逐动作误差 **中位数 / P95 / 最大值**。T条动作全部执行；原生最终成功12/12，按每种方案计为3/3，但不是12条独立示范。

| Episode / 方案 | T | 关节最大差(rad) | 末端位置差(mm) | 首次原生成功动作号 |
|---|---:|---:|---:|---:|
| demo_0 / official | 80 | 0.147679 / 0.243071 / 0.263265 | 28.530 / 33.668 / 38.758 | 75 |
| demo_0 / history_raw | 80 | 0.022428 / 0.100251 / 0.105163 | 4.997 / 13.323 / 15.868 | 80 |
| demo_0 / history_sync | 80 | 0.005193 / 0.083024 / 0.087544 | 1.177 / 12.430 / 15.156 | 79 |
| demo_0 / stored_xml_sync | 80 | 0.001079 / 0.002497 / 0.002674 | 0.109 / 0.631 / 0.651 | 69 |
| demo_1 / official | 96 | 0.061294 / 0.151658 / 0.160360 | 15.714 / 24.817 / 26.843 | 87 |
| demo_1 / history_raw | 96 | 0.019021 / 0.062415 / 0.076375 | 3.975 / 14.463 / 16.348 | 87 |
| demo_1 / history_sync | 96 | 0.002974 / 0.068352 / 0.079439 | 0.851 / 13.180 / 14.951 | 88 |
| demo_1 / stored_xml_sync | 96 | 0.000814 / 0.001444 / 0.001537 | 0.106 / 0.440 / 0.484 | 85 |
| demo_2 / official | 89 | 0.034186 / 0.212674 / 0.240657 | 9.059 / 27.459 / 27.495 | 72 |
| demo_2 / history_raw | 89 | 0.008670 / 0.022354 / 0.025388 | 2.062 / 5.550 / 5.808 | 78 |
| demo_2 / history_sync | 89 | 0.002786 / 0.009542 / 0.011889 | 1.118 / 5.825 / 5.965 | 78 |
| demo_2 / stored_xml_sync | 89 | 0.000905 / 0.001379 / 0.001431 | 0.088 / 0.537 / 0.566 | 78 |

official 起点7关节最大差已达0.085021、0.038668、0.028124 rad，因此其开环曲线不能用于断言动力学不兼容。它的价值在于确认专家动作在当前官方场景仍有完成原任务的能力。

### 5.2 偏差的发生时间和接触关系

“显著偏差”仅为诊断标记：关节最大差>0.05 rad或末端差>10 mm，不改变任务成功门槛。下表动作号从1起，时间为动作号×0.05 s，不含初始化预热。

| Episode | history_sync 首次端点接触 | history_raw 首次显著偏差 | history_sync 首次显著偏差 | official 首次显著偏差 |
|---|---:|---:|---:|---:|
| demo_0 | #41 / 2.05 s | #67 / 3.35 s | #69 / 3.45 s | #1 / 0.05 s |
| demo_1 | #51 / 2.55 s | #82 / 4.10 s | #85 / 4.25 s | #1 / 0.05 s |
| demo_2 | #47 / 2.35 s | 无 | 无 | #57 / 2.85 s |

history_sync 中接触前→首次接触后，关节误差中位数分别为0.002121→0.018907、0.001584→0.018975、0.001440→0.008360 rad；末端误差为0.982→3.442、0.780→5.823、0.831→4.755 mm。demo_0/1 的较大漂移出现在接触之后；demo_2没有越过诊断阈值，说明误差不一定单调累积或必然发散。

接触只在每个控制周期末采样，不能精确定位周期内部第一次碰撞，也不能作为真实力传感器读数。JSON保留几何体对及 `mj_contactForce` 输出；history_sync 的端点法向力总和最大分别为436.35、415.14、417.23 N，属于原生刚性接触求解结果，未据此调参。没有发现动作方向或夹爪定义被取反的证据，但未保存夹爪控制历史仍妨碍精确重放。

本机保留图片（未随仓库发布）：专家动作多步误差曲线，`<local-artifacts>/phase5b2_error_curves.png`。

曲线使用对数纵轴；圆点为首次原生成功，竖线为 history_sync 首次端点炉灶接触。

## 6. 实验 D：最小因果隔离与模型检查

### 6.1 固定场景是重要因素

历史 XML 的固定夹具位置没有包含在 `states` 中。demo_0 相比当前场景，炉灶、酒架、柜体位置分别相差约21.49、17.14、16.10 mm（欧氏距离）。在同一原生模型上，**只在回放起点写入 XML 中原样记录的这三个固定 body_pos**，然后按 history_sync 连续运行；没有后续状态注入，没有拟合或改变动力学参数。

| demo_0 对照 | 最大关节差(rad) | 最大末端差(mm) | 原生最终成功 |
|---|---:|---:|---|
| history_sync | 0.087544 | 15.156 | True |
| fixtures_sync：仅恢复固定场景初始位置 | 0.005356 | 1.303 | True |
| current_xml_sync：只将当前模型导出并重载 | 0.086976 | 15.122 | True |
| stored_xml_sync：恢复历史XML，多个模型差异一起变化 | 0.002674 | 0.651 | True |

fixtures_sync 前后质量/惯性数组逐元素一致；动作、控制器、物理解算和其他模型参数保持不变。接触后的末端最大差仅0.175 mm，整条最大1.303 mm来自接触前。它证明**当前场景与记录场景的固定初始位置不一致，足以解释 demo_0 的主要接触后偏差**；不能由此推出所有任务都有同一根因。

### 6.2 XML重编译不是无损模型恢复

审计比较的是正在运行的 `MjModel`，另外单独编译导出的XML作对照：

- 当前原生源XML的 compiler 是 `inertiagrouprange="0 0"`。
- `sim.model.get_xml()` 经 `mj_saveLastXML` 导出后没有保留该属性；当前默认值是`0 5`，会改变自动质量/惯性推断使用的几何体组。[MuJoCo 3.3.2 文档](https://mujoco.readthedocs.io/en/3.3.2/XMLreference.html#compiler-inertiagrouprange)
- 实测重新编译当前导出XML后，碗质量0.005591→0.045900 kg、酒架0.152748→0.616398 kg，并伴随惯性变化。直接重编译原生源XML没有这些质量差异。
- 历史XML同样没有该属性，在当前运行时重编译的质量与当前导出重编译结果一致。因此 **stored_xml_sync 并不是只替换夹具位置的单因素实验**。

三份历史XML与当前 live 模型各有185个字段差异；导出并重编译后的当前模型与历史XML只有14项差异。后一个数字不能冒充原生活跃模型差异。还发现当前模型缺少历史 `robot0_link7_collision`（ngeom 240→239）、辅助相机改变、炉灶前方区域 half-size 0.03→0.04 m及微小数值差异。关节顺序、执行器参数、机器人运动学及抽取的物理 options 一致：dt=0.002、gravity=-9.81、Newton/100 iterations等。

current_xml_sync 单独引入当前导出/重载过程后，demo_0 末段最大偏差仍约15.12 mm，故不能将这条轨迹的主要晚期漂移归咎于导出引起的质量变化。历史场景位置恢复的独立对照提供了更直接证据。

历史采集时确切 MuJoCo/robosuite 提交和资产字节版本没有记录，无法独立复原旧引擎条件。本阶段没有切换物理版本，也没有手改 compiler、质量、摩擦或求解器来追求轨迹归零。

## 7. 证据等级与剩余不确定性

| 等级 | 结论 |
|---|---|
| 已证实 | 本文件 pre-state/post-observation 约定及移位监督正确；源/当前OSC动作定义一致；恢复物理状态不恢复完整控制器；刷新缓存及零旋转目标处理显著影响采样单步结果 |
| 已证实 | 固定夹具位置不在状态向量；只恢复历史场景位置使 demo_0 接触后偏差大幅降低；XML导出再编译会改变部分质量/惯性；三个专家序列在当前正式初始化中均原生成功 |
| 有证据支持，尚未普遍证实 | 场景差异是另外两条示范及其他接触任务的重要偏差来源；遗漏历史目标、夹爪累加器和nullspace参考解释一部分残差 |
| 在当前检查范围内已排除 | 将7维动作误作关节角、额外缩放或符号交换；该数据监督错移一拍；记录/当前的物理步长、重力、已抽取solver options不同。导出重编译本身不能解释demo_0主要接触后漂移 |
| 尚无法判断 | 历史引擎和资产的独立影响、全部隐藏历史的准确值、未测试任务的兼容性、训练后视觉策略的闭环性能 |

没有完整控制器检查点和缺失前5步，不可能声称已恢复相同完整状态。当前结果也不能把全部残差归到MuJoCo版本、warmstart或某个单一模型参数。

## 8. 训练可用性与最小建议

**Q1：能否做离线监督学习？GO。** 4410对监督样本经过数据及源码核验，动作语义、图像转换和状态字段明确。保持 Episode 划分、既有预处理、原始归一化动作，不用未知历史目标补造标签。此处没有证明模型已经学会任务。

**Q2：是否适合在当前环境评测？CONDITIONAL GO。** 三条专家动作在当前原生场景实际成功，支持接口可用；历史场景与正式场景不完全相同，接触附近存在域差异。正式评测必须继续固定上游版本、官方初态和原生成功规则，不能使用本诊断的历史注入提高成绩。报告成功率时注明测试任务、种子、初态及版本，不能将重放误差等同于策略性能。

**Q3：是否换数据、匹配旧版本或重采？目前无此必要性证据。** 优先级如下：

1. 保留现有数据/runtime；将本报告与版本、动作、预处理、时间对齐规则随实验归档。
2. 需要研究精确回放时，在隔离诊断中同时检查固定场景、控制器缓存/目标和实际活跃模型；不要把仅qpos/qvel恢复称为完整检查点。XML导出重载必须审计质量/惯性，不默认无损。
3. 后续若学习策略在当前场景失败，先用独立任务/种子、闭环动作与接触阶段分析区分泛化误差和环境偏移，再决定是否需要匹配旧资产/版本。现无证据支持先进行大规模迁移或重采集。

**Q4：能否进入 Phase 5B.3？CONDITIONAL GO。** 可以从现有单任务视觉 BC 开始受控训练与闭环评测；必须明确场景偏移和样本范围。本阶段到此停止，不实现 BC、Trainer、Checkpoint 或训练。

对后续算法的影响：BC逐步闭环可能修正开环轨迹漂移，开环偏差不能直接证明不可训练；ACT的长动作块需要单独检查接触阶段误差累积；VLA仍需保持同一OSC动作和图像/状态约定。对于世界模型，不能把这个79维片段当作完整Markov状态。上述均是研究风险判断，尚无模型实验结果。

## 9. 实际运行方式、改动与回归

项目只新增 `tools/diagnose_libero_consistency.py` 和本报告。Environment、Dataset、Runner、Policy、Replay、Evaluator、上游源码和官方HDF5均未修改。辅助分析脚本与结果保存在本机研究产物目录，未引入新的框架或依赖。

以下命令在项目目录执行。定义环境变量后，每个回放命令使用它们；缓存指向/tmp是因为本次执行环境不能写项目缓存，和仿真配置无关。

```bash
cd <project>
export MUJOCO_GL=egl
export NUMBA_CACHE_DIR=/tmp/embodiedlab-libero-numba
export MPLCONFIGDIR=/tmp/embodiedlab-libero-matplotlib
unset DISPLAY WAYLAND_DISPLAY

# 实际分批执行的12个主回放；输出文件名必须未存在。
.venv-libero/bin/python tools/diagnose_libero_consistency.py \
  --episodes demo_0 --plans history_raw history_sync stored_xml_sync \
  --output /tmp/phase5b2_replay_pilot.json
.venv-libero/bin/python tools/diagnose_libero_consistency.py \
  --episodes demo_0 --plans official \
  --output /tmp/phase5b2_official_pilot.json
.venv-libero/bin/python tools/diagnose_libero_consistency.py \
  --episodes demo_1 demo_2 --plans official history_raw history_sync stored_xml_sync \
  --output /tmp/phase5b2_replay_remaining.json

# 两个单因素对照。
.venv-libero/bin/python tools/diagnose_libero_consistency.py \
  --episodes demo_0 --plans current_xml_sync --output /tmp/phase5b2_xml_roundtrip.json
.venv-libero/bin/python tools/diagnose_libero_consistency.py \
  --episodes demo_0 --plans fixtures_sync --output /tmp/phase5b2_fixtures.json

# 只读源码/模型审计、单步实验、全部数据对齐检查。
.venv-libero/bin/python <local-artifacts>/phase5b2_scripts/audit_models.py
.venv-libero/bin/python <local-artifacts>/phase5b2_scripts/phase5b2_single.py
.venv-libero/bin/python <local-artifacts>/phase5b2_scripts/verify_alignment.py

.venv-libero/bin/python -m pytest -q
```

实际初次运行的 replay 输出写入本报告第2节所述本机研究产物目录，随后合并成 `phase5b2_replays.json`；上面使用新的/tmp文件名便于复跑，不会覆盖已有结果。审计/单步的初次执行位置为 `work/phase5b2/audit_models.py`、`work/phase5b2_single.py`、`work/phase5b2/verify_alignment.py`，本机研究产物目录中的脚本为相同代码的归档副本；它们的输出路径固定为第2节位置。

回归结果：**82 passed，4 warnings，72.94 s**。3条为robosuite日志API弃用告警，1条为只读环境不能写pytest缓存；无测试失败。诊断价值来自真实因果对照，未通过增加测试数量代替解释偏差。

**停止点：Phase 5B.2 完成。没有启动 Phase 5B.3。**

