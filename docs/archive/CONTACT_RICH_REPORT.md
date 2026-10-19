# Contact-Rich Benchmark：实际结果与任务取舍

更新2026-10-19。**Wipe此前已完成60条正式闭环；本次精简项目后，官方Factory GearMesh PPO再完成1条Smoke与10条固定种子闭环，正式最终成功10/10。** 保留LIBERO、VisualBC、OpenPI和OpenVLA-OFT。未Push、未训练VLA、未接入ROS 2或自建任务。第1–3节保留2026-10-18 Wipe实测结果，第4节为本次Factory结果。

## 1. 接入范围与策略

- 环境：原生 **robosuite 1.4.0 Wipe / Panda / WipingGripper**，MuJoCo3.3.2。官方100个污点、20Hz、1000步时限、奖励、安全终止及成功判定均保留。动作是六维世界坐标OSC位姿增量，平移±0.05m、旋转±0.5rad；擦拭工具没有夹爪动作。
- **touch_position**：原生固定阻抗OSC_POSE（kp150、阻尼比1）；首次触碰定位表面后固定虚拟压入深度3.5mm，闭环跟随剩余污点。不故意将它绑死在名义桌高。
- **force_admittance**：同一OSC、污点选择、姿态与触碰流程；额外用真实腕部传感器调节Z方向虚拟平衡位置，目标8N，导纳增益0.001m/(N·s)。

两者是本项目实现的简洁传统控制基线，**不是官方预训练专家或学习模型**。它们使用官方污点XY/缓存标记、末端姿态、腕部力，属于oracle物体状态基线；真实桌高和仿真接触力不输入策略。没有找到经核验、可直接匹配当前Wipe工具与版本的现成学习权重，因此没有将LIBERO OpenPI/OFT换个任务名就接入。

Wipe采用原有LIBERO运行环境，未安装或升级其依赖。官方Wipe、arena、工具及OSC源码与固定[tag v1.4.0 / commit fbee5844](https://github.com/ARISE-Initiative/robosuite/tree/fbee5844ff5632f5b5698e204ec5357ca50be0df)核对一致，差异仅为发行包换行符。

## 2. 实验协议与真实结果

开发试跑使用seed0/1，额外用seed2验证实际非零桌高变化；不计入正式分母。正式种子固定 **42–51**，每条件、每策略10条，初始机器人/污点状态指纹逐对相同。

| 条件 | 设置 |
|---|---|
| 默认 | 原生默认任务与物理参数 |
| 高度变化 | 官方`table_height_std=0.01`，采样`min(0, Normal(0,1cm))`；本批6/10实际降低，幅度2.91–15.60mm，另外4/10恰为名义高度；不筛掉这些样本 |
| 短时外载 | 首次>1N工具接触后30步，向工具质心施加12N世界Z向上力，持续0.5s；不修改位置、模型参数或成功判定 |

每个2ms物理步读取工具—桌面接触对法向力。接触保持率定义为擦拭阶段`Fn>1N`的物理样本比例；力RMSE相对8N，表中均为先计算单Episode指标、再对10条取平均，非将不同长度样本混池。

| 条件 | 策略 | 原生成功 | 平均长度/步 | 接触保持率 | 平均法向力/N | 力RMSE/N |
|---|---|---:|---:|---:|---:|---:|
| 默认 | 触碰定位＋固定深度OSC | 10/10 | 292.2 | 98.61% | 8.10 | 4.70 |
| 默认 | 腕部力反馈导纳 | 10/10 | 291.2 | 97.88% | 8.06 | 4.57 |
| 高度变化 | 触碰定位＋固定深度OSC | 10/10 | 298.9 | 98.67% | 8.05 | 4.74 |
| 高度变化 | 腕部力反馈导纳 | 10/10 | 294.7 | 97.75% | 8.06 | 4.57 |
| 短时外载 | 触碰定位＋固定深度OSC | 10/10 | 295.5 | 88.59% | 7.32 | 5.14 |
| 短时外载 | 腕部力反馈导纳 | 10/10 | 299.4 | 85.71% | 7.15 | 5.11 |

全部完成、无异常、无超时、无原生安全终止；全批500Hz法向力样本峰值 **33.52N**，没有>60N样本。60条正式数据共 **17,719动作、442,975个物理接触样本**，运行184.75秒。每格仅10条，10/10的Wilson95%区间为72.25%–100%，不能据此证明一般成功率为100%或方法总体优劣。

### 外载恢复

接触丢失：外载期间出现`Fn>1N`样本少于20%的控制区间。稳定恢复：撤载后连续三步，每步至少80%物理样本重新接触；记录这三步中的首步时间。未满足前Episode已结束的样本作右删失，不填零，也不等同于确定恢复失败。

| 策略 | 施加载荷 | 出现接触丢失 | 观察到稳定恢复 | 未确认即任务结束 | 已观察恢复的均值/中位数 |
|---|---:|---:|---:|---:|---|
| 固定深度OSC | 10/10 | 9/10 | 8/9 | 1/9 | 0.244s / 0.225s |
| 力反馈导纳 | 10/10 | 10/10 | 9/10 | 1/10 | 0.356s / 0.300s |

固定深度另有1条未丢失接触，不需要恢复；两策略未确认稳定恢复的都是seed49，任务在撤载后11步结束，末步已有接触，但未积累连续三步合格窗口。共同具有可观测恢复时间的8对中，导纳比位置策略平均慢0.05s，只作描述。

## 3. 主要发现、修复与局限

**没有任务层面的失败，存在接触质量不足。** 平均力接近8N不等于精确跟踪：动态RMSE仍约4.6–5.1N，外载时存在接触中断。导纳稍降低力误差，却没有提高接触保持率，不能预设力反馈必然更优。

**腕部载荷不等于桌面接触力。** seed45真实曲线显示，工具受12N向上外载时，工具—桌面法向力已经为零，但腕部估计仍接近12N；简单单轴导纳不能区分载荷来源。图保存在`evaluation_results/contact_wipe_formal_v2/contact_force.png`。这是传感与控制范围的证据，不是对所有导纳方法的否定。

**修复了一处重要时间错位。** 原生[step流程](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/environments/base.py)先缓存观测，随后[Wipe.reward](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/environments/manipulation/wipe.py)更新擦除列表。375步实际回放确认：末帧缓存标记显示98，原生真实计数已100。初版按缓存差分得到的“零力擦除”是错位诊断，已撤回；没有据此声称物理异常。

修复后同时保留原始缓存策略输入与奖励后的真实计数，**相同全部种子重新执行60条**。v1完整保留并附`SUPERSEDED.json`，只以v2汇总，不合并分母。两批60条的全部共同原始数组逐值一致；修复只影响记录诊断。正确对齐后，本批没有零法向力时的实际擦除事件。

原生成功本质仍是几何覆盖代理，不单独保证持续压力；结合成功率饱和，Wipe更适合作为接触控制微基准，而不是单凭擦净率排名的新算法挑战集。没有通过提高不合理摩擦、改质量或极端扰动制造难度。

## 4. Factory GearMesh：2026-10-19实际接入

采用官方 **Isaac Lab v2.3.0 / `3c6e67bb5c7ada942a6d1884ab69338f57596f77`**、**Isaac Sim5.1.0**、NVIDIA RL-Games **1.6.1 / `6b3534f29568158e9e29ec8bf83cc88fce5f0cae`**、Torch **2.7.0+cu128**、Python3.11.15，全部安装到独立`.venv-factory`。本机RTX5090 D v2已实际执行sm120张量运算和原生PPO闭环。

任务为[官方Factory GearMesh](https://github.com/isaac-sim/IsaacLab/tree/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab_tasks/isaaclab_tasks/direct/factory)，ID `Isaac-Factory-GearMesh-Direct-v0`，**没有接入Forge**。使用官方Sim5.1发布的循环PPO，Checkpoint为212,889,005B，epoch127，SHA256：`c5096f3ea1f8e1fadb6e5546296c3db57a135e25ea2645a6effc2403d93401e1`。19维actor输入、43维critic及两层1024维LSTM的权重/归一化维度均匹配；原始训练Git commit未随Checkpoint给出，不宣称已确认其精确训练源码版本。

选择2.3.0是因为2.3.2的Starlette0.49.1与Sim5.1固定FastAPI0.115.7冲突；2.3.0正式依赖一致，214包检查通过，未修改上游。默认PyTorch CUDA12.6 wheel缺少sm120，隔离环境改用官方cu128 wheel。Factory实际任务逻辑和参数经源码对照保持一致。

动作是六维原生末端增量，位置比例0.02m、旋转比例0.097rad，保留官方限幅、EMA0.2、原生控制器及夹爪闭合流程。actor仅接收原生末端相对位置/姿态/速度和前一动作；物体、接触和成功诊断不输入网络。没有人工动作修正、训练、力矩冒充接触力或Episode内状态覆盖。

物理120Hz、控制15Hz。原生配置20秒/300步，源码在计数299时结束，**所有Episode均完整执行299步/19.933秒**。原生成功要求齿轮底部XY偏差<2.5mm、带符号Z偏差<1mm；不提前终止。分别保存最终成功、曾经成功及首次成功步，并在自动reset之前捕获终态。原生末步两个done标志均为真；适配层明确将时限标为truncated，不能将它当作失败。

先运行seed7 Smoke，成功于第80步且终态成功。之后正式固定seed **42–51**，不依表现更换初态，全部保留。这是官方随机reset的十个固定种子，不是LIBERO的离散初态索引；保存了各自实际起点和指纹，十个起点不同。未增加额外扰动。

| Seed | 最终结果 | 首次成功/步 | 最终XY/mm | 最终Z/mm | 齿轮法向合向量采样峰值/N |
|---|---|---:|---:|---:|---:|
| 42 | 成功 | 35 | 0.200 | 0.188 | 6.38 |
| 43 | 成功 | 253 | 0.238 | 0.320 | 9.22 |
| 44 | 成功 | 44 | 0.083 | 0.140 | 17.95 |
| 45 | 成功 | 127 | 0.192 | 0.239 | 17.92 |
| 46 | 成功 | 33 | 0.240 | 0.477 | 24.56 |
| 47 | 成功 | 147 | 0.197 | 0.302 | 20.13 |
| 48 | 成功 | 61 | 0.238 | 0.388 | 11.01 |
| 49 | 成功 | 100 | 0.063 | 0.108 | 28.23 |
| 50 | 成功 | 76 | 0.165 | 0.470 | 17.86 |
| 51 | 成功 | 44 | 0.187 | 0.565 | 15.59 |

**正式最终成功 10/10，曾经成功10/10，共2,990个模型动作，耗时784.91秒。** 首次成功33–253步；10/10的Wilson95%区间为72.25%–100%，不代表总体成功率100%或任务饱和。平均GPU推理0.868ms/步，计时包含官方player前向/采样，不包含仿真。

**观察到了接触受阻与自主恢复，没有本批最终失败。** seed43第100–200步Z偏差持续9.64–9.80mm，99.0%的采样时刻至少一个齿轮接触对的法向合向量>0.1N，直到第253步才首次成功。轨迹支持“受阻后恢复”的描述；仅凭合向量与位姿尚不能确定具体齿面卡滞原因或恢复因果机制。没有为了得到失败而改物理或继续加扰动。

接触来自只读PhysX contact view，过滤held gear与原生底座、小/大齿轮及左右手指。保存的是每个控制末物理步的**世界法向力合向量**，不含切向摩擦，多个接触点可能抵消；全批采样峰值28.23N。它不是120Hz真峰值、完整wrench或腕部传感器，不与Wipe的500Hz标量峰值直接比较。

连续运行采样RSS为5589.64–5589.72MiB，句柄始终142；GPU总显存7874–7922MiB（包含桌面，采样不覆盖启动峰值）。进程退出后回到2769MiB桌面基线。上游单环境Fabric克隆错误日志及退出时USD引用警告仍出现；实际源环境、全部刚体、接触、十条完整闭环和进程资源释放均验证通过，没有改物理来压制日志。

入口只增加`scripts/evaluate_gearmesh.py`：两个局部包装复用现有`run_episode()`与官方player，不建立新环境/算法注册系统。独立依赖为`requirements-factory.lock`。正式实验入口版本`6067cf9`，协议另含源码SHA；随后只修正`--help`参数注册顺序，运行函数/类AST核对一致。安装和真实命令见[本报告安装附录](#附录factory-安装与运行)：

```bash
env PYTHONPATH="$PWD/src" OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-factory/bin/python scripts/evaluate_gearmesh.py --headless --device cuda:0 \
  --checkpoint .deps/isaac-factory/checkpoint.pth --contacts \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --output-dir evaluation_results/factory_gearmesh_repeat01
```

正式结果与十条NPZ位于`evaluation_results/factory_gearmesh_formal_v1/`。记录T动作/奖励、T+1原生观测/位姿、接触、时间戳与模型延迟，加载不使用pickle；全部形状、有限性、时间轴和按物理位姿重算的原生成功标志核验通过。未把Factory数据塞入LIBERO专用HDF5/Replay；目前只支持单环境原生状态PPO，未声称支持GearMesh视觉模型或完整力觉策略。原生任务从已经夹持齿轮开始，不评估抓取规划或视觉泛化。

## 5. 代码、复现与回归

下列为2026-10-18 Wipe验收记录；本次项目收敛与回归见5.1。

新增三个实际使用文件：`envs/wipe_env.py`（官方环境与物理步接触监测）、`policies/wipe.py`（两个共享流程的传统策略）、`scripts/evaluate_wipe.py`（真实评测、数值记录、恢复指标、少量视频）。复用现有Policy、Runner及Wilson统计；原LIBERO Environment、Evaluator、Recorder、Dataset、Replay、VisualBC和VLA实现不改。

原HDF5记录器严格要求LIBERO七维动作、八维状态及双相机，不将无夹爪Wipe伪装进去。Wipe用简单NPZ保存T动作、T+1姿态/腕力/原始缓存标记/真实擦除计数，以及每步25个接触样本；不使用pickle或把所有RGB存进NPZ。视频独立保存。

最终实验代码commit：`e50395e2eecc5891b3733969a76fbacaf9d0d743`；原生控制器、奖励、动力学参数及成功判定未改，声明的场景条件见上表。原始结果、protocol、60条数值轨迹与首seed的6个视频位于`evaluation_results/contact_wipe_formal_v2/`。

实际命令（从项目根目录执行；重跑时换全新输出目录）：

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export NUMBA_CACHE_DIR=/tmp/embodiedlab-contact-numba
export MPLCONFIGDIR=/tmp/embodiedlab-contact-matplotlib
unset DISPLAY WAYLAND_DISPLAY
.venv-libero-gpu/bin/python scripts/evaluate_wipe.py \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --scenarios nominal height_variation upward_pulse --videos \
  --output-dir evaluation_results/contact_wipe_formal_v2
```

验收以真实实验为主，没有新增Mock测试体系：

- 无监测包装的官方环境重执行253个真实动作，位置/姿态/奖励偏差0、擦除标记完全一致，证明监测未改变该轨迹的物理。
- 60条记录维度、数值、时间轴及30对初态核验通过；从记录的真实pre-action观测重算17,719个策略动作，最大误差0；原输入文件SHA不变。
- 实际拒绝七维、NaN和越界动作，不推进仿真；close可重复调用。
- 原有完整测试：**GPU 308通过；CPU 304通过、4项CUDA检查跳过**。没有为增加测试数编写重复用例。
- 原LIBERO炉灶任务新执行5步接口Smoke、原Recorder保存、原Replay重执行；状态、关节、RGB、奖励和时间偏差均0，终止标志一致。该短Zero策略实验不算性能基线。

### 5.1 2026-10-19项目收敛与回归

清理前创建标签`pre-consolidation-factory-20261019`，新分支`cleanup-factory-gearmesh`。检查全项目引用后删除三个独立历史工具：`diagnose_libero_consistency.py`243行、`benchmark_bc.py`271行、`inspect_torch_dataset.py`209行，共723行。它们未被核心代码或测试导入；有效的数据来源/对齐校验保留在Dataset与测试中。历史报告、全部数据/权重/实验结果/环境/缓存均保留，旧脚本可从标签恢复。

保留真实使用的E5兼容、模型服务、挑战集分析及五行OFT入口。清理阶段Python文件54→51；加入Factory入口后为52，Python总行8411→7973，净减少438行；追踪文件94→93。没有为了目录形式合并职责不同的核心模块。

- GPU原有完整测试：**308通过**（31.28s）；CPU：**304通过、4项CUDA跳过**（30.75s）。未新增Mock测试。
- 所有原13个scripts入口`--help`通过；新Factory入口也在其独立环境验证帮助参数；冻结的214包lock dry-run无需变更、`uv pip check`全部兼容。
- 真实抽屉数据GPU训练入口执行2个更新，固定Batch loss **0.73952→0.67046**，Checkpoint恢复一致、源HDF5未变。这是短工程Smoke，不是模型重训实验。
- 原炉灶E5原权重实际闭环：seed42/init0，第225步成功；原Recorder记录后Replay225步，状态/关节/RGB/奖励/时间偏差全部0，终止标志一致。
- 现有Wipe两策略各跑seed42 nominal完整Episode，均成功。未重跑旧120条VLA挑战集；其源码/环境未改，相关现有回归和入口检查通过。
- Isaac安装后再检查原环境：Torch2.9.1+cu128、MuJoCo3.3.2、robosuite1.4.0保持原版本；E5 CUDA推理、真实LIBERO五步记录/零偏差Replay、Wipe两步仿真均通过。

新实验写入独立`cleanup_*`和`factory_gearmesh_*`输出目录，原训练结果、原始示范、历史Checkpoint和Phase6A/6B结果没有删除或覆盖。

## 6. 长期保留判断

| 任务 | 建议 |
|---|---|
| LIBERO | 保留，继续承担视觉与多阶段任务评测；不与Wipe合并成功率 |
| Wipe | **有条件保留为低层接触/力跟踪/恢复回归任务**；目前强传统基线成功率已饱和，不适合作为单一学习算法排名依据 |
| GearMesh | **保留为复杂接触与受阻恢复专项Benchmark**；官方PPO已实际可用。成功率之外应看首次成功时间、受阻过程与接触；本批10条尚不足以判断总体饱和，当前仅一个原生状态PPO基线 |

本次到此停止。没有Push、训练VLA、接入Forge/ROS 2或新增其他模型。

## 附录：Factory 安装与运行

以下步骤从项目根目录执行；已有环境直接复用，无需重新下载或安装。


使用独立Python3.11环境：官方Isaac Lab **v2.3.0**、Isaac Sim **5.1.0**、NVIDIA RL-Games **1.6.1**、Torch **2.7.0+cu128**。不要将[Factory锁](../../requirements-factory.lock)安装到现有LIBERO或VLA环境。这里仅接入Factory，不接入Forge。

新安装先取得固定上游源码（已有环境无需重建）：

```bash
mkdir -p .deps/isaac-factory
curl --http1.1 -fL --retry 2 -o .deps/isaac-factory/IsaacLab-3c6e67bb.tar.gz \
  https://codeload.github.com/isaac-sim/IsaacLab/tar.gz/3c6e67bb5c7ada942a6d1884ab69338f57596f77
tar -xzf .deps/isaac-factory/IsaacLab-3c6e67bb.tar.gz -C .deps
curl --http1.1 -fL --retry 2 -o .deps/isaac-factory/rl_games-6b3534f.tar.gz \
  https://codeload.github.com/isaac-sim/rl_games/tar.gz/6b3534f29568158e9e29ec8bf83cc88fce5f0cae
tar -xzf .deps/isaac-factory/rl_games-6b3534f.tar.gz -C .deps/isaac-factory
uv venv --python 3.11 .venv-factory
printf 'setuptools==80.9.0\n' > .deps/isaac-factory/build-constraints.txt
uv pip install --python .venv-factory/bin/python --index-strategy unsafe-best-match \
  --build-constraints .deps/isaac-factory/build-constraints.txt -r requirements-factory.lock
```

仿真器及其扩展缓存下载较大。首次使用需接受NVIDIA发行条款，任务资产由官方运行时按需获取。官方Sim5.1发布的GearMesh PPO：

```bash
curl --http1.1 -fL --retry 2 -C - -o .deps/isaac-factory/checkpoint.pth \
  https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rl_games/Isaac-Factory-GearMesh-Direct-v0/checkpoint.pth
echo 'c5096f3ea1f8e1fadb6e5546296c3db57a135e25ea2645a6effc2403d93401e1  .deps/isaac-factory/checkpoint.pth' | sha256sum --check
env PYTHONPATH="$PWD/src" OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-factory/bin/python scripts/evaluate_gearmesh.py --headless --device cuda:0 \
  --checkpoint .deps/isaac-factory/checkpoint.pth --contacts \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --output-dir evaluation_results/factory_gearmesh_repeat01
```

输出目录必须不存在。六维原生增量动作、19维actor输入、循环PPO与原生reset均保留；成功不提前终止，单独报告最终成功和曾经成功。官方20秒配置实际在第299步结束。NPZ保存T动作/T+1观测和自动reset前的终态；接触量为15Hz采样的世界法向力合向量，不能当作120Hz峰值或完整力矩。仅复用现有Runner，不将Factory记录伪装为七维LIBERO HDF5。
