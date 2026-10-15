# EmbodiedLab Phase 5B.3：视觉 BC 模型与离线训练

> 历史实验记录；本机路径已脱敏，诊断工具路径已更新。原始结果保留在本机，数据、权重和配图未随仓库发布。当前安装与入口以[项目 README](../../README.md)为准。

日期：2026-10-15。项目：`<project>`。

## 1. 完成情况

已实现并实际验证一个 **194,920 参数**的共享 CNN + 状态 MLP 视觉 BC 模型，完成真实单 Batch 拟合、3轮短训练、20轮单任务离线训练、最佳 checkpoint 保存与恢复。没有实现在线 BC Policy，没有运行训练策略的 LIBERO 闭环评测。

固定40/10条 Episode、3539/871对监督样本。最佳模型为第5轮，验证 Loss **0.076097**，常量预测基线 **0.733590**；连续六维 MSE 相比常量下降 **70.30%**，夹爪准确率 **97.82%**。这些是离线动作预测指标，**不是原生任务成功率**。

结果并非各方面都改善：后期出现过拟合，rx分量仍弱于常量基线；夹爪错误主要集中在动作标签切换附近。没有扩大网络或进行超参数搜索来掩盖这些问题。

## 2. 实际项目结构与职责

```text
embodiedlab/
├── src/embodiedlab/policies/visual_bc.py   # 新增：网络、监督损失、7维动作解码
├── scripts/train_bc.py                    # 新增：统计、单Batch/完整训练、评估和checkpoint
├── tests/test_bc.py                       # 新增：真实数据梯度、动作语义和恢复验证
├── PHASE5B3_REPORT.md                      # 新增：本报告
├── README.md                             # 更新：运行、恢复、结果及边界
├── .gitignore                            # 更新：忽略training_results大文件
└── training_results/
    ├── phase5b3_overfit_seed42/
    ├── phase5b3_pilot_seed42/
    └── phase5b3_train_seed42/
        ├── best.pt
        ├── config.json
        ├── gradient_check.json
        ├── constant_baseline.json
        ├── history.json
        ├── summary.json
        └── losses.png
```

复用现有 `LiberoTorchDataset`、Episode划分和预处理；未增加 Trainer 基类、模型注册系统、另一套 HDF5 读取器或图像预处理。Environment、Runner、Recorder、Replay、原生任务/控制器/成功判定均未修改。Phase 5B.2 的报告和诊断脚本也保留原样。

交付副本位于 `<local-artifacts>/`：

- `phase5b3_results/train/`、`overfit/`：checkpoint、配置、数值结果及曲线。
- `phase5b3_results/pilot/`：短训练配置、历史与汇总；不重复归档pilot权重。
- `phase5b3_results/source/`、`manifest.json`：代码/依赖快照及各文件SHA256，结果包约1.89 MB。
- `phase5b3_action_audit.json/.py`：真实动作分布、分位数、常量基线和复核脚本。
- `phase5b3_error_analysis.json/.py`：最佳模型逐Episode和夹爪切换诊断。
- `phase5b3_training.log`、`phase5b3_pytest.txt`：正式训练与完整测试的实际输出。

## 3. 数据与损失设计

### 固定数据与对齐

数据仍为 `data/libero/libero_goal/turn_on_the_stove_demo.hdf5`，来源 `yifengzhu-hf/LIBERO-datasets@147fa5eb607bb696dd922c3518084a90954cd2f3`。训练前后SHA256均为：

```text
387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5
```

继续使用已验证的 `obs[t-1] → action[t]`，每条轨迹产生T-1个样本。Episode划分固定 `split_episodes(seed=42)`：Train 40条/3539样本，Validation 10条/871样本，集合不相交。`config.json`保存全部Episode名称，验证集不参与归一化或常量预测参数拟合。

RGB保持原方向、通道和数值约定，经现有预处理得到两路 `(B,3,128,128)` float32/[0,1]，没有翻转、旋转或随机增强。状态为 `(B,8)`：eef xyz[m]、hand body旋转向量[rad]、两指qpos[m]。动作是 `(B,7)` 原始归一化OSC增量，不是关节角。

状态仅做训练集均值/总体标准差标准化；std下限0.001，本次各实际std均大于该下限。8维mean/std存于模型buffer和配置中。RGB逐Batch懒读，只暂存小型状态/动作数值统计，不缓存全部图像。

### 动作分布决定夹爪头

实际监督标签频率：

| 集合 | -1/open | 0/hold | +1/close |
|---|---:|---:|---:|
| Train | 1819（51.40%） | 0 | 1720（48.60%） |
| Validation | 439（50.40%） | 0 | 432（49.60%） |

训练集前六维分布（原始归一化单位）：

| 维度 | 均值 | 标准差 | min | max |
|---|---:|---:|---:|---:|
| dx | -0.197288 | 0.235106 | -0.881250 | 0.460714 |
| dy | 0.211359 | 0.263816 | -0.393750 | 0.894643 |
| dz | -0.348310 | 0.278123 | -0.937500 | 0.458036 |
| rx | 0.010021 | 0.041112 | -0.172500 | 0.198214 |
| ry | 0.037129 | 0.050081 | -0.128571 | 0.245357 |
| rz | 0.115125 | 0.169934 | -0.143571 | 0.375000 |

因此夹爪采用2类 logits：class0=-1/open、class1=+1/close；不构造缺失的0类，不使用类别加权。`bc_loss`明确拒绝0或其他夹爪监督标签。这个模型没有学习hold能力，不能不加检查推广到含0动作的其他来源。

```text
motion_mse = mean((prediction[:6] - target[:6])²)  # 对Batch和6维一起求均值
gripper_ce = CrossEntropy(logits, class_target)
loss = motion_mse + gripper_ce
```

连续头使用tanh，保持[-1,1]。夹爪经argmax映射回-1/+1，与六维输出拼接成LIBERO动作；没有额外控制缩放。评估独立记录平移、旋转、逐维MSE/MAE、夹爪CE/混淆矩阵/precision/recall/宏F1。

常量基线前六维为训练动作均值；夹爪概率为训练频率，CE使用该固定分布，硬预测为多数类-1。它没有利用验证集统计，也没有被故意削弱。

## 4. 网络结构与执行流程

两个相机复用**同一个**编码器实例及权重：

| 层 | 单路输出 |
|---|---|
| 输入RGB | 3×128×128 |
| Conv 3→16，5×5，stride2，ReLU，MaxPool2 | 16×32×32 |
| Conv 16→32，3×3，ReLU，MaxPool2 | 32×16×16 |
| Conv 32→64，3×3，ReLU | 64×16×16 |
| AdaptiveAvgPool 4×4，Flatten，Linear 1024→128，ReLU | 128 |

状态MLP为8→32→32；两路128维图像特征和32维状态特征拼接为288维，经288→128/ReLU，再分别接6维tanh头和2维分类头。保留4×4粗空间网格，避免把整幅图直接压成一个全局平均点。没有Transformer、预训练骨干、BatchNorm、Dropout或注意力。

一次训练的实际顺序：

1. 用现有Dataset确定40/10条Episode及T-1索引范围。
2. 在训练DataLoader上拟合状态统计量和常量基线；建立网络及训练/验证Loader。
3. 全局样本索引→Episode/动作t→现有Dataset读取obs[t-1]及action[t]→既定RGB/状态转换。
4. 模型标准化状态、共享CNN分别编码图像、融合特征、输出两个动作头。
5. 计算MSE+CE，反向传播，检查有限梯度，Adam更新。
6. 每轮以eval/no-grad在独立验证Episode计算指标，按**总验证Loss**保存最佳checkpoint。
7. 重新加载最佳参数和归一化buffer，验证预测一致，并重新评估完整两个集合。

网络只接收Tensor，不访问MuJoCo。动作解码只是网络输出转换，尚未构成在线Policy。

## 5. 三步实际验证结果

### Step 1：前向、反向与设备

环境是Python3.10.20、PyTorch2.5.1+cpu、NumPy1.26.4；CUDA构建版本为None，`torch.cuda.is_available()==False`。`nvidia-smi`无法连接驱动。本阶段未安装GPU包或改变已验证LIBERO环境。

真实32样本Batch验证：输出解码shape `(32,7)`；初始Loss 0.779921，梯度L2为0.259081，所有可训练参数梯度有限且总体非零。自动化测试另用真实Batch4验证两输出头、归一化、动作语义和输入不被修改。CUDA路径因不可用明确skip，没有声称进行GPU实测。

### Step 2：单 Batch 过拟合

固定一个真实shuffle Batch，32样本含14个open、18个close。Adam lr=0.001，300次更新，仅训练该Batch，没有从验证集取样。每10步记录一次固定权重指标。

| 指标 | 初始 | 第300步 | 最佳记录，第270步 |
|---|---:|---:|---:|
| 总Loss | 0.779921 | 0.001938 | 0.001202 |
| motion MSE | 0.083709 | 0.001735 | 0.001134 |
| gripper CE | 0.696212 | 0.000202 | 0.000067 |
| gripper准确率 | 43.75% | 100% | 100% |

第300步逐维MAE `[dx,dy,dz,rx,ry,rz]`：`[0.045683,0.027672,0.043364,0.018946,0.025326,0.027231]`。梯度/损失正常，Loss下降约99.75%，足以验证小模型和训练管线具备拟合能力；不宣称已把该Batch误差降为0。

记录耗时14.36 s，使用CPU4线程。曲线有局部波动，最佳采样点早于最后一步。恢复最佳checkpoint的motion/logits最大差均为0。

本机保留图片（未随仓库发布）：真实单Batch拟合曲线，`<local-artifacts>/phase5b3_results/overfit/losses.png`。

### Step 3：完整离线训练

同一架构、Adam lr=0.001、batch32、seed42、4CPU线程、num_workers=0；不使用scheduler、weight decay或增强。先3轮pilot观察验证Loss `0.108888→0.087996→0.079251`，再从相同随机初始化运行20轮，不从单Batch模型或pilot权重继续训练。

pilot与正式训练第0～3轮的Loss、逐维误差及混淆矩阵逐项完全相同（不比较耗时），提供本机固定配置下的复现证据。没有搜索多组学习率/网络/随机种子。

**统一在保存的第5轮模型上重新评估：**

| 指标 | Train 3539样本 | Validation 871样本 | 常量Validation |
|---|---:|---:|---:|
| 总Loss | 0.090620 | 0.076097 | 0.733590 |
| motion MSE | 0.011452 | 0.011960 | 0.040277 |
| 平移MSE | 0.019405 | 0.019974 | 0.069451 |
| 旋转MSE | 0.003499 | 0.003946 | 0.011102 |
| 平移MAE | 0.103281 | 0.107507 | 0.218228 |
| 旋转MAE | 0.043632 | 0.046330 | 0.074180 |
| gripper CE | 0.079168 | 0.064137 | 0.693314 |
| gripper准确率 | 96.67% | 97.82% | 50.40% |
| gripper宏F1 | 0.966657 | 0.978186 | 0.335115 |

验证集平移/旋转MSE相比常量分别降低71.24%/64.45%。同一固定模型的训练与验证连续误差接近；总Loss在验证集较低，不能据此推导发生泄漏或验证更难，Episode不同导致分类难度也不同。固定列表、训练统计来源和独立逐Episode复算已核对。

验证集逐维结果（统一使用第5轮，不挑选不同轮次的最佳分量）：

| 分量 | 模型MAE | 常量MAE | 模型MSE | 常量MSE |
|---|---:|---:|---:|---:|
| dx | 0.113273 | 0.207425 | 0.021823 | 0.058855 |
| dy | 0.089442 | 0.216226 | 0.014780 | 0.069216 |
| dz | 0.119807 | 0.231034 | 0.023320 | 0.080283 |
| rx | **0.032655** | **0.023802** | **0.001581** | **0.001222** |
| ry | 0.043249 | 0.043667 | 0.002853 | 0.003063 |
| rz | 0.063087 | 0.155072 | 0.007405 | 0.029021 |

均为归一化动作误差。平移MAE若仅按控制器缩放换算，每轴平均约5.38 mm指令差；旋转向量分量平均约0.02317 rad指令差。**这不是仿真实际跟踪位置或SO(3)姿态误差。**

验证夹爪混淆矩阵（行=真值open/close，列=预测open/close）：

```text
[[425, 14],
 [  5,427]]
```

open/close precision为98.84%/96.83%，recall为96.81%/98.84%；常量只预测open，不能凭多数类准确率宣称夹爪学会了控制。

平均每轮train+validation耗时 **5.63 s**（中位数5.62 s，范围5.08～6.33 s）；20轮run总计 **124.44 s**。pilot为27.21 s，单Batch为14.36 s。各run计时含其统计/初始评估/恢复检查，不含末尾绘图；不是纯神经网络计算时间。

本机保留图片（未随仓库发布）：训练与验证曲线，`<local-artifacts>/phase5b3_results/train/losses.png`。

纵轴为对数。训练曲线是该轮权重持续更新时的逐Batch加权均值；验证曲线是每轮结束后的固定模型指标，二者不是同一权重快照。上面的最终Train/Validation表则使用同一个恢复后的最佳模型。

## 6. 过拟合与主要失败模式

1. **后期训练主要没有改善验证总Loss。** 第5轮Validation=0.076097；第20轮训练过程均值0.054092、Validation=0.107339。第20轮验证motion MSE降至0.010433，但gripper CE升至0.096906、准确率降至96.44%，所以按预定总Loss准则保留第5轮。没有把更小的连续误差与另一轮更好的分类成绩拼成虚构模型。
2. **夹爪时机比总体准确率提示的更难。** 对最佳模型全部50条Episode重新真实推理。验证集11个标签切换点的±2个监督步窗口，共55/871样本（6.31%），包含17/19个分类错误（89.47%）。窗口内错误率30.91%、CE0.703908；窗口外816样本仅2错，错误率0.245%、CE0.021015。错误主要集中于切换时序，而不是持续open/close段。该相关性不证明历史信息缺失或某个传感量是唯一原因。
3. **rx单轴没有超过均值基线。** 验证rx MSE高29.45%、MAE高37.20%；训练rx MSE也高17.78%，不能简单归因为只在未见Episode上泛化失败。该轴训练方差0.001690，显著低于平移轴0.055～0.077；在等权六维MSE和分类总目标下贡献较小。尚未进行权重/时序消融，不断言调整权重一定解决，也不靠加大模型来回避。
4. **Episode难度不均匀。** 验证总Loss最高的demo_36/33/12分别为0.247433/0.086228/0.078004；连续MSE最高为demo_0/12/36。demo_36有6次夹爪错误，说明分类与连续动作的难点不完全一致。逐Episode原始数值见error_analysis JSON。

独立逐Episode聚合与训练summary最大差约`2.65e-8`（Batch规约/浮点边界）；不是另一组模型结果。未做state-only消融，因此不能从当前实验断言RGB本身带来多少收益。验证集参与了checkpoint选择，也不是独立最终测试集。

Phase 5B.2 已确认的场景、控制器隐藏状态及模型版本风险仍然存在；本阶段的离线拟合没有消除这些风险。8维状态也没有速度/控制器目标历史，可能影响单帧决策，但本次未做因果验证。

## 7. Checkpoint内容及实际恢复

`training_results/phase5b3_train_seed42/best.pt`约771 KiB，包含：`format_version=1`、`model_state_dict`、epoch=5、验证指标、完整配置。归一化mean/std随state_dict保存；配置保存数据来源/指纹、划分列表、对齐、图像规则、类别映射、学习率、Batch、种子、线程、Torch版本和架构版本。

```python
import torch
from embodiedlab.policies.visual_bc import VisualBC, decode_actions

checkpoint = torch.load("training_results/phase5b3_train_seed42/best.pt",
                        map_location="cpu", weights_only=True)
model = VisualBC().eval()
model.load_state_dict(checkpoint["model_state_dict"])
with torch.no_grad():
    output = model(batch["agentview_image"], batch["wrist_image"], batch["robot_state"])
    actions = decode_actions(output)  # batch来自原有Dataset；输出(B,7)
```

训练程序实际从磁盘重新加载并比较保存时4样本的motion/logits，最大差均为 **0**，然后用恢复模型重新评估Train/Validation全量样本。测试另验证非默认归一化buffer恢复。checkpoint SHA256：

```text
00e37e0be92296c2a9a2753d83320768e62165cfbf6526196c860b79a405978b
```

这是推理恢复/从固定种子完整重跑所需的最小保存；**不包含优化器续训状态，不提供断点续训接口**。

## 8. 实际命令与测试

以下是实际执行的配置。结果目录已存在，复跑时省略`--output-dir`自动创建新运行名，或改为新目录，不能覆盖原实验。

```bash
cd <project>
export MPLCONFIGDIR=/tmp/embodiedlab-libero-matplotlib

.venv-libero/bin/python scripts/train_bc.py --mode overfit --steps 300 \
  --batch-size 32 --seed 42 --threads 4 \
  --output-dir training_results/phase5b3_overfit_seed42

.venv-libero/bin/python scripts/train_bc.py --mode train --epochs 3 \
  --batch-size 32 --seed 42 --threads 4 \
  --output-dir training_results/phase5b3_pilot_seed42

.venv-libero/bin/python scripts/train_bc.py --mode train --epochs 20 \
  --batch-size 32 --seed 42 --threads 4 \
  --output-dir training_results/phase5b3_train_seed42

env -u DISPLAY -u WAYLAND_DISPLAY MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  NUMBA_CACHE_DIR=/tmp/embodiedlab-libero-numba \
  MPLCONFIGDIR=/tmp/embodiedlab-libero-matplotlib OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-libero/bin/python -m pytest -q -p no:cacheprovider
```

实际完整测试：**100 passed，1 skipped，3 warnings in 72.21 s**。原82项通过；新增18项通过，CUDA项因不可用跳过。3条warning为原有robosuite日志API弃用提示。新测试覆盖真实前向/反向、有限非零梯度、两个动作头、输入不变、训练/评估重复性、归一化与checkpoint、零夹爪标签拒绝及关键非法维度。

训练脚本不创建仿真；完整回归测试仍包含原有真实LIBERO数据录制/回放验证，未进行BC在线策略评测。官方HDF5及其预处理代码未被修改。

## 9. 停止点与下一阶段

Phase 5B.3完成：得到第一个简洁、可恢复、实际训练过的视觉BC模型，同时保留其过拟合和分量退步证据。

下一阶段Phase 5B.4应先用当前E5 checkpoint实现最小BC Policy包装，在既定官方reset、原生OSC和成功规则下做小规模真实闭环评测，重点记录夹爪切换与接触阶段偏差。尚无任务成功率结论；本阶段到此停止，不运行在线评测，也不开发其他算法。

