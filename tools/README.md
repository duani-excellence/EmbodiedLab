# 数据、模型服务与结果分析

正式下载、训练、评测和自录回放入口位于 `scripts/`。这里的工具用于检查数据、运行官方模型服务或分析已有实验，从项目根目录运行：

```bash
.venv-libero/bin/python tools/inspect_dataset.py --help
```

| 工具 | 用途与范围 |
|---|---|
| `inspect_dataset.py` | 读取一个HDF5时间步并报告形状、对齐和元数据；可选导出单帧预览，不加载整段RGB。 |
| `analyze_challenge.py` | 只读manifest与PI/OFT结果，检查身份、异常分母和配对关系，输出统计和可选视频；不运行模型。 |
| `serve_openpi.py` | 可选官方π₀.₅模型服务入口；检查固定源码、锁文件和全部权重指纹，只监听本机。独立环境与GPU参数见[OpenPI说明](../docs/openpi/README.md)。 |
| `serve_openvla.py` | 可选官方联合OpenVLA-OFT模型服务；独立BF16 CUDA环境，复用官方预处理/动作head/夹爪，原件校验后创建运行视图。安装及挑战集命令见[Benchmark说明](../docs/benchmark/README.md)。 |

例如，只读取抽屉示范的一个时间步：

```bash
.venv-libero/bin/python tools/inspect_dataset.py \
  --dataset data/libero/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5 \
  --episode demo_0 --suite libero_goal --task-id 0
```

原始LIBERO示范的观测位于动作之后；训练Dataset使用`obs[t-1] → action[t]`。自录HDF5采用T动作/T+1观测，不使用上述移位。单步读取器保留来源中的原始任务别名；训练Dataset只按审核过的真实文件指纹解析标准任务身份。

Headless运行需要可访问EGL设备及项目README中的环境配置。历史炉灶一致性诊断、旧E5微基准和早期全量数据检查工具已移除，可从`pre-consolidation-factory-20261019`标签恢复；原始数据、权重和实验结果仍保留。当前数据校验由`LiberoTorchDataset`和现有测试执行。
