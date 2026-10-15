# 检查与诊断工具

正式下载、训练、评测和自录回放入口位于 `scripts/`。这里的工具用于检查数据或复核特定实验，从项目根目录运行：

```bash
.venv-libero/bin/python tools/inspect_dataset.py --help
.venv-libero/bin/python tools/inspect_torch_dataset.py --help
.venv-libero/bin/python tools/diagnose_libero_consistency.py --help
.venv-libero/bin/python tools/benchmark_bc.py --help
```

| 工具 | 用途与范围 |
|---|---|
| `inspect_dataset.py` | 读取一个HDF5时间步并报告形状、对齐和元数据；可选导出单帧预览，不加载整段RGB。 |
| `inspect_torch_dataset.py` | 对已审核来源执行真实train/validation读取检查；`--dataset`支持来源清单中的两项任务。完整命令会遍历选中数据的全部样本，不训练模型。 |
| `diagnose_libero_consistency.py` | 炉灶专家数据的部分状态/历史诊断。直接恢复状态仅用于该诊断，不代表生产环境的reset或官方专家完全复现。 |
| `benchmark_bc.py` | 原炉灶E5配置的CPU/CUDA微基准，会执行模型更新。不是任意任务的通用训练或任务成功率基准。 |

例如，只读取抽屉示范的一个时间步：

```bash
.venv-libero/bin/python tools/inspect_dataset.py \
  --dataset data/libero/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5 \
  --episode demo_0 --suite libero_goal --task-id 0
```

原始LIBERO示范的观测位于动作之后；训练Dataset使用`obs[t-1] → action[t]`。自录HDF5采用T动作/T+1观测，不使用上述移位。单步读取器保留来源中的原始任务别名；训练Dataset只按审核过的真实文件指纹解析标准任务身份。

Headless原生诊断需要可访问EGL设备及项目README中的环境配置。完整诊断和微基准要求显式新输出路径，避免覆盖历史结果。无需为了查看`--help`启动仿真或运行模型更新。
