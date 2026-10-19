# 历史实验与开发资料

本目录保留实测结论、协议和静态导出，原实验数字、Commit及指纹不改写。当前操作见[项目README](../../README.md)；下列旧阶段命令只描述当时环境，不自动适用于当前分支。安装/运行命令均从仓库根执行。

| 资料 | 内容 |
|---|---|
| [Phase5B.2](PHASE5B2_DIAGNOSIS.md) | 数据/仿真一致性诊断 |
| [Phase5B.3](PHASE5B3_REPORT.md) | VisualBC离线训练 |
| [GPU配置](GPU_SETUP_REPORT.md) | 5090设备与CUDA验证 |
| [Phase5B.4](PHASE5B4_REPORT.md) | E5真实闭环与失败分析 |
| [Phase5C](PHASE5C_REPORT.md) | 第二任务和稳定性验收 |
| [Phase6A](PHASE6A_OPENPI_REPORT.md) | 官方OpenPI接入 |
| [Phase6B](PHASE6B_BENCHMARK_REPORT.md) | 两模型六任务挑战集 |
| [接触任务](CONTACT_RICH_REPORT.md) | Wipe/GearMesh结果及Factory安装附录 |
| [ROS Phase7](PHASE7_REPORT.md)、[7B](PHASE7B_REPORT.md)、[7C](PHASE7C_REPORT.md)、[7D](PHASE7D_REPORT.md)、[7E](PHASE7E_REPORT.md)、[7F](PHASE7F_REPORT.md) | Fake Hardware与ROS视觉闭环历史 |
| [LIBERO-PRO调查](libero_pro_feasibility.md) | 可行性研究，未执行扰动Benchmark |

静态导出：[Phase6A汇总](phase6a_results.json)；Phase6B的[结果](phase6b/results.json)、[配对初态](phase6b/paired_initial_states.csv)、[验收摘要](phase6b/validation_summary.json)、[谓词过程](phase6b/native_predicate_progress.md)。只读分析入口现为`tools/analyze_challenge.py`。训练门禁仍在`src/embodiedlab/data/verified_sources.json`，审核证据见[数据来源](../data_sources/README.md)。

ROS完整恢复分支为本机`archive/ros2-phase7f-20261019`（`6160d00`），源码/脚本/Docker/controller及`upstream-ros2.json`不再位于当前分支；ROS报告中的命令在该分支使用。第一轮瘦身前全项目恢复标签为`pre-slimming-round1-20261019`。更早诊断脚本从`pre-consolidation-factory-20261019`恢复。

HDF5、Checkpoint、训练/评测结果、视频和第三方环境继续保留本机并被忽略，没有随归档删除。报告中的`<project>`、`<local-artifacts>`及本地结果路径需要按实际位置替换。新实验必须使用独立输出目录。
