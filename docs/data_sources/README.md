# 已审核的训练数据来源

这两份 JSON 是 Phase 5C 对固定官方 HDF5 的实际审核记录，保留来源 revision、文件 SHA256、50 条 Episode 的时间对齐检查，以及少量原生仿真/图像约定诊断。

| 证据 | LIBERO-Goal 任务 | HDF5 SHA256 |
|---|---|---|
| [stove_task7.json](stove_task7.json) | task 7，turn_on_the_stove | `387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5` |
| [drawer_task0.json](drawer_task0.json) | task 0，open_the_middle_drawer_of_the_cabinet | `20252c7cf98cd7437061f7f200ae7b6cb6219fabbd53b4536dfaa8abda6ab737` |

公开副本仅将原审核记录中的本机路径替为占位符；测量值没有改写。`reproduce_command` 记载历史实验上下文，包含未随仓库分发的当时辅助脚本路径，不能原样执行。当前数据查看入口是 `tools/inspect_dataset.py`，训练来源与对齐校验由 `LiberoTorchDataset` 执行，主流程见[README](../../README.md)。原始完整审核文件继续保留在本机实验目录。

两份来源均验证为 post-action observation，训练采用 `obs[t-1] → action[t]`；这不是对其他 HDF5 的默认假设。运行时审核清单位于 `src/embodiedlab/data/verified_sources.json`，未知或被修改的数据会被拒绝。抽屉数据中的旧任务别名通过精确文件指纹对应当前 task 0。

历史状态诊断没有恢复完整控制器内部状态，不能据此声称专家动力学已精确复现。数据仍可用于已说明风险的监督学习；离线损失和原生闭环成功率必须分别报告。

示范来源为 [LIBERO 官方项目](https://github.com/Lifelong-Robot-Learning/LIBERO) 指向的 [LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)，固定 revision `147fa5eb607bb696dd922c3518084a90954cd2f3`。官方数据遵循 CC BY 4.0；本仓库不分发 HDF5、RGB 或模型权重，第三方许可见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
