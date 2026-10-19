# LIBERO-PRO Position Perturbation 可行性审查

审查日期：2026-10-16。结论：**可隔离研究，但本阶段不应直接作为标准 LIBERO 的附加初态运行；建议先完成标准120条，再决定独立小实验。** 本次实际执行源码、BDDL、初始化数组和解析器检查；没有执行 PRO 仿真、模型推理或安装依赖，没有改动 EmbodiedLab 或现有 LIBERO 资产。

## 1. 固定来源与实际取得内容

- 官方代码：[Zxy-MLlab/LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/eafdb809426b13153aa1e4c42d6601844217dfec)，固定 SHA `eafdb809426b13153aa1e4c42d6601844217dfec`，提交时间2026-03-23，克隆后工作树干净。
- 官方数据：[zhouxueyang/LIBERO-Pro](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro/tree/c86fc3b8293185a6f373677018ff3e37f8391602)，固定 revision `c86fc3b8293185a6f373677018ff3e37f8391602`，API实际列出676个文件。
- Git已有旧版 x/y 初始化位置扰动：`libero_object_temp_x0.1...x0.5`、`y0.1...y0.5`，每档10份BDDL及10份`.pruned_init`，共100对。全部已逐文件读取、比较并计算SHA256。
- HF当前文件清单包含四个Suite的`*_swap`及新的七类扰动，但**不含**上述`temp_x/temp_y`目录，不能把HF revision当作Git x/y实验的资产版本。实际下载一份Spatial swap BDDL、对应init，以及新`07_initial_pose_position_angle` BDDL和索引。新BDDL、索引、manifest、静态验证JSON共4份可对照上游`SHA256SUMS.txt`，四份全部匹配；旧swap文件不在该校验清单中，另记录实际SHA256。

本机证据：`position_static_audit.json`、`hf_repository.json`、`hf_sample_validation.json`、`xml_asset_comparison.json`，位于本报告同目录。下载数据仅为审查样本，不复制到公开仓库。

## 2. 三种不同的“位置扰动”不能混用

| 形式 | 实际含义 | 固定资产与兼容性 |
|---|---|---|
| `*_swap` / `use_swap` | 交换BDDL的`On object region`关系，可能移动目标、容器或干扰物 | HF有BDDL及init；Git注册独立Suite |
| `libero_object_temp_x/y...` | 旧论文位置实验，修改采样区域并附带独立生成的历史init | Git有100对；没有同名正式Suite，README要求复制到`libero_object_temp` |
| HF `07_initial_pose_position_angle` | BDDL增加`:perturbation_config`，请求XY位置及yaw变化 | 当前Git解析器**实际忽略该段**，不能直接运行后宣称扰动生效 |

`SwapPerturbator`只替换匹配的初始`On`语句，使用Python随机选择：[perturbation.py:50–127](https://github.com/Zxy-MLlab/LIBERO-PRO/blob/eafdb809426b13153aa1e4c42d6601844217dfec/perturbation.py#L50-L127)。例如Spatial0官方swap文件交换plate与cookies的初始区域，并非只平移被抓取的黑碗。

对于新HF Spatial0 BDDL，配置写明`delta_xy=(0.17,-0.06)`、yaw=`0.7853981633974483`；见[固定BDDL:135–206](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro/blob/c86fc3b8293185a6f373677018ff3e37f8391602/bddl_files/07_initial_pose_position_angle/bddl/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl#L135)。本次直接调用固定Git解析器，stdout为`:perturbation_config is not recognized in problem`，返回字典没有该配置。源码[envs/bddl_utils.py:143–167](https://github.com/Zxy-MLlab/LIBERO-PRO/blob/eafdb809426b13153aa1e4c42d6601844217dfec/libero/libero/envs/bddl_utils.py#L143-L167)确认未知段只打印后忽略。上游静态验证JSON的`failure_count=0`不能替代当前运行时支持性检查。

## 3. 旧x/y资产的真实检查结果

- 100对BDDL的language、fixtures、objects、obj_of_interest、init关系及goal段均保持不变；区域数值发生变化。
- `x0.1`不是10cm：10任务目标区分别沿X平移±0.07m；`y0.1`沿Y平移+0.07m。字母汤原区`(-0.145,-0.265,-0.095,-0.215)`变为`(-0.075,-0.265,-0.025,-0.215)`：[原始文件:15](https://github.com/Zxy-MLlab/LIBERO-PRO/blob/eafdb809426b13153aa1e4c42d6601844217dfec/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl#L15)、[x0.1文件:15](https://github.com/Zxy-MLlab/LIBERO-PRO/blob/eafdb809426b13153aa1e4c42d6601844217dfec/libero/libero/bddl_files/libero_object_temp_x0.1/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl#L15)。目录名不能直接解释为米。
- 较大档还改动其他区域：字母汤x0.2的`other_object_region_2`沿X平移**10m**：[文件:36](https://github.com/Zxy-MLlab/LIBERO-PRO/blob/eafdb809426b13153aa1e4c42d6601844217dfec/libero/libero/bddl_files/libero_object_temp_x0.2/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl#L36)。y0.5部分任务连basket区域也移动10m；x0.5 butter任务目标区宽度改变。无论其上游用意如何，都不能称单一、温和的目标位置变量实验。
- 99份扰动init为有限float64数组`(50,110)`；`libero_object_temp_y0.5/pick_up_the_cream_cheese_and_place_it_in_the_basket.pruned_init`实际为空数组`(0,)`。存在文件不代表有可评测初态。
- 相同编号也不是严格的反事实配对：字母汤x0.1的idx0除目标位置外，flattened state索引1–7以及多处其他位置也不同。50条对应样本前7个qpos坐标差的最大值约0.0646rad；idx0原始起始姿态有噪声，扰动起始姿态对应源码默认home。源码[OnTheGroundPanda.init_qpos:35–38](https://github.com/Zxy-MLlab/LIBERO-PRO/blob/eafdb809426b13153aa1e4c42d6601844217dfec/libero/libero/envs/robots/on_the_ground_panda.py#L35-L38)。这说明“相同seed/idx”不足以声称只改变目标位置。

本次检查的是文件值与初始化模型顺序，没有完成运行时joint地址、可达性、碰撞或稳定性验收；这些结论不代表所有PRO任务不可运行。

## 4. 成功判定与物理隔离

旧x/y版本保留原BDDL成功谓词，例如字母汤仍为`In alphabet_soup_1 basket_1_contain_region`。实际任务类对解析后的每个goal调用原生谓词：[libero_floor_manipulation.py:132–155](https://github.com/Zxy-MLlab/LIBERO-PRO/blob/eafdb809426b13153aa1e4c42d6601844217dfec/libero/libero/envs/problems/libero_floor_manipulation.py#L132-L155)。wrapper只转发`step`和`check_success`，[env_wrapper.py:87–104](https://github.com/Zxy-MLlab/LIBERO-PRO/blob/eafdb809426b13153aa1e4c42d6601844217dfec/libero/libero/envs/env_wrapper.py#L87-L104)。谓词逻辑仍然存在，**任务是否仍可完成、初始是否已经成功需实际reset后验证，未在本次确认**。

与EmbodiedLab固定hf-LIBERO `8561c60eea2fb93096146f240194649df73d8b1e`做字节比较：7个problem文件、2个predicate文件、3个robot文件、env_wrapper和bddl_utils完全相同。bddl_base_domain只在资产路径加载处不同（PRO使用源码目录assets，hf使用资产定位函数）；region sampler旋转类型判断不同。153份PRO XML中，与当前固定资产有相同相对路径的106份全部字节一致，其余47份为PRO新增资产。没有全面比较mesh/texture或运行时编译模型，不能据此宣称两个完整仿真环境动力学完全一致。

可完全隔离：PRO自己支持`LIBERO_CONFIG_PATH`，由独立`config.yaml`指定BDDL、init和资产路径：[libero/libero/__init__.py:5–34](https://github.com/Zxy-MLlab/LIBERO-PRO/blob/eafdb809426b13153aa1e4c42d6601844217dfec/libero/libero/__init__.py#L5-L34)。但它也占用Python模块名`libero`；应在独立进程/环境加载，不能在同一已导入hf-libero的进程切换。不要执行README将文件复制进当前安装目录的步骤，不要覆盖`.libero-gpu`。

上游requirements仍有旧Python/NumPy/Torch安装路径；RTX5090不能照抄CUDA11.3 Torch命令。环境进程可仅用CPU仿真+EGL，模型继续使用已有独立服务，不需要在PRO环境重新安装大模型；安装兼容现代MuJoCo的依赖是否足够，仍需独立验证。当前`LiberoEnv`仅允许标准Suite，Recorder/Replay又绑定标准来源身份，因此直接传`*_swap`或`*_temp`目前会被拒绝。需要一个很小的**显式外部任务来源/资产身份路径**或独立PRO wrapper，而不能伪装成标准Suite。

## 5. 最小后续研究方案

1. 先完成标准挑战集；PRO结果单独目录、单独manifest、单独分母。
2. 若启动小实验，优先只选Git版本字母汤、奶酪中的**非空x0.1或y0.1**，逐任务核对BDDL、model布局、init SHA、实际初态、原生goal及20Hz控制。也可选HF的Spatial swap任务，但应明确这是区域交换实验。不要用新07配置，除非找到并固定真正支持该字段的官方运行时代码。
3. 原始与扰动均使用同一独立PRO runtime和相同依赖；明确初态是否严格配对。直接使用官方两套init只能报告“官方原始/扰动初态集合对比”，不能作只改变目标位置的因果归因。若需严格反事实，先取得上游提供的配对状态或经审查的官方可复现生成方案，再做实验；不在评测step中修改物体状态。
4. OpenPI、OpenVLA-OFT无需改权重：接口仍是双RGB、机器人本体状态、原任务语言及七维OSC动作；分别保留各自官方图像变换、归一化和chunk。先验证真实帧与原native协议一致，再跑少量完整闭环。无需复制对方预处理。
5. 每条记录`benchmark_family=libero_pro`、扰动类型/幅度（读取实际BDDL）、源Git/HF版本、原始任务名、BDDL/init/model指纹、官方idx/seed、时限、模型身份与结果。Replay必须恢复同一PRO来源，拒绝被标准LIBERO Replay静默接受。

## 6. 已执行命令与范围

```bash
# 克隆固定官方源码；代理仅作用于本命令。
git clone --depth 1 https://github.com/Zxy-MLlab/LIBERO-PRO.git work/phase6b/libero_pro/source
# 实际输出 HEAD=eafdb809426b13153aa1e4c42d6601844217dfec

.venv-libero/bin/python \
  work/phase6b/libero_pro/audit_position_files.py
```

另实际使用固定HF revision下载上述少量样本，调用固定Git parser解析新07 BDDL，比较XML与源码并输出JSON；原始HDF5、权重、虚拟环境、标准LIBERO源码均未写入。没有PRO成功率、没有PRO视频，也没有将静态检查当成真实仿真验证。
