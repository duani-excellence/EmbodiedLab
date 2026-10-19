# OpenVLA-OFT 固定版本与协议审计

审计时间：2026-10-16。本文保留实验前源码和元数据审计；后续真实加载、协议对照及120条闭环结果见[Phase6B报告](../archive/PHASE6B_BENCHMARK_REPORT.md)。

## 1. 固定来源与下载身份

- 官方仓库：<https://github.com/moojink/openvla-oft>，固定提交 `e4287e94541f459edc4feabc4e181f537cd569a8`（2025-09-08，Pin diffusers version）。本机忽略目录 `.deps/openvla-oft-e4287e94541f459edc4feabc4e181f537cd569a8/`，未修改源码。
- 官方 Transformers fork：<https://github.com/moojink/transformers-openvla-oft>，固定提交 `bc339d9ad707454c0c115970db43c260067c61ab`（4.40.1；2025-02-25，Full bidirectional attention）。不能以普通 HF Transformers 4.40.1 替代。
- 官方 dlimp fork：<https://github.com/moojink/dlimp_openvla>，固定提交 `040105d256bd28866cc6620621a3d5f7b6b91b46`。
- Checkpoint：`moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`，HF Revision **`638918f3d1c2e43a39a8a20772bdb8b91835e4b7`**。
- `openvla_hf_metadata.json` 保存官方 API 的每文件大小、Git blob ID、大文件 LFS SHA256。完整仓库 25 个对象，15,939,168,050 bytes；四份已合并模型 safetensors 共 15,082,600,832 bytes，action_head 302,242,674、proprio_projector 67,275,256 bytes。LoRA adapter 484,458,600 bytes 不用于已合并模型推理，无需另行合并。
- 所有下载使用上述固定 Revision。大文件按 LFS SHA256 验证；小文件可按 Git blob ID 或本地 SHA256 固定。仅完成校验后才能宣称官方权重验证通过。

## 2. 源码证据（行号均对应固定提交）

代码链接前缀：`https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/`。

| 项目 | 官方实现 | 实际语义 |
|---|---|---|
| 模型配置 | `experiments/robot/libero/run_libero_eval.py:81–128` | L1 regression=True、diffusion=False、FiLM=False、2相机、proprio=True、center_crop=True、open_loop_steps=8、无量化、原始渲染256、settling10 |
| 时限 | 同文件 `62–69` | Spatial220、Object280、Goal300、LIBERO-10 520、90 400；settling不算策略步 |
| 任务身份 | 同文件 `179–192` | 按实际 Suite 选择归一化 key，若必要加 `_no_noops`；不能把Spatial统计用于10 |
| 原始图像 | `experiments/robot/libero/libero_utils.py:33–44` | agentview 与 robot0_eye_in_hand 的 HWC RGB 各 `[::-1, ::-1]`，180°一次 |
| 观测状态 | `run_libero_eval.py:243–262`；`libero_utils.py:63–87` | `[eef_world_xyz_m(3), quat(xyzw)→axisangle_rad(3), gripper_qpos_m(2)]`，共8维 |
| resize | `experiments/robot/openvla_utils.py:520–543` | TF encode_jpeg→decode_image uint8→Lanczos3 antialias resize224→round/clip→uint8；非PIL路径 |
| center crop | 同文件 `546–626,682–712` | 先224，再90%**面积**的居中crop（边长sqrt(.9)），TF crop_and_resize到224，dtype转换；不是随意90%边长或简单整数裁剪 |
| 输入视觉骨干 | `prismatic/extern/hf/processing_prismatic.py:32–145` | 每相机经过Checkpoint官方Processor，DINO+SigLIP两组normalization，拼6通道；两相机再拼12通道，网络自己解包 |
| prompt | `openvla_utils.py:743–770` | `In: What action should the robot take to {task_label.lower()}?\nOut:`；agentview先、wrist后 |
| proprio normalize | 同文件 `649–679,772–778` | 按该Suite官方 q01/q99，公式 `2*(x-low)/(high-low+1e-8)-1`，mask后clip[-1,1]；函数会写入传入obs['state']，每次必须用新dict避免二次归一化 |
| 模型解码 | 同文件 `784–798`；`prismatic/extern/hf/modeling_prismatic.py:888–940` | inference_mode，do_sample=False，连续L1 action head，一次8×7，不是token自回归，不是diffusion |
| action inverse | `modeling_prismatic.py:772–791` | 前6维按Suite q01/q99逆归一化；action mask第7=False，保留gripper原始[0,1]语义；没有输出clamp/tanh |
| gripper环境变换 | `run_libero_eval.py:265–275`；`robot_utils.py:149–199` | 官方`normalize_gripper_action(binarize=True)`再`invert_gripper_action`，即`-sign(2*g-1)`；-1=open，+1=close。只有该官方变换可用，不加启发式规则 |
| chunk执行 | `run_libero_eval.py:278–354`；`prismatic/vla/constants.py:25–31` | 缓存8动作，执行完整8再查询（最后成功可提前结束），每步实际native OSC控制；不套用OpenPI前5/10逻辑 |
| 初态/seed | `libero_utils.py:18–25`；`run_libero_eval.py:292–320`；`robot_utils.py:38–51` | native reset→set_init_state→10 dummy(open grip)；官方helper环境seed0、全局模型seed默认7；统一Benchmark明确使用相同固定环境seed时需记录与上游默认差异，不能声称逐位官方评测复现 |

## 3. 精度与注意力兼容性

官方 `get_vla`（openvla_utils.py:253–308）使用 `torch.bfloat16`、eval、no quantization、low_cpu_mem_usage，将模型放到GPU。`attn_implementation='flash_attention_2'` 这一行当前已注释（284）。SETUP.md:19 指出 FlashAttention2 用于training，推理没有必须安装2.5.5的理由。

必须使用官方 Transformers fork 中 `src/transformers/models/llama/modeling_llama.py:672–731` 的 **LlamaSdpaAttention**：复制旧causal mask最后一行，保留padding屏蔽，同时`scaled_dot_product_attention(...,is_causal=False)`。标准HF版或强制eager可能恢复错误因果语义。加载时应打印/断言真实 language_model 的attention实现是sdpa、fork SHA正确；不要通过换模型或注意力语义绕过错误。

官方文章评测环境为 Python3.10.14、Torch2.2.0、custom Transformers4.40.1、A100（LIBERO.md:67–71附近，以实际文件为准）。当前RTX5090 sm_120不能使用旧Torch2.2 wheel的原生Blackwell kernels；独立 `.venv-openvla` 选择已在项目验证的 Torch2.9.1+cu128 / torchvision0.24.1+cu128，保留fork与其余协议，属于显式软件兼容差异。不得把当前GPU结果当作论文数值复现。

只安装 TensorFlow CPU2.15.0 以执行官方resize/crop，避免TF争占GPU。官方import链会导入训练/数据模块，因此有些TFDS/dlimp/TF Graphics依赖在纯推理时也会因import而必要；无需训练、RLDS下载、CUDA Toolkit或FlashAttention编译。需进行真实import检查与准确freeze，不能只凭依赖解析成功。

## 4. Checkpoint只读保护方案

官方 local `get_vla` 在 `openvla_utils.py:270–279` 调用 `update_auto_map` 和 `check_model_logic_mismatch`：会备份/改写config并将当前仓库的modeling/configuration.py复制进本地路径（56–198）。不能直接把已校验原始下载目录传入这个修改路径。

最小做法：创建单独运行视图；小JSON/Python/tokenizer配置复制为普通文件，大型safetensors和.pt仅symlink到已校验对象（不能硬链接可写配置）。记录视图及原件各自hash，运行前后再次校验原件。官方loader只可同步运行视图中的小代码与config。当前HF模型modeling SHA `f40ee7883e16aab1a2d89b6e8f31cc81f6b8055120b1fefe169e05c7031098fa` 与源码SHA `b5431a074c0025a12e46dc954a5e18d1d73477babb5ae42e3a12ab4b907f33a6` 的唯一差异是未使用diffusion分支的两行scheduler初始化；本次连续L1路径无差异。configuration完全相同SHA `68cc5ae34f1b46af3168d8d479cb81bb776965653453fd904aa8eefb6c8f9f68`。

如使用自定义只读loader，应逐项证明等价官方get_vla/get_processor/get_action_head/get_proprio_projector，而不是静默跳过官方初始化。优先运行视图方式能更直接复用官方实现。

## 5. Adapter审查要点

- 模型进程与LIBERO环境进程隔离，OpenPI、OpenVLA大模型顺序运行。
- 线上传输原始256 HWC RGB、原始8维state、完整task language及suite；模型进程调用官方图像/state处理，夹爪转换调用官方函数。
- 原始8×7 chunk和实际传给环境action都记录；未执行的尾部不能算动作步。
- 网络输出可能超[-1,1]：官方L1 head无tanh/clamp，不能先验假定。若真遇到边界，需单独记录并与native clipping语义核对，不加新幅度缩放。
- eval/no diffusion/do_sampleFalse 下模型前向通常确定，但仍记录随机种子、请求顺序、cudnn设置和Torch版本；不能假装与OpenPI随机流相同。
- 相机、state、prompt、action应至少有一次真实观测/真实模型chunk逐项对比；unit mock不能当作模型推理证据。
- 最小实际推理需要base merged权重+action head+proprio projector+官方processor/tokenizer+stats，无需LoRA重新合并、量化或训练。
