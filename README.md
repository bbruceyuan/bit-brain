<p align="center">
<h1 align="center">《🧠 BitBrain - Train Your Own tiny-LLM》</h1>
<h4 align="right">迈出自己训练 LLM 的关键一步～ Let's just take action!</h4>
</p>

# 🧠BitBrain
`Bitbrain-0.6B-base` 是一个基于 `Qwen3-0.6B` 架构的语言模型。我们在大约 113B tokens 的高质量中英文数据上对其进行了预训练。
> 最少使用 3090 即可训练自己的比特大脑🧠（进行中）. Train your own BitBrain with just an RTX 3090 minimum.(Stay tuned)

## 环境安装
- step0: 使用 python3.12 （可选）
- step1: 首先安装 uv。见[链接](https://docs.astral.sh/uv/getting-started/installation/)
- step2: `uv sync`

## 推理
- 方式 1：使用 modelscope 运行，`uv run example/modelscope/run_bitbrain_base.py`
```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "hh2395959141/Bitbrain-0.6b-base"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "你好，请用一句话介绍一下你自己。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```
- 方式 2：使用 transformers 运行，`uv run example/transformers/run_bitbrain_base.py`
- 样例：
```
输入: 你好，请用一句话介绍一下你自己。
输出：你好，请用一句话介绍一下你自己。我是一个来自中国的程序员，拥有超过十年的编程经验。我的主要技能包括Java、Python和C++。请用简短的一句话来概括你的主要工作内容和技能特点。我的工作风格是逻辑严谨、追求卓越（省略....）
```

## 模型详情 (Model Details)

*   **基础架构 (Base Architecture)**: `Qwen3-0.6B`
*   **模型参数调整 (Parameter Adjustments)**:
    *   `rope_theta`: 我们将 `rope_theta` 参数从 `Qwen3-0.6B` 原始的 1M (1,000,000) 调整为 10,000。
    *   `max_position_embeddings`: 将上下文长度设置为 4096。
*   **训练序列长度 (Training Sequence Length)**: 在预训练阶段，为了提升训练效率，主要对长度为 2048 的样本进行了训练。


## 训练数据 (Training Data)

*   **数据集 (Dataset)**: `chinese-fineweb-edu-v2`
*   **数据子集 (Data Subset)**: 使用了该数据集中约 27% 的数据进行训练。
*   **中英文比例**：参考原数据集中的3：1比例

## 训练设置 (Training Setup)

为了尽可能实现小模型的高效训练，我们采用了以下技术和配置：

*   **分布式训练 (Distributed Training)**: 使用 `DDP` (Distributed Data Parallel) 在多个 GPU 上进行并行训练。
*   **混合精度 (Mixed Precision)**: 开启 `bf16` 混合精度训练。
*   **优化器 (Optimizer)**: 使用 `Muon` 优化器，进一步减少了优化器状态的显存占用
*   **性能加速 (Performance Acceleration)**:
    *   使用 `torch.compile` 对模型进行编译优化加速。
    *   使用 `liger kernel` 进行关键运算加速。

## 性能 (Performance)

通过上述优化，我们实现了高效的训练。我们使用模型浮点运算利用率 (Model Flops Utilization, MFU) 来衡量硬件的有效利用程度。MFU 的计算方式参考了 `deepseek-LLM` 论文中提出的**非嵌入FLOPs**方法，具体公式如下：

$$
MFU = \frac{\text{有效计算量}}{\text{训练时间} \times \text{理论算力}} = \frac{6ND(1+\frac{s}{12h}+\frac{V}{16Lh})}{\text{训练时间} \times \text{理论算力}}
$$

其中 `N` 为模型参数量, `D` 为处理的 token 总数, `s` 为序列长度, `h` 为隐藏层维度, `L` 为模型层数, `V` 为词表大小。

基于该标准，我们在不同硬件上的性能表现为：

*   **在 4 * H800 GPU 上**: MFU 达到了 **46%**。
*   **在 8 * 4090 GPU 上**: MFU 达到了 **34%**。

## 评测

我们使用 [OpenCompass](https://opencompass.org.cn/) 框架对 `Bitbrain-0.6B-base` 模型进行了评测。

我们主要在 [C-Eval](https://ceval.ai/) 这个权威的中文评测基准上，通过两种不同的方式测试了模型的性能：
- **PPL (Perplexity) 评测**: 这种方式通过计算模型对于每个选项的困惑度来判断答案，更侧重于衡量模型本身对知识的掌握程度，适合基础模型（Base Model）。
- **GEN (Generative) 评测**: 这种方式要求模型直接生成答案选项（如 "A", "B", "C"），更侧重于衡量模型遵循指令和生成特定格式的能力。通常指令微调（Instruction-tuned）后的模型会在此模式下表现更好。

**C-Eval 平均分**

| 评测方式 (Method) | 平均分 (Average Score) |
| :--- | :--- |
| **C-Eval-PPL** | **27.99%** |
| **C-Eval-GEN** | **20.30%** |

<details>
<summary>点击查看 C-Eval-PPL 各科详细得分 (平均分: 27.99%)</summary>

| 数据集 (Dataset) | 准确率 (Accuracy) |
| :--- | :--- |
| ceval-computer_network | 52.63 |
| ceval-operating_system | 10.53 |
| ceval-computer_architecture | 14.29 |
| ceval-college_programming | 18.92 |
| ceval-college_physics | 47.37 |
| ceval-college_chemistry | 25.00 |
| ceval-advanced_mathematics | 10.53 |
| ceval-probability_and_statistics | 44.44 |
| ceval-discrete_mathematics | 12.50 |
| ceval-electrical_engineer | 21.62 |
| ceval-metrology_engineer | 33.33 |
| ceval-high_school_mathematics | 44.44 |
| ceval-high_school_physics | 15.79 |
| ceval-high_school_chemistry | 21.05 |
| ceval-high_school_biology | 26.32 |
| ceval-middle_school_mathematics | 26.32 |
| ceval-middle_school_biology | 28.57 |
| ceval-middle_school_physics | 36.84 |
| ceval-middle_school_chemistry | 20.00 |
| ceval-veterinary_medicine | 26.09 |
| ceval-college_economics | 27.27 |
| ceval-business_administration | 27.27 |
| ceval-marxism | 36.84 |
| ceval-mao_zedong_thought | 33.33 |
| ceval-education_science | 10.34 |
| ceval-teacher_qualification | 22.73 |
| ceval-high_school_politics | 63.16 |
| ceval-high_school_geography | 36.84 |
| ceval-middle_school_politics | 28.57 |
| ceval-middle_school_geography | 33.33 |
| ceval-modern_chinese_history | 21.74 |
| ceval-ideological_and_moral_cultivation | 31.58 |
| ceval-logic | 31.82 |
| ceval-law | 20.83 |
| ceval-chinese_language_and_literature | 17.39 |
| ceval-art_studies | 12.12 |
| ceval-professional_tour_guide | 13.79 |
| ceval-legal_professional | 43.48 |
| ceval-high_school_chinese | 15.79 |
| ceval-high_school_history | 15.00 |
| ceval-middle_school_history | 22.73 |
| ceval-civil_servant | 25.53 |
| ceval-sports_science | 42.11 |
| ceval-plant_protection | 22.73 |
| ceval-basic_medicine | 42.11 |
| ceval-clinical_medicine | 40.91 |
| ceval-urban_and_rural_planner | 26.09 |
| ceval-accountant | 32.65 |
| ceval-fire_engineer | 35.48 |
| ceval-environmental_impact_assessment_engineer | 22.58 |
| ceval-tax_accountant | 36.73 |
| ceval-physician | 26.53 |

</details>

<details>
<summary>点击查看 C-Eval-GEN 各科详细得分 (平均分: 20.30%)</summary>

| 数据集 (Dataset) | 准确率 (Accuracy) |
| :--- | :--- |
| ceval-computer_network | 0.00 |
| ceval-operating_system | 5.26 |
| ceval-computer_architecture | 19.05 |
| ceval-college_programming | 18.92 |
| ceval-college_physics | 5.26 |
| ceval-college_chemistry | 16.67 |
| ceval-advanced_mathematics | 31.58 |
| ceval-probability_and_statistics | 0.00 |
| ceval-discrete_mathematics | 0.00 |
| ceval-electrical_engineer | 16.22 |
| ceval-metrology_engineer | 16.67 |
| ceval-high_school_mathematics | 22.22 |
| ceval-high_school_physics | 26.32 |
| ceval-high_school_chemistry | 15.79 |
| ceval-high_school_biology | 36.84 |
| ceval-middle_school_mathematics | 10.53 |
| ceval-middle_school_biology | 19.05 |
| ceval-middle_school_physics | 21.05 |
| ceval-middle_school_chemistry | 15.00 |
| ceval-veterinary_medicine | 26.09 |
| ceval-college_economics | 10.91 |
| ceval-business_administration | 18.18 |
| ceval-marxism | 21.05 |
| ceval-mao_zedong_thought | 29.17 |
| ceval-education_science | 24.14 |
| ceval-teacher_qualification | 27.27 |
| ceval-high_school_politics | 0.00 |
| ceval-high_school_geography | 21.05 |
| ceval-middle_school_politics | 23.81 |
| ceval-middle_school_geography | 8.33 |
| ceval-modern_chinese_history | 17.39 |
| ceval-ideological_and_moral_cultivation | 21.05 |
| ceval-logic | 13.64 |
| ceval-law | 20.83 |
| ceval-chinese_language_and_literature | 21.74 |
| ceval-art_studies | 39.39 |
| ceval-professional_tour_guide | 31.03 |
| ceval-legal_professional | 4.35 |
| ceval-high_school_chinese | 26.32 |
| ceval-high_school_history | 30.00 |
| ceval-middle_school_history | 18.18 |
| ceval-civil_servant | 10.64 |
| ceval-sports_science | 10.53 |
| ceval-plant_protection | 31.82 |
| ceval-basic_medicine | 0.00 |
| ceval-clinical_medicine | 18.18 |
| ceval-urban_and_rural_planner | 19.57 |
| ceval-accountant | 22.45 |
| ceval-fire_engineer | 25.81 |
| ceval-environmental_impact_assessment_engineer | 9.68 |
| ceval-tax_accountant | 32.65 |
| ceval-physician | 22.45 |

</details>

## 愿景 (Purpose)

该模型在一个相对纯净且可追溯的数据集上进行预训练，在为后续的指令微调、领域适配等后训练（Post-training）中可以更方便地分析和追溯模型行为的来源。

## 备注
本项目还在施工中，项目结构还会重构~


> 最后欢迎大家使用 [AIStackDC](https://aistackdc.com/phone-register?invite_code=D872A9) 算力平台，主打一个便宜方便（有专门的客服支持），如果你需要的话可以使用我的邀请链接: [https://aistackdc.com/phone-register?invite_code=D872A9](https://aistackdc.com/phone-register?invite_code=D872A9)
