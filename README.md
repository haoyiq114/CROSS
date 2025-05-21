# 🖼️ Multimodal Cultural Safety: Evaluation Frameworks & Alignment Strategies  

<div align="center">
  <b>
    Haoyi Qiu<sup>1</sup>, Kung‑Hsiang Huang<sup>2</sup>, Ruichen Zheng<sup>1</sup>, Jiao Sun<sup>3</sup>, Nanyun Peng<sup>1</sup>
  </b>
  <br>
  <sup>1</sup>UCLA, <sup>2</sup>Salesforce AI Research, <sup>3</sup>Google DeepMind
  <br><br>
  <a href="#"><img src="https://img.shields.io/badge/Paper-arXiv-orange"></a>
</div>


## Contents
1. [CROSS Evaluation Benchmark](#cross-evaluation-benchmark)
2. [CROSS‑Eval Metrics](#cross‑eval-metrics)
3. [Quick‑Start](#quick‑start-openai-models)
4. [Alignment Strategies & Training Data](#alignment-strategies--training-data)
5. [Citation](#citation)
6. [Disclaimer](#disclaimer)


## 🧩 CROSS Evaluation Benchmark  

```
data/
├── casa/
│   ├── english.json
│   └── multilingual.json
└── safeworld/
    ├── english.json
    └── multilingual.json
```

Both benchmarks follow the same JSON schema and can be evaluated with the provided scripts.


## ⚖️ CROSS‑Eval Metrics  

CROSS‑Eval reports four dimensions of culturally safe reasoning: 

- **Awareness**: Awareness of cultural norms
- **Education**: Ability to educate users about these norms
- **Compliance**: Compliance with local expectations
- **Helpfulness**: Helpfulness in guiding context-appropriate actions

Metric implementations are in `code/evaluation_.py`.


## 🔧 Quick‑Start

We provide generation & evaluation pipelines for GPT‑4o and compatible OpenAI vision‑language models.

```bash
# 🔹 Direct responses
bash scripts/run.sh

# 🔹 Responses with explicit reasoning
bash scripts/run_reasoning.sh
```

For **non‑OpenAI models** (e.g., InternVL 2.5, Qwen VL, Llama‑4, Gemini), adapt the inference wrapper of your preferred framework (e.g., vllm) to match the input/output format used in our scripts.


## 🧨 Alignment Strategies & Training Data  

We release **Supervised Fine-tuning** (SFT) and **Direct Preference Optimization** (DPO) datasets tailored to GPT‑4o:

```
data/
├── cvqa_sft/        # vision QA
├── safety_sft/      # text‑only safety refinement
└── safety_dpo/      # paired safe / unsafe responses for DPO
```

| Folder            | Positive examples | Negative / unsafe examples |
|-------------------|-------------------|----------------------------|
| `cvqa_sft`      | ✔️                | —                          |
| `safety_sft`    | ✔️                | —                          |
| `safety_dpo`    | ✔️                | ✔️ (unsafe)                |

Training scripts for GPT‑4o are in `code/gpt4o_tuning.ipynb`.  
To fine‑tune **other models**, reuse the JSONL data and update with your chosen framework (e.g., vllm).


## Citation  

If you use CROSS, CROSS‑Eval, or our alignment data, please cite:

```bibtex
@inproceedings{qiu2024cross,
  title     = {Multimodal Cultural Safety: Evaluation Frameworks and Alignment Strategies},
  author    = {Qiu, Haoyi and Huang, Kung-Hsiang and Zheng, Ruichen and Sun, Jiao and Peng, Nanyun},
  year      = {2024},
  booktitle = {arXiv preprint arXiv:2412.06483}
}
```


## ⚠️ Disclaimer  

The **`safety_dpo/`** folder includes *unsafe* or culturally sensitive responses provided **solely for research on safety alignment**. Handle these files with care. Do **not** redistribute or deploy them in production systems.
