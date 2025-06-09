[English](https://github.com/Sunrepe/AoPSFT)/中文

---

## AoPSFT 介绍 🚀

![AoPSFT](https://github.com/Sunrepe/AoPSFT/blob/main/assets/aopsft.png)

### 1. 思想创新 💡

* 受到论文 [rrde](https://ieeexplore.ieee.org/abstract/document/10650094) 启发，我们在大语言模型（LLM）推理之前引入了规划（plan）机制，充分发挥LLM在长序列任务中的规划能力。

* 借鉴论文 [eto](https://arxiv.org/abs/2403.02502) 的思路，实现了步骤级别的 Chain-of-Thought（CoT）推理，大幅提升了Agent的可解释性、正确性与推理效率。

### 2. 算法创新 ⚙️

* 本项目支持多轮交互（multi-turn）级别的LLM微调训练，可在整个trajectory（行为轨迹）上进行算法优化。

* 针对单个episode中的不同step，赋予了不同的重要性权重，从而实现了重要性加权的损失优化机制。

* 特别感谢 [openrlhf](https://github.com/OpenRLHF/OpenRLHF)，本项目代码基于其代码库进行修改与扩展。

---

## 环境配置 🛠️

### 1. Python 环境 🐍

请参考 [openrlhf](https://github.com/OpenRLHF/OpenRLHF) 的说明完成基础环境配置。

### 2. 游戏环境 🎮

通过以下命令安装 ScienceWorld 游戏环境：

```bash
pip install scienceworld
```

### 3. LLM 微调训练（SFT）🧠

运行以下脚本开始模型训练：

```bash
bash training/scripts/train_mtsft.sh
```

确保 Python 环境配置无误后，即可启动训练流程。

### 4. 运行 ScienceWorld 游戏 🌍

* 使用 vLLM 启动模型推理 API：

  ```bash
  MODEL_PATH="checkpoint/meta/Llama-3.1-8B-Instruct/aopsft/"
  API_KEY="token-abc123"
  MODEL_NAME="llama"    
  PORT="8003"

  vllm serve $MODEL_PATH \
  --dtype auto \
  --served-model-name $MODEL_NAME \
  --port $PORT \
  --api-key $API_KEY
  ```

* 启动游戏并使用 LLM 进行推理：

  ```bash
  python gamerun/sft_run.py $MODEL_NAME $PORT
  ```
