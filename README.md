English/[中文](https://github.com/Sunrepe/AoPSFT/assets/readme_zh.md)

------

## Introduction to AoPSFT 🚀

![AoPSFT](https://github.com/Sunrepe/AoPSFT/assets/aopsft.png)

1. **Innovative Concepts** 💡

- Inspired by the paper [rrde](https://ieeexplore.ieee.org/abstract/document/10650094), AoPSFT introduces planning before LLM inference to fully leverage the long-sequence planning capability of LLMs.
- Inspired by the paper [eto](https://arxiv.org/abs/2403.02502), AoPSFT implements step-level Chain-of-Thought (CoT) reasoning, enhancing the interpretability, correctness, and efficiency of agent reasoning.

1. **Algorithmic Innovations** ⚙️

- AoPSFT implements multi-turn level LLM fine-tuning, enabling algorithmic optimization over entire trajectories.
- It assigns different importance weights to different steps within a single episode, enabling importance-weighted loss optimization.
- Thanks to [openrlhf](https://github.com/OpenRLHF/OpenRLHF), our code is based on modifications to their codebase.

------

## Setup 🛠️

### 1. Python Environment 🐍

Please refer to [openrlhf](https://github.com/OpenRLHF/OpenRLHF) for the basic environment setup.

### 2. Game Environment 🎮

Install the environment via:

```bash
pip install scienceworld
```

### 3. LLM SFT (Supervised Fine-Tuning) 🧠

```bash
bash training/scripts/train_mtsft.sh
```

Once the Python environment is set up, you can run the above script to start model training.

### 4. Run the ScienceWorld Game 🌍

- Start the API for inference using vLLM:

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

- Launch the game and use the LLM for reasoning:

  ```bash
  python gamerun/sft_run.py $MODEL_NAME $PORT
  ```

