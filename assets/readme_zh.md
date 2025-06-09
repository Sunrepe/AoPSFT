[English](https://github.com/Sunrepe/AoPSFT)/中文

## AoPSFT介绍:

![AoPSFT](https://github.com/Sunrepe/AoPSFT/assets/aopsft.png)

1, 思想创新

- 受到论文[rrde](https://ieeexplore.ieee.org/abstract/document/10650094)启发,在LLM推理之前先进行plan, 充分利用LLM本身的长序列规划能力. 

- 受到论文[eto](https://arxiv.org/abs/2403.02502)启发, 实现了步骤级别的COT 推理, 增强了Agent推理可解释性,正确性与效率.

2, 算法创新

- 本文实现了multi-turn级别的LLM微调训练, 也就是可以在trajectory进行算法优化.
- 文本对单条episode中的不同step赋予了不同的重要性, 实现了具有重要性加权的损失优化.
- 感谢[openrlhf](openrlhf/openrlhf at main · OpenRLHF/OpenRLHF), 我们的代码在其代码库上修改后实现.



## Setup

### 1, Python env

请参考[openrlhf](openrlhf/openrlhf at main · OpenRLHF/OpenRLHF)配置基本环境

### 2, Game env

通过`pip install scienceworld`安装环境

### 3, LLM SFT

```bash
bash training/scripts/train_mtsft.sh
```

在配置好Python解释器之后, 可以运行上述代码进行模型训练.

### 4, Run ScienceWorld Game

- 利用vllm 启动api 推理

  ```bash
  MODEL_PATH="checkpoint/meta/Llama-3.1-8B-Instruct/aopsft/"
  API_KEY="token-abc123"
  MODEL_NAME="llama"    
  PORT="8003"
  
  vllm serve $MODEL_PATH \
  --dtype auto \
  --served-model-name $MODEL_NAME \
  --port $PORT \
  --api-key $API_KEY \
  ```

- 启动游戏并利用LLM进行推理

  ```bash
  python gamerun/sft_run.py $MODEL_NAME $PORT
  ```