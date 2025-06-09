from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import exist_and_not_none, zero_pad_sequences

class DPOMultiturnDataset(Dataset):
    """
    专门处理 DPO 数据集的多轮对话数据，数据中每条样本包含3组 messages:
      - history
      - chosen
      - rejected

    处理流程：
      1. 分别将 history 与 chosen 拼接，和 history 与 rejected 拼接，生成两段文本；
      2. 仅为 chosen 与 rejected 部分中的 assistant 回答计算 token 范围，作为 response_ranges；
      3. process 每条数据返回：
           {
             "chosen": chosen_text, 
             "reject": reject_text, 
             "info": {"chosen_response_ranges": chosen_response_ranges,
                      "reject_response_ranges": reject_response_ranges}
           }
      4. __getitem__ 返回：
           (
              chosen_token["input_ids"],
              chosen_token["attention_mask"],
              reject_token["input_ids"],
              reject_token["attention_mask"],
              info,  # 包含 chosen_text, reject_text, chosen_response_ranges, reject_response_ranges
           )
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=8,
        multiple_of=1,
        multiturn=True,
    ) -> None:
        # 固定多轮对话
        assert multiturn, "只能处理多轮对话数据"
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiple_of = multiple_of
        self.multiturn = multiturn

        self.input_template = input_template
        # 此处假定数据中的对话信息分别存储在 "history", "chosen", "rejected" 字段中
        # 可根据实际情况进行调整
        self.input_key_history = getattr(self.strategy.args, "input_key_history", "history")
        self.input_key_chosen = getattr(self.strategy.args, "input_key_chosen", "chosen")
        self.input_key_reject = getattr(self.strategy.args, "input_key_reject", "rejected")
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # 并行预处理所有样本
        processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )
        # 过滤处理失败的样本
        processed_dataset = processed_dataset.filter(lambda x: x["chosen"] is not None and x["reject"] is not None)
        self.chosen_texts = processed_dataset["chosen"]
        self.reject_texts = processed_dataset["reject"]
        self.infos = processed_dataset["info"]
        # 可选：展示一个样本
        self.print_sample()

    def preprocess_dpo_messages(self, history, responses, max_length, tokenize_fn):
        """
        拼接 history 与 responses，且仅为 responses 部分中的 assistant 回答计算 token 范围。
        tokenize_fn: 用于 tokenizer 编码的方法（apply_chat_template）
        返回： (full_text, response_ranges)
        """
        messages = history + responses
        response_ranges = []
        # 仅处理 responses 部分，从 index = len(history) 开始
        for idx in range(len(history), len(messages)):
            message = messages[idx]
            if message["role"] == "assistant":
                # prompt_part 为 assistant 回答之前的内容（包含 history 与 responses 中 assistant 回答之前的部分）
                prompt_part = tokenize_fn(messages[:idx], tokenize=False, add_generation_prompt=True)
                full_part = tokenize_fn(messages[: idx + 1], tokenize=False)
                # assistant 部分为两者差值
                assistant_part = full_part[len(prompt_part):]
                start_idx = self.tokenizer(
                    prompt_part,
                    max_length=max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["attention_mask"].int().sum().item()
                assistant_token_count = self.tokenizer(
                    assistant_part,
                    max_length=max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["attention_mask"].int().sum().item()
                end_idx = start_idx + assistant_token_count
                response_ranges.append((start_idx, end_idx))
        final_text = tokenize_fn(messages, tokenize=False)
        return final_text, response_ranges

    def process_data(self, data):
        """
        data 中包含:
           data[self.input_key_history]: history 列表
           data[self.input_key_chosen]: chosen 列表
           data[self.input_key_reject]: rejected 列表
        返回:
           {
             "chosen": chosen_text,
             "reject": reject_text,
             "info": {"chosen_response_ranges": chosen_response_ranges,
                      "reject_response_ranges": reject_response_ranges}
           }
        """
        history = data[self.input_key_history]
        chosen_msgs = data[self.input_key_chosen]
        reject_msgs = data[self.input_key_reject]

        # 拼接 history+chosen，并计算 chosen 部分 assistant 回答的 token 范围
        chosen_text, chosen_response_ranges = self.preprocess_dpo_messages(
            history, chosen_msgs, self.max_length, self.apply_chat_template
        )
        # 拼接 history+rejected，并计算 rejected 部分 assistant 回答的 token 范围
        reject_text, reject_response_ranges = self.preprocess_dpo_messages(
            history, reject_msgs, self.max_length, self.apply_chat_template
        )

        # 对文本进行 token 编码，判断长度是否超限
        chosen_token = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        chosen_len = chosen_token["attention_mask"].int().sum().item()
        reject_token = self.tokenizer(
            reject_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        reject_len = reject_token["attention_mask"].int().sum().item()

        if (not chosen_text) or (chosen_len >= self.max_length - 2) or (not reject_text) or (reject_len >= self.max_length - 2):
            print(f"to long{'*'*100}\n{chosen_text}\n\n{reject_text}")
            chosen_text, reject_text = None, None
            chosen_response_ranges, reject_response_ranges = None, None

        return {
            "chosen": chosen_text, 
            "reject": reject_text, 
            "info": {
                "chosen_response_ranges": chosen_response_ranges,
                "reject_response_ranges": reject_response_ranges
            }
        }

    def __len__(self):
        return len(self.chosen_texts)

    def __getitem__(self, idx):
        chosen_text = self.chosen_texts[idx]
        reject_text = self.reject_texts[idx]
        infos = self.infos[idx]
        chosen_ranges = infos["chosen_response_ranges"]
        reject_ranges = infos["reject_response_ranges"]

        # 确保文本以 eos_token 结尾
        if not chosen_text.rstrip("\n").endswith(self.tokenizer.eos_token):
            chosen_text = chosen_text.rstrip("\n") + self.tokenizer.eos_token
        if not reject_text.rstrip("\n").endswith(self.tokenizer.eos_token):
            reject_text = reject_text.rstrip("\n") + self.tokenizer.eos_token

        # 分别编码 chosen 与 reject 文本（仅调用一次 tokenizer）
        chosen_token = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        reject_token = self.tokenizer(
            reject_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        chosen_input_ids = chosen_token["input_ids"][0]
        chosen_attention_mask = chosen_token["attention_mask"][0]
        reject_input_ids = reject_token["input_ids"][0]
        reject_attention_mask = reject_token["attention_mask"][0]

        # info 中保留文本和 response_ranges，labels 可在后续阶段根据 response_ranges 构建
        info = {
            "chosen_text": chosen_text,
            "reject_text": reject_text,
            "chosen_response_ranges": chosen_ranges,
            "reject_response_ranges": reject_ranges,
        }
        return (
            chosen_input_ids,
            chosen_attention_mask,
            reject_input_ids,
            reject_attention_mask,
            info,
        )

    def collate_fn(self, batch):
        # batch 为 list，每项为 __getitem__ 返回的元组
        chosen_ids_list = []
        chosen_masks_list = []
        reject_ids_list = []
        reject_masks_list = []
        infos = {"chosen_text": [], "reject_text": [], "chosen_response_ranges": [], "reject_response_ranges": []}

        for chosen_ids, chosen_mask, reject_ids, reject_mask, info in batch:
            chosen_ids_list.append(chosen_ids)
            chosen_masks_list.append(chosen_mask)
            reject_ids_list.append(reject_ids)
            reject_masks_list.append(reject_mask)
            infos["chosen_text"].append(info["chosen_text"])
            infos["reject_text"].append(info["reject_text"])
            infos["chosen_response_ranges"].append(info["chosen_response_ranges"])
            infos["reject_response_ranges"].append(info["reject_response_ranges"])

        # 利用右侧补齐
        chosen_ids_padded = zero_pad_sequences(chosen_ids_list, "right", self.tokenizer.pad_token_id)
        chosen_masks_padded = zero_pad_sequences(chosen_masks_list, "right")
        reject_ids_padded = zero_pad_sequences(reject_ids_list, "right", self.tokenizer.pad_token_id)
        reject_masks_padded = zero_pad_sequences(reject_masks_list, "right")
        # response_ranges 保持原样，不进行 padding
        return chosen_ids_padded, chosen_masks_padded, reject_ids_padded, reject_masks_padded, infos

    def print_sample(self, index=0):
        # 展示一个样本的部分信息，方便调试
        chosen_ids, chosen_mask, reject_ids, reject_mask, info = self[index]
        print("🔹 Chosen Input:")
        print(info["chosen_text"])
        # 根据 chosen_input_ids 和 response_ranges 构造 labels 样例
        chosen_labels = [-100] * len(chosen_ids)
        for start, end in info["chosen_response_ranges"]:
            for i in range(start, min(end, len(chosen_ids))):
                chosen_labels[i] = chosen_ids[i].item()
        visible_chosen = [2 if x == -100 else x for x in chosen_labels]
        print("\n🔹 Chosen Labels (assistant部分已标记):")
        print(self.tokenizer.decode(visible_chosen, skip_special_tokens=False))
        print("\n🔹 Reject Input:")
        print(info["reject_text"])
        # 根据 reject_input_ids 和 response_ranges 构造 labels 样例
        reject_labels = [-100] * len(reject_ids)
        for start, end in info["reject_response_ranges"]:
            for i in range(start, min(end, len(reject_ids))):
                reject_labels[i] = reject_ids[i].item()
        visible_reject = [2 if x == -100 else x for x in reject_labels]
        print("\n🔹 Reject Labels (assistant部分已标记):")
        print(self.tokenizer.decode(visible_reject, skip_special_tokens=False))
