from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os

from .utils import zero_pad_sequences
from openrlhf.gameenv.chatutils import make_default_ipt


def preprocess_data(
    data, input_template=None, input_key="input", output_key=None, apply_chat_template=None, multiturn=False
):
    if apply_chat_template:
        if output_key:
            prompt_message = data[input_key]
            response_message = data[output_key]

            if isinstance(prompt_message, str) and isinstance(response_message, str):
                prompt_message = [{"role": "user", "content": prompt_message}]
                response_message = [{"role": "assistant", "content": response_message}]

            prompt = apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(prompt_message + response_message, tokenize=False)[len(prompt) :]
        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tokenize=False)[len(prompt) :]
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        response = data[output_key] if output_key else ""
    return prompt, response

class mtSFTDataset(Dataset):
    """
    专门处理多轮对话数据的 SFT 数据集

    参数:
        dataset: 原始数据集，每个样本中 data[input_key] 为对话轮次列表
        tokenizer: 用于编码文本的 tokenizer，同时需要包含 apply_chat_template 方法
        max_length: 输入最大长度
        strategy: 包含配置信息，其 args 中应提供 input_key 和 tokenizer_chat_template 等参数
        input_template: 可选的文本模板
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=8,  # Specify the number of processors you want to use
        multiple_of=1,
        multiturn=True,
    ) -> None:
        assert multiturn, "只能处理多轮对话数据"

        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiple_of = multiple_of
        self.multiturn = multiturn  # 固定为 True，多轮对话

        # self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", "input")
        # 对于多轮对话，不再使用 output_key
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", True)
        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # 并行预处理数据：利用 map 方法处理所有样本
        processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        self.prompts = processed_dataset["prompt"]
        self.response_ranges = processed_dataset["response_ranges"]
        self.importants = processed_dataset["important"]
        # if self.multiturn and os.getppid() == os.getpid():
        #     self.print_sample()


    def process_data(self, data):
        # 对于多轮数据，假设 data[self.input_key] 为包含对话轮次的列表
        ipt = data.get("ipt", None)
        prompt, response_ranges, ipts = preprocess_mtdata(
            messages=data[self.input_key],
            msg_ipt=ipt,
            apply_chat_template=self.apply_chat_template,
            max_length=self.max_length,
            tokenizer=self.tokenizer,
        )
        return {"prompt": prompt, "response_ranges":response_ranges, "important": ipts}

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        final_text = self.prompts[idx]
        response_ranges = self.response_ranges[idx]
        ipt = self.importants[idx]
        
        final_text = final_text.rstrip("\n")
        if not final_text.endswith(self.tokenizer.eos_token):
            final_text += self.tokenizer.eos_token

        # 对最终文本调用 tokenizer（仅调用一次）得到 input_ids 与 attention_mask
        input_token = self.tokenizer(
            final_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = input_token["input_ids"][0]  # tensor, shape 为 (seq_len,)
        attention_mask = input_token["attention_mask"][0]
        prompt_ids_len = attention_mask.int().sum().item()
        seq_len = input_ids.shape[0]
        # 构造 label 序列，初始全部为 -100
        labels = [-100] * seq_len
        impts = [0] * seq_len

        for (start, end), imp in zip(response_ranges, ipt):
            for i in range(start, min(end, seq_len)):
                labels[i] = input_ids[i].item()
                impts[i] = imp

        info = {
            "input": final_text,
            "labels": torch.tensor(labels),
            "important": torch.tensor(impts)
        }
        return prompt_ids_len, input_ids, attention_mask, info
    
    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids_list = []
        attention_masks_list = []
        labels_list = []
        ipts_list = []
        # infos = {"input": [], "labels": []}

        for prompt_ids_len, input_ids, attention_mask, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)
            labels_list.append(info["labels"])
            ipts_list.append(info["important"])
            # infos["input"].append(info["input"])

        # 对 input_ids 和 attention_masks 使用右侧补齐
        input_ids_padded = zero_pad_sequences(input_ids_list, "right", self.tokenizer.pad_token_id)
        attention_masks_padded = zero_pad_sequences(attention_masks_list, "right")
        # 对 labels 进行补齐，注意填充值为 -100
        labels_padded = zero_pad_sequences(labels_list, "right", -100)
        ipts_padded = zero_pad_sequences(ipts_list, "right", 0)
        # infos["labels"] = labels_padded

        return input_ids_padded, attention_masks_padded, labels_padded, ipts_padded
    
    def print_sample(self, index=None):        
        # ✅ 只允许 rank 0 打印（分布式主进程）
        if not self.strategy.is_rank_0(): 
            return  # 非主进程，直接跳过
        
        if index is None:
            import random
            index = random.randint(0, len(self)-1)        
        _, input_ids, _, info = self[index]  # 调用 __getitem__

        # 获取原始 input 文本
        # input_text = info["input"]
        # print("🔹 Input:")
        # print(input_text)
        # print()
        
        # 修改 input_ids：在每个 token 后插入一个 2
        input_ids_modified = []
        for token_id in input_ids.tolist():
            input_ids_modified.extend([token_id, 2])  # 插入 token 和 2

        # 解码修改后的 input_ids 序列
        decoded_modified_input = self.tokenizer.decode(input_ids_modified, skip_special_tokens=False)
        print("🔹 Input:(with # inserted as cut):")
        print(decoded_modified_input)

        # 解码可视化 label 序列: 获取 labels，并替换 -100 为 2（或其他可视化符号）
        labels = info["labels"].tolist()
        visible_labels = [2 if x == -100 else x for x in labels]
        decoded = self.tokenizer.decode(visible_labels, skip_special_tokens=False)
        print("🔹 Labels (decoded, -100 → #):")
        print(decoded)
        print()
        # 可视化 important 序列
        ipts = info["important"].tolist()
        print("🔹Important:")
        print(ipts)
        print()
