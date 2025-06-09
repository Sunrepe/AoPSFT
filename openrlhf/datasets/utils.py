import torch
import torch.nn.functional as F


def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)

def exist_and_not_none(d, key):
    return key in d and not d[key] is None

def print_sample_from_dataset(dataset, index=0, datype="sft"):
    """
    打印 RewardDataset 的一个样例，包括 prompt、chosen、reject 及其 input_ids。
    
    Args:
        dataset: RewardDataset 实例
        index: 要展示的样例索引，默认展示第一个样例
    """
    if datype == "rm":
    # 获取样例
        input_ids_chosen, attn_mask_chosen, input_ids_reject, attn_mask_reject, extra = dataset[index]

        # 将 token id 解码为文本（去掉 padding）
        chosen_text = dataset.tokenizer.decode(
            input_ids_chosen[0][input_ids_chosen[0] != dataset.tokenizer.pad_token_id], skip_special_tokens=False
        )
        reject_text = dataset.tokenizer.decode(
            input_ids_reject[0][input_ids_reject[0] != dataset.tokenizer.pad_token_id], skip_special_tokens=False
        )

        # 打印结果
        print("=== Sample Index:", index, "===")
        print("\n[Prompt + Chosen Text]:\n", chosen_text)
        print("\n[Chosen input_ids]:\n", input_ids_chosen)
        print("\n[Attention Mask - Chosen]:\n", attn_mask_chosen)

        print("\n[Prompt + Rejected Text]:\n", reject_text)
        print("\n[Reject input_ids]:\n", input_ids_reject)
        print("\n[Attention Mask - Reject]:\n", attn_mask_reject)

        print("\n[Extra]:\n", extra)
    elif datype=="sft":
        prompt_ids_len, input_ids, attention_mask, info = dataset[index]
        decoded_input = dataset.tokenizer.decode(input_ids[0][:prompt_ids_len], skip_special_tokens=True)
        decoded_output = dataset.tokenizer.decode(input_ids[0][prompt_ids_len:], skip_special_tokens=True)

        print(f"[Prompt] {decoded_input}")
        print(f"[Response] {decoded_output}")
        print(f"[Full text] {dataset.tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
        print(f"[Input length] {info['input_length']}")
        if dataset.multiturn:
            print(f"[Response ranges] {info['response_ranges']}")
        print()
