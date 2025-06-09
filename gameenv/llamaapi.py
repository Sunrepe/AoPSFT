import json
from openai import OpenAI
from tqdm import tqdm  # 引入tqdm库
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

class LlamaAgent:
    def __init__(self, modelname="llama", api_key="0", base_url="http://0.0.0.0", port=8006):
        self.api_key = api_key
        self.base_url = f"{base_url}:{port}/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def call_openai_api(self, instruction, model="llama3.2-1b"):
        """调用 OpenAI API 并返回结果"""
        messages = [{"role": "user", "content": instruction}]
        result = self.client.chat.completions.create(messages=messages, model=model)
        return result.choices[0].message.content
    
    # @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def get_completion_from_messages(self, messages, model="llama", temperature=0.95, do_sample=False):
        """获取模型返回的完成信息"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        res = response.choices[0].message.content
        
        return res
    
    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action:(.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)
        action = action[0].strip()
        if "\n" in action:
            action = self.parse_action(action)
        assert action is not None
        return action

    def run(self, test_data):
        """运行测试并输出结果"""
        correct_count = 0
        total_count = len(test_data)

        # 使用 tqdm 显示进度条
        for entry in tqdm(test_data, desc="Processing", unit="entry"):
            instruction = entry['instruction']
            expected_output = entry['output']
            
            # 调用 OpenAI API
            api_result = self.call_openai_api(instruction)

            # 比较 API 输出与 expected_output
            if api_result.strip() == expected_output.strip():
                correct_count += 1

        accuracy = correct_count / total_count if total_count else 0
        print(f"Accuracy: {accuracy * 100:.2f}%")
    
    def runtest(self):    
        messages = [
            {'role': 'user', 'content': "tell me a joke about school life."},
        ]
    
        ss = self.get_completion_from_messages(messages)
        print(ss)


# 使用示例
if __name__ == "__main__":
    agent = LlamaAgent(api_key="0", port=9234)
    agent.runtest()

