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
    def __init__(self, modelname="qwen", api_key="token-abc123", base_url="http://0.0.0.0", port=8006):
        self.api_key = api_key
        self.base_url = f"{base_url}:{port}/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = modelname

    def call_openai_api(self, instruction, model="llama3.2-1b"):
        """调用 OpenAI API 并返回结果"""
        messages = [{"role": "user", "content": instruction}]
        result = self.client.chat.completions.create(messages=messages, model=model)
        return result.choices[0].message.content
    
    # @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def get_completion_from_messages(self, messages, temperature=0.95):
        """获取模型返回的完成信息"""
        response = self.client.chat.completions.create(
            model=self.model,
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
        messages = [
        {'role': 'user', 'content': "You are a helpful assistant to do some scientific experiment in an environment. Completing the entire experiment will earn a full score of 100 points, meaning that the reward to go starts at 100 points. When you complete all the correct actions, the reward to go will be 0 points, indicating that you have completed all operations. \nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.\n\nThe available actions are:\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"},
        {"role": 'assistant', "content": "OK"},
        {'role': 'user', 'content': 'Reward to go: 100\nObservation:Your task is to boil Mercury. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter.\n\nStep: 1\n'}
    ]
    
        ss = self.get_completion_from_messages(messages)
        print(ss)


# 使用示例
if __name__ == "__main__":
    agent = LlamaAgent(port=8001)
    # agent.runtest()
    # res = "Thought: I should pour the red paint from the wood cup that's directly in the art studio into the metal pot, as it's more straightforward.\nAction: 1\nAction: 1"
    
    # print(agent.parse_action(res))
    # 示例消息
    messages = [
        {'role': 'user', 'content': "You are a helpful assistant to do some scientific experiment in an environment. Completing the entire experiment will earn a full score of 100 points, meaning that the reward to go starts at 100 points. When you complete all the correct actions, the reward to go will be 0 points, indicating that you have completed all operations. \nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.\n\nThe available actions are:\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"},
        {"role": 'assistant', "content": "OK"},
        {'role': 'user', 'content': 'Reward to go: 100\nObservation:Your task is to boil Mercury. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter.\n\nStep: 1\n'}
    ]
    
    ss = agent.get_completion_from_messages(messages)
    print(ss)

