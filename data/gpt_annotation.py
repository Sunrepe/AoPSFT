import json
import re
from typing import List, Dict, Tuple
import sys
import os
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())
from tqdm import tqdm
import random
import io
import shutil
from contextlib import redirect_stdout

from openrlhf.gameenv.apichat import chat_with_gpt
from openrlhf.gameenv.chatutils import (
    save_json, load_json, split_and_save_json, plans_obs, summary_obs, reflact_obs, run_in_parallel, load_jsonl,
    print_color
)
from openrlhf.gameenv.swprompter import Prompter

def format_trajectory_for_cot(trajectory):    
    parts = []
    for i, step in enumerate(trajectory):
        # 公共字段解析
        step_idx = step["step"]
        action = step.get('action', '')
        observation = step.get('observation', '')
        inv = step.get('inventory', '')
        score = step.get("score")
        reward = step.get('reward')
        llm = step.get("llm")
        if reward and reward > 0:
            observation = f"{observation}[You have received a {reward} point reward.]\n{inv}"
        if score and score ==100:
            assert False, "last step need not to be processed."

        parts.append(f"Observation: {observation}\nStep: {step_idx}\n<think>{llm}\n<action>{action}")
    return parts

def build_trajectory_string(trajectory: List[Dict], mode: str) -> str:

    if mode == "cot_step":        
        parts = format_trajectory_for_cot(trajectory)
        return "\n→\n".join(parts)
    else:
        raise ValueError(f"Invalid mode: {mode}. Supported modes: plan/step/summary")


class DataAnnotator:
    def __init__(self, llm_chat_func):
        self.llm_chat = llm_chat_func
        self.step_counter = 1
        self.prompter = Prompter("openrlhf/gameenv/tdt-4o-prompts.json")

    def _extract_subgoals(self, response: str) -> List[str]:
        return re.findall(r'\d+[\.、]?\s*(.+?)(?=\n|$)', response)
    
    def make_plans(self, task_desc, taskid):        
        msgs = self.prompter.make_prompts(task_desc, "make plan", taskid)
        responses = chat_with_gpt(msgs, model="gpt-4o-2024-11-20")
        think, plans = self.prompter.extract_commands(responses)
        return think, plans    
    
    def make_cot_step(self, trajectory:list) -> Dict:
        history_context = build_trajectory_string(trajectory[:-1], "cot_step")
        # make_current_obs()
        step = trajectory[-1]
        step_idx = step["step"]
        action = step.get('action', '')
        observation = step.get('observation', '')
        inv = step.get('inventory', '')
        score = step.get("score")
        reward = step.get('reward')
        if reward and reward > 0:
            observation = f"{observation}[You have received a {reward} point reward.]\n{inv}"

        current_obs = f"Current Observation: {observation}\nStep: {step_idx}\n\n<think>?\n<action>{action}"

        messages = [
            {
                "role": "system",
                "content": "You are a text-based game player. You need to provide the chain-of-thought reasoning for choosing a certain action x under the current state. Specifically:"
                            "\n1. You need to return a first-person analysis to fill in the '?' part."
                            "\n2. In your chain-of-thought analysis, appropriately consider historical information. At important states (usually when a sub-goal is completed or a score/reward is obtained), analyze whether your sub-goal is completed and what the next sub-goal is."
                            "\n3. Keep the analysis concise and professional; never exceed 3 sentences(At key states, briefly mention the upcoming general objective). In most cases, a single sentence explaining the current objective and related action is sufficient."
            },
            {
                "role": "user",
                "content": f"Interaction history: {history_context} \n→\n {current_obs}\n\n"
                        f"Based on Current Observation and the actual optimal action {action}, "
                        f"please analyze in first-person perspective(Use I or my) to fill in the '?' part — that is, explain why {action} is the optimal action. Please keep the analysis concise."
            }
        ]

        return self.llm_chat(messages, model="gpt-4o-2024-11-20").strip()
        
    def analyze_trajectory(self, taskid:str, trajectory:list) -> Dict:
        if trajectory[2].get("llm") is not None: 
            return
        if trajectory[1]["action"]=="look around":
            obs = [
                "I've just started the game, so I need to look around first to figure out where I am.",
                "At the beginning of the game, I look around to orient myself.",
                "Since the game has just begun, my first step is to look around and identify my current location."
            ]
            llm_out = random.choice(obs)
            trajectory[1]["llm"]=llm_out
        for i in range(3,len(trajectory)):
            llm_out = self.make_cot_step(trajectory[:i])
            trajectory[i-1]["llm"]=llm_out
    
def make_cot_step_action():    
    # 用于给定了plans的, step级别的COT Think过程标注
    alldatafile = "data/sw/Humanplay-3series-max10.json"
    data = load_json(alldatafile)
    
    annotator = DataAnnotator(chat_with_gpt)
    for k, v in data.items():
        task_id, val = k.split("_")
        annotator.analyze_trajectory(task_id, v["path"])
        save_json(data, alldatafile)

def add_plan():    
    file = "data/sw/sw_aop_play-new.json"
    data = load_json(file)
    prompter = Prompter("openrlhf/gameenv/tdt-4o-prompts.json")
    already_done = set()
    for k, v in data.items():
        if k in already_done:
            continue
        print(f"{'*' * 100}\n{k}")
        taskdesc = v.get("taskDescription", None)
        assert taskdesc is not None, f"taskdesc is None: {k}"
        taskid = k.split("_")[0]
        msgs = prompter.make_prompts(taskdesc, "make plan", taskid)
        already_done.add(k)
        responses = chat_with_gpt(msgs, model="gpt-4o-2024-11-20")
        think, plans = prompter.extract_commands(responses)
        print(f"plans: {plans}")


if __name__ == "__main__":
    # add plans for each task
    add_plan()
    # make COT thinking for each step action
    make_cot_step_action()
