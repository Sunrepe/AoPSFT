# from jericho import *
# from prompter import Prompter

import os
import sys
sys.path.append(os.getcwd())
print(sys.path)
print("Current working directory:", os.getcwd())
import json
from openrlhf.gameenv import chatutils
from gameenv import llamaapi
from gameenv.swgame import SwGame, FORMER_TASK_NAMES
import numpy as np
from gameenv import apichat

def save_json(dic, filename=""):    
    if not filename:
        filename = "data/sw-all.json" 
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)

def load_json(filename=""):
    if not filename:
        filename = "rungame/oneshot.json" 
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if data else {}

FORMER_TASK_NAMES = {
    "1-1": "task-1-boil",
    "1-2": "task-1-melt",
    "1-3": "task-1-freeze",
    "1-4": "task-1-change-the-state-of-matter-of",
    "2-1": "task-10-use-thermometer",
    "2-2": "task-10-measure-melting-point-(known-substance)",
    "2-3": "task-10-measure-melting-point-(unknown-substance)",
    "3-1": "task-2-power-component",
    "3-2": "task-2-power-component-(renewable-vs-nonrenewable-energy)",
    "3-3": "task-2a-test-conductivity",
    "3-4": "task-2a-test-conductivity-of-unknown-substances",
    "4-1": "task-3-find-living-thing",
    "4-2": "task-3-find-non-living-thing",
    "4-3": "task-3-find-plant",
    "4-4": "task-3-find-animal",
    "5-1": "task-4-grow-plant",
    "5-2": "task-4-grow-fruit",
    "6-1": "task-5-chemistry-mix",
    "6-2": "task-5-chemistry-mix-paint-(secondary-color)",
    "6-3": "task-5-chemistry-mix-paint-(tertiary-color)",
    "7-1": "task-6-lifespan-(longest-lived)",
    "7-2": "task-6-lifespan-(shortest-lived)",
    "7-3": "task-6-lifespan-(longest-lived-then-shortest-lived)",
    "8-1": "task-7-identify-life-stages-1",
    "8-2": "task-7-identify-life-stages-2",
    "9-1": "task-8-inclined-plane-determine-angle",
    "9-2": "task-8-inclined-plane-friction-(named-surfaces)",
    "9-3": "task-8-inclined-plane-friction-(unnamed-surfaces)",
    "10-1": "task-9-mendellian-genetics-(known-plant)",
    "10-2": "task-9-mendellian-genetics-(unknown-plant)",
}


start_prompt = "You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.\n\nThe available actions are:\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"

def print_color(text, background_color="blue"):
    colors = {
        "blue": "\033[38;5;15m\033[48;5;32m",  
        "green": "\033[38;5;0m\033[48;2;82;209;204m",  
        "pink": "\033[38;5;0m\033[48;2;221;160;221m", 
        "orange": "\033[38;5;0m\033[48;5;208m", 
        "reset": "\033[0m", 
    }

    if background_color in colors:
        colored_text = f"{colors[background_color]}{text}{colors['reset']}"
        print(colored_text)
    else:
        print(text)


def make_massages_reset(desc, taskidx, oneshot=False, traintype="tdt"): 
    if traintype=="sft":
        if oneshot:   
            oneshot_data = load_json()
            msgs = oneshot_data[taskidx]        
            msgs.append({"role": "user", "content": f"Here is the new task:\n\nObservation:{desc}\n"})
        else:
            msgs = [
                {"role": "user", "content": start_prompt},
                {"role": 'assistant', "content": "OK"},
                {"role": "user", "content": f"Observation: {desc}"}
            ]
    elif traintype=="tdt":        
        if oneshot:    
            oneshot_data = load_json()
            msgs = oneshot_data[taskidx]
            msgs.append({"role": "user", "content": f"Here is the new task:\n\nReward to go: 100\nObservation:{desc}\n\nStep: 0\n"})
        else:
            msgs = [
            {"role": "user", "content": start_prompt},
            {"role": 'assistant', "content": "OK"},
            {"role": "user", "content": f"Reward to go: 100\nObservation:{desc}\n\nStep: 0\n"}
            ]
    return msgs

def add_messages(msgs, res, obs):
    msgs.append({
        "role":'assistant', "content": res
    })
    msgs.append({
        "role": "user", "content": f"Observation: {obs}"
    })
    return msgs


def run_vals(env:SwGame, agent:llamaapi.LlamaAgent, varitions, taskname, taskidx):  
    vals_res = {}
    last_score = []
    for val in varitions:
        print(f"taskName:{taskname}\t, val:{val}")
        val = int(val) if type(val) == str else val
        env.load(taskidx, val, generateGoldPath=False, simplificationStr="easy")
        taskdesc = env.get_task_description()
        print(taskdesc)
        messages = make_massages_reset(desc=taskdesc, taskidx=taskidx, oneshot=oneshot)
        
        obs, info = env.reset()
        done = False
        # step = 100  
        act_list = []
        scores = [info['score']]
        paths = [{
                "observation": obs,
                "action": "start",
                "look": info['look'],
                "inv": info['inv'],
                "score": info['score']
            }]
        while not done:
            response = agent.get_completion_from_messages(messages)
            action = response.split(":")[-1].strip()
            if not action:
                action = "look around"
            act_list.append(action)
            obs, reward, done, info = env.step(action)
            add_messages(messages, response, obs)
            scores.append(info['score'])
            recod = {
                "observation": obs,
                "gpt": response,
                "look": info['look'],
                "inv": info['inv'],
                "score": info['score']
            }
            paths.append(recod)

        last_score.append(info['score'])
        val_dic = {str(val): {
            "scores": scores,
            "path":paths
        }}
        vals_res.update(val_dic)        
        result_dic[taskidx]["tmpres"] = vals_res
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result_dic, f, ensure_ascii=False, indent=4)
        apichat.pretty_print_json(filename, 2)

    return vals_res, last_score

def runenv(taskidx, taskname, val=None):
    if not result_dic.get(taskidx):
        result_dic[taskidx]={"task":taskname,"ave_seen":0.0, "ave_unseen":0.0,"seen":{}, "unseen":{}}
    
    env = SwGame(taskidx, max_step=60, print_steps=True)
    agent = llamaapi.LlamaAgent(modelname=modelname, port=port)
    val_lth = 10    


    env.load(taskidx, 0, generateGoldPath=False, simplificationStr="easy")
    all_train_datalist = load_json("data/sw/sw_trainlist.json")

    varitions = all_train_datalist.get(taskidx, None)
    if varitions:     
        varitions = varitions[:val_lth]
        # varitions=  [ 0,1]
        res_seen, scs = run_vals(env, agent, varitions, taskname, taskidx)
        result_dic[taskidx]["seen"] = res_seen    
        result_dic[taskidx]["ave_seen"] = sum(scs)/len(scs)

    varitions = env.get_variations_test()
    varitions = varitions[:val_lth]
    
    res_unseen,scs = run_vals(env, agent, varitions, taskname, taskidx)
    result_dic[taskidx]["unseen"] = res_unseen
    result_dic[taskidx]["ave_unseen"] = sum(scs)/len(scs)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result_dic, f, ensure_ascii=False, indent=4)
    apichat.pretty_print_json(filename, 2)


def runsw():
    test = True
    for k, v in FORMER_TASK_NAMES.items():
        runenv(k, v)
    

def all_train():
    data = load_json("gamerun/data/oneshot.json")
    for k,v in FORMER_TASK_NAMES.items():
        if not data.get(v):
            print(k,v )

if __name__ == "__main__":

    modelname, port="llama3.2-8b", 8003
    oneshot = False
    if oneshot:        
        filename = f"result/sw-eval-score_{modelname}_1hot.json" 
    else:        
        filename = f"result/sw-eval-score_{modelname}_0shot.json"

    if os.path.exists(filename):
        result_dic = load_json(filename)
    else:
        result_dic = {}

    runsw()