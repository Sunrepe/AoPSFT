import json
import re

class Prompter(object):
    __slots__ = ("template", "act_template")

    def __init__(self, template_name):
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            raise ValueError(f"Can't find templates")
        with open(template_name) as fp:
            self.template = json.load(fp)
            # self.template = self.template[taskid]

    def sw_plan(self, task):
        prompts = self.template["make plan"]
        notice = prompts.get("notice")
        num = len(prompts) >> 1
        if notice:
            task = f"{task}\n{notice}"
            num = (len(prompts)-1) >> 1
        msgs = [{"role": "system", "content": prompts["sw_plan_syt"]}]
        for i in range(num):
            msgs.append({"role": "user", "content": prompts[f"sw_plan_user_{i+1}"]})
            msgs.append({"role": "assistant", "content": prompts[f"sw_plan_assi_{i+1}"]})
        msgs.append({"role": "user", "content": task})
        return msgs

    def sw_sub_plan(self, task):
        prompts = self.template["sub-plan"]
        notice = prompts.get("notice")
        num = len(prompts) >> 1
        if notice:
            task = f"{task}\n{notice}"
            num = (len(prompts)-1) >> 1
        msgs = [{"role": "system", "content": prompts["sw_plan_syt"]}]
        for i in range(num):
            msgs.append({"role": "user", "content": prompts[f"sw_plan_user_{i+1}"]})
            msgs.append({"role": "assistant", "content": prompts[f"sw_plan_assi_{i+1}"]})
        msgs.append({"role": "user", "content": task})
        return msgs

    def sw_open_containers(self, look):
        prompts = self.template["open container"]
        msgs = [
            {"role": "system", "content": prompts},
            {"role": "user", "content": look},
        ]
        return msgs


    def sw_make_kg(self, text):
        prompts = self.template["world graph"]
        num = len(prompts) >> 1
        msgs = [{"role": "system", "content": prompts["sw_world_syt"]}]
        for i in range(num):
            msgs.append({"role": "user", "content": prompts[f"sw_world_user_{i+1}"]})
            msgs.append({"role": "assistant", "content": prompts[f"sw_world_assi_{i+1}"]})
        msgs.append({"role": "user", "content": text})
        return msgs

    def sw_make_acts(self, text):
        prompts = self.template["make action"]
        num = len(prompts) >> 1
        msgs = [{"role": "system", "content": prompts["sw_act_syt"]}]
        for i in range(num):
            msgs.append({"role": "user", "content": prompts[f"sw_act_user_{i+1}"]})
            msgs.append({"role": "assistant", "content": prompts[f"sw_act_assi_{i+1}"]})
        msgs.append({"role": "user", "content": text})
        return msgs


    def plan_check(self, plan, ob, last_act):
        text = f"The goal is to: {plan}.\nLast action was: {last_act}.\nObservation: '''{ob}'''"
        prompts = self.template["check plan"]
        num = len(prompts) >> 1
        msgs = [{"role": "system", "content": prompts["sw_checkplan_syt"]}]
        for i in range(num):
            msgs.append({"role": "user", "content": prompts[f"sw_checkplan_user_{i+1}"]})
            msgs.append({"role": "assistant", "content": prompts[f"sw_checkplan_assi_{i+1}"]})
        msgs.append({"role": "user", "content": text})
        return msgs

    def checkif_plan(self, text):
        prompts = self.template["check if"]
        num = len(prompts) >> 1
        msgs = [{"role": "system", "content": prompts["sw_checkif_syt"]}]
        for i in range(num):
            msgs.append({"role": "user", "content": prompts[f"sw_checkif_user_{i+1}"]})
            msgs.append({"role": "assistant", "content": prompts[f"sw_checkif_assi_{i+1}"]})
        msgs.append({"role": "user", "content": text})
        return msgs

    def make_prompts(self, text, label, taskidx="default"):
        prompts = self.template[taskidx].get(label)
        if not prompts:
            prompts = self.template["default"][label]

        notice = prompts.get("notice")
        num = len(prompts) >> 1
        if notice:
            text = f"{text}\n{notice}"
            num = (len(prompts)-1) >> 1
        msgs = [{"role": "system", "content": prompts["sw_syt"]}]
        for i in range(num):
            msgs.append({"role": "user", "content": prompts[f"sw_user_{i+1}"]})
            msgs.append({"role": "assistant", "content": prompts[f"sw_assi_{i+1}"]})
        msgs.append({"role": "user", "content": text})
        return msgs   
    
    def extract_commands(self, input_string):
        think_match = re.search(r"Think:\s*(.*?)\s*Sub-plans:", input_string, re.DOTALL)
        subplans_match = re.search(r"Sub-plans:\s*\[(.*?)\]", input_string, re.DOTALL)

        think_content = think_match.group(1).strip() if think_match else ""
        subplans_raw = subplans_match.group(1).strip() if subplans_match else ""
        subplans_list = [item.strip() for item in subplans_raw.split("→")] if subplans_raw else []

        return think_content, subplans_list
    
    def make_action_cot_msgs(self, taskidx, label):
        prompts = self.template[taskidx].get(label)
        if not prompts:
            prompts = self.template["default"][label]
        return prompts