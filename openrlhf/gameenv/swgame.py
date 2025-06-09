from scienceworld import ScienceWorldEnv
from openrlhf.gameenv.chatutils import (
    print_color, print_game_ob
)
from openrlhf.gameenv.swprompter import Prompter
import json
import os
from openrlhf.gameenv.apichat import get_completion_from_messages

from openrlhf.gameenv.chatutils import plans_obs, summary_obs, reflact_obs, wrong_obs

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

def get_simstr(taskidx):
    # To solve the bugs in the game environment, bees cannot be considered animals.
    if taskidx.startswith("4-") or taskidx.startswith("7-"):
        return "teleportAction"
    else:
        return "easy"
    
class SwGame(ScienceWorldEnv):
    def __init__(self, taskidx, print_steps=False, max_step=200, *args, **kwargs):
        assert os.path.exists('data/scienceworld.jar'), "file does not exist."
        current_path = os.getcwd()
        jar_path = f"{current_path}/data/scienceworld.jar"
        super().__init__(serverPath=jar_path, envStepLimit=200, *args, **kwargs) 
        # super().__init__(*args, **kwargs)   

        self.prompter = Prompter(f"openrlhf/gameenv/tdt-4o-prompts.json")
        self.taskidx = taskidx
        self.simstr = get_simstr(taskidx)
        self.print_steps = print_steps
        self.max_step = max_step
        self.custom_actions = {
            "plan": self.custom_action_plan,
            "long": self.custom_action_long
        }
        self.recode_actions = []

    def reset(self):
        self.step_id = 0
        self.task_desc = self.get_task_description()
        observation, info = super().reset()
        observation = self.task_desc
        
        if self.print_steps:
            print("*" * 120)
            print_color(observation, "pink")

        self.current_act = ""
        self.current_obs = observation
        self.current_info = info
        self.current_rwd = info['score']
        
        self.recode_actions = []
        return observation, info

    def step(self, action):
        self.recode_actions.append(action)
        if self.step_id > self.max_step:
            return "You meet the max step in the game!", self.current_rwd, 1, self.current_info
        
        if action in self.custom_actions:
            observation, reward, isCompleted, info = self.custom_actions.get(action)(action)
        elif action.startswith("jump to step"):
            observation, reward, isCompleted, info = self.custom_action_reflect(action)
        else:
            observation, reward, isCompleted, info = super().step(action)

        if self.print_steps:
            print_game_ob(observation, self.step_id, info["score"], action, info['inv'])

        self.current_act = action
        self.current_obs = observation
        self.current_info = info
        self.current_rwd = reward
        
        self.step_id += 1
        
        return observation, reward, isCompleted, info
    
    def custom_action_plan(self, act):
        # Define the custom behavior for custom_action_1
        observation = plans_obs
        reward = 0  # Custom reward
        isCompleted = False
        info = self.current_info
        return observation, reward, isCompleted, info
       
    def custom_action_long(self,act):
        # Define the custom behavior for custom_action_1
        observation = wrong_obs
        reward = 0  # Custom reward
        isCompleted = False
        info = self.current_info
        return observation, reward, isCompleted, info

    def run_test(self, taskidx, idv=None):
        if type(idv) == list:
            for v in idv:
                self.run_test(taskidx, v)
        self.load(taskidx, 0, generateGoldPath=False, simplificationStr="easy")     
        if idv is None:
            idvs = self.get_variations_test()
            for idv in idvs:
                self.run_test(taskidx, idv)
            return
        self.load(taskidx, idv, generateGoldPath=False, simplificationStr="easy")
        print(f"{taskidx}_{idv}")
        
        self.reset()
        done = False
        step = 100
        lastact = ""
        
        while not done and step:
            step -= 1
            action = input("Action: >>>")
            if action != 'last': lastact=action
            if action == "quit" or action == "q":
                return
            elif action == "next" or action == "n":
                return
            elif action == "last":
                action = lastact
            _, _, done, info = self.step(action)


if __name__=='__main__':
    taskidx, idv = "1-1", 20
    env = SwGame("5-2", True)
    env.run_test(taskidx, idv)
