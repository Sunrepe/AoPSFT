from openrlhf.gameenv import chatutils
from openrlhf.gameenv import swgame
import os
import json


FORMER_TASK_NAMES = {
    "1-1": "task-1-boil",
    "1-2": "task-1-melt",
    "1-3": "task-1-freeze",
    "1-4": "task-1-change-the-state-of-matter-of",
    # "2-1": "task-10-use-thermometer",
    # "2-2": "task-10-measure-melting-point-(known-substance)",
    # "2-3": "task-10-measure-melting-point-(unknown-substance)",
    # "3-1": "task-2-power-component",
    # "3-2": "task-2-power-component-(renewable-vs-nonrenewable-energy)",
    # "3-3": "task-2a-test-conductivity",
    # "3-4": "task-2a-test-conductivity-of-unknown-substances",
    # "4-1": "task-3-find-living-thing",
    # "4-2": "task-3-find-non-living-thing",
    # "4-3": "task-3-find-plant",
    # "4-4": "task-3-find-animal",
    # "5-1": "task-4-grow-plant",
    # "5-2": "task-4-grow-fruit",
    "6-1": "task-5-chemistry-mix",
    "6-2": "task-5-chemistry-mix-paint-(secondary-color)",
    "6-3": "task-5-chemistry-mix-paint-(tertiary-color)",
    # "7-1": "task-6-lifespan-(longest-lived)",
    # "7-2": "task-6-lifespan-(shortest-lived)",
    # "7-3": "task-6-lifespan-(longest-lived-then-shortest-lived)",
    "8-1": "task-7-identify-life-stages-1",
    "8-2": "task-7-identify-life-stages-2",
    # "9-1": "task-8-inclined-plane-determine-angle",
    # "9-2": "task-8-inclined-plane-friction-(named-surfaces)",
    # "9-3": "task-8-inclined-plane-friction-(unnamed-surfaces)",
    # "10-1": "task-9-mendellian-genetics-(known-plant)",
    # "10-2": "task-9-mendellian-genetics-(unknown-plant)",
}


def print_game_ob(ob, step):
    black = "\033[38;5;0m"
    reset = "\033[0m"
    background_color = "221;160;221"
    colored_text = (
        f"{black}\033[48;2;{background_color}m"
        f"{'='*20} Step {step} {'='*20}\n{ob}"
        f"{reset}"
    )
    print(colored_text)


def print_game_action(act, score):
    black = "\033[38;5;0m"
    reset = "\033[0m"
    background_color = "82;209;204"
    colored_text = (
        f"{black}\033[48;2;{background_color}m"
        f">>>{act}\nTotal score: {score}"
        f"{reset}"
    )
    print(colored_text)


if __name__ == "__main__":
    
    taskidx= "1-1"
    idvs = [0]

    env = swgame.SwGame(taskidx, True)
    env.run_test(taskidx, idvs)
