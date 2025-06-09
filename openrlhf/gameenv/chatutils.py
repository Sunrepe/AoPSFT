from openai import OpenAI
import os
import re
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


default_system = "You are a helpful assistant to do some scientific experiment in an environment. \nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.\n\nThe available actions are:\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step\n"

def save_json(dic, filename):
    folder = os.path.dirname(filename)    
    if folder and not os.path.exists(folder):
        os.makedirs(folder)    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)

def load_json(filename=""):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if data else {}
    except FileNotFoundError:
        return {}
   
def append_messages_to_jsonl(messages, file_path):
    folder = os.path.dirname(file_path)
    
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    
    with open(file_path, "a", encoding="utf-8") as f:
        for message in messages:
            json_line = json.dumps(message, ensure_ascii=False)
            f.write(json_line + "\n")

def load_jsonl(file_path):
    aggregated_data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if isinstance(data, dict):
                    for outer_key, inner_dict in data.items():
                        if outer_key not in aggregated_data:
                            aggregated_data[outer_key] = {}
                        if isinstance(inner_dict, dict):
                            for inner_key, inner_value in inner_dict.items():
                                aggregated_data[outer_key][inner_key] = inner_value
            except json.JSONDecodeError:
                pass 
    return aggregated_data

def make_default_ipt(msgs):    
    role_to_ipt = {"system": 0, "user": 0, "assistant": 1}
    ipt = [role_to_ipt[msg["role"]] for msg in msgs]
    return ipt


def add_messages_to_json(new_messages, file_path):
    with open(file_path, "r+") as file:
        data = json.load(file)
        data.extend(new_messages)
        file.seek(0)
        json.dump(data, file)
        file.truncate()

def add_message_to_json(file_path, new_message):
    with open(file_path, "r+") as file:
        data = json.load(file)
        data.append(new_message)
        file.seek(0)
        json.dump(data, file)
        file.truncate()

def print_chat_response(text):
    reset = "\033[0m"
    font_color = "0"
    background_color = "82;209;204"
    colored_text = (
        f"\033[38;5;{font_color}m\033[48;2;{background_color}m" f"Assitant: {text}" f"{reset}"
    )
    print(colored_text)


TASK_MAX_STEP ={'1-1': 42, '1-2': 44, '1-3': 73, '1-4': 38, '10-1': 54, '10-2': 61, '5-2': 58, '6-1': 38, '6-2': 37, '6-3': 43, '8-1': 45, '8-2': 35, '9-1': 34, '9-2': 34, '9-3': 34, '2-2': 77, '2-3': 55, '3-3': 40, '4-4': 29, '7-3': 27, '3-4': 39, '2-1': 39, '4-3': 29, '4-1': 29, '5-1': 41, '7-2': 26, '4-2': 27, '7-1': 26, '3-1': 31, '3-2': 42}


def print_color(text, background_color="blue"):
    def rgb_color(fg_rgb=None, bg_rgb=None):
        fg = f"\033[38;2;{fg_rgb[0]};{fg_rgb[1]};{fg_rgb[2]}m" if fg_rgb else ""
        bg = f"\033[48;2;{bg_rgb[0]};{bg_rgb[1]};{bg_rgb[2]}m" if bg_rgb else ""
        return fg + bg

    def is_bright_color(rgb):
        r, g, b = rgb
        brightness = (r*299 + g*587 + b*114) / 1000
        return brightness > 186

    color_map = {
        "blue":     ((255, 255, 255), (36, 92, 174)),     
        "green":    ((255, 255, 255), (33, 115, 70)),     
        "pink":     ((40, 40, 40), (233, 215, 223)),      
        "orange":   ((255, 255, 255), (204, 102, 0)),     
        "purple":   ((255, 255, 255), (85, 60, 105)),     
        "gray":     ((0, 0, 0), (240, 240, 240)),         
        "reset":    None
    }

    reset_code = "\033[0m"

    if background_color in color_map and color_map[background_color]:
        fg_rgb, bg_rgb = color_map[background_color]

        if is_bright_color(bg_rgb):
            fg_rgb = (0, 0, 0)
        else:
            fg_rgb = (255, 255, 255)

        color_code = rgb_color(fg_rgb, bg_rgb)
        print(f"{color_code}{text}{reset_code}")
    elif background_color == "reset":
        print(f"{reset_code}{text}")
    else:
        print(f"[Unsupported color '{background_color}'] {text}")
        print(f"Available colors: {', '.join(color_map.keys())}")



def print_chat_prompt(text):
    style = ("\033[37;100m", "\033[0m")

    styled_text = f"{style[0]}{text}{style[1]}"
    print(styled_text)


def print_env_ob(act, obs):
    black = "\033[38;5;0m"
    reset = "\033[0m"
    background_color = "245;245;220"
    colored_text = f"{black}\033[48;2;{background_color}m" f">>{act}\n{obs}" f"{reset}"
    print(colored_text)


def print_game_ob(ob, step, score, act, reward):
    black = "\033[38;5;0m"
    reset = "\033[0m"
    background_color = "221;160;221"
    colored_text = (
        f"{black}\033[48;2;{background_color}m"
        f"{'='*20} Step {step} {'='*20}\n"
        f">>> {act}\n{ob}\nScore: {score}\nInventory: {reward}"
        f"{reset}"
    )
    print(colored_text)

def append_to_file(s, l, file_path):
    s_str = str(s)
    l_str = ",".join(map(str, l))

    with open(file_path, "a") as file:
        file.write(s_str + "," + l_str + "\n")



if __name__ == "__main__":
    mess = [{"role": "user", "content": "Say this is a test"}]