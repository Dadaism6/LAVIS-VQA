import datetime
import os
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_api_key_file(wandb_key_file):
    if wandb_key_file is not None:
        default_path = os.path.expanduser(wandb_key_file)
    else:
        default_path = os.path.expanduser("~/wandb_api_key_file.txt")
    if os.path.exists(default_path):
        print("We are using this wandb key file: ", default_path)
        return default_path
    path = os.path.join(root, "wandb", "wandb_api_key_file.txt")
    print("We are using this wandb key file: ", path)
    return path