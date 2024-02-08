import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image
raw_image = Image.open("D:\\research\\metavqa-merge\\MetaVQA\\vqa\\verification1\\12_104\\rgb_12_104.png").convert("RGB")

model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

model.generate({"image": image, "prompt": "Write a short description for the image."})