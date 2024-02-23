import torch
import torch.nn as nn
from PIL import Image
from lavis.models import load_model_and_preprocess
import time

# setup device to use
device = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"
start_time_load_model = time.time()
model, vis_processors, _ = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
end_time_load_model = time.time()
print(f"====Load Model took {end_time_load_model-start_time_load_model} seconds to complete.")
# load sample image
raw_image = Image.open("/home/chenda/metadrive.jpg").convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
caption = "Imaging you are the driver, can you safely drive straightforward? Please Explain."
sample = {"image": image, "text_input": [caption]}


start_time_multimodal = time.time()
features_multimodal = model.extract_features(sample)
end_time_multimodal = time.time()
print(f"====Multimodal Feature Extraction took {end_time_multimodal-start_time_multimodal} seconds to complete.")


start_time_image = time.time()
features_image = model.extract_features(sample, mode="image")
end_time_image = time.time()
print(f"====Image Feature Extraction took {end_time_image-start_time_image} seconds to complete.")


start_time_text = time.time()
features_text = model.extract_features(sample, mode="text")
end_time_text = time.time()
print(f"====Text Feature Extraction took {end_time_text-start_time_text} seconds to complete.")

