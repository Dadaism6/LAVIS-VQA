import torch
import torch.nn as nn
from PIL import Image
from lavis.models import load_model_and_preprocess
# setup device to use
device = torch.device("cuda:3") if torch.cuda.is_available() else "cpu"
# load sample image
raw_image = Image.open("/home/chenda/metadrive.jpg").convert("RGB")

model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

result, last_vision_embedding, second_last_vision_embedding, last_llm_embedding, second_last_llm_embedding = (
    model.generate({"image": image, "prompt": "Imaging you are the driver, can you safely drive straightforward? Please Explain."}))
print(result)
class CustomTransformerModel(nn.Module):
    def __init__(self):
        super(CustomTransformerModel, self).__init__()
        self.vision_proj = nn.Linear(768, 512)  # Project vision token size to 512
        self.llm_proj = nn.Linear(4096, 512)  # Project llm token size to 512
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))  # CLS token
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2, dim_feedforward=512, dropout=0.1, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_proj = nn.Linear(512, 128)  # Project transformer output to summary info size

    def forward(self, vision_tokens_1, vision_tokens_2, llm_tokens_1, llm_tokens_2):
        # Project tokens
        vision_tokens_1 = self.vision_proj(vision_tokens_1)
        vision_tokens_2 = self.vision_proj(vision_tokens_2)
        llm_tokens_1 = self.llm_proj(llm_tokens_1)
        llm_tokens_2 = self.llm_proj(llm_tokens_2)

        # Concatenate all tokens and CLS token
        concatenated_tokens = torch.cat([self.cls_token.repeat(vision_tokens_1.size(0), 1, 1), vision_tokens_1, vision_tokens_2, llm_tokens_1, llm_tokens_2], dim=1)

        # Pass through transformer
        transformer_output = self.transformer_encoder(concatenated_tokens)

        # Extract CLS token and project to summary vector
        cls_output = transformer_output[:, 0, :]  # Assuming CLS token is the first token
        summary_info = self.output_proj(cls_output)

        return summary_info