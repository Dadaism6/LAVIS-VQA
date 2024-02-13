import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from PIL import Image
from lavis.models import load_model_and_preprocess


class CustomVLMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128,
                 device: torch.device = torch.device("cuda")):
        super().__init__(observation_space, features_dim)

        # Load the pretrained model and preprocessors
        self.device = device
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct",
                                                                       model_type="vicuna7b", is_eval=True,
                                                                       device=self.device)

        # Initialize the custom transformer model
        self.custom_transformer_model = CustomTransformerModel()

        # Move to the appropriate device
        self.model.to(self.device)
        self.custom_transformer_model.to(self.device)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Preprocess the observations (images) using the loaded visual preprocessors
        # Assuming observations are already tensors of the correct shape and dtype
        processed_images = [
            self.vis_processors["eval"](obs).unsqueeze(0).to(
                self.device) for obs in observations]

        # Generate embeddings using the pretrained model
        embeddings = [self.model.generate({"image": image, "prompt": "Describe this scene."})[1:5] for image in
                      processed_images]  # Extracts last and second-last vision and LLM embeddings

        # Prepare embeddings for the transformer model
        vision_tokens_1, vision_tokens_2, llm_tokens_1, llm_tokens_2 = zip(*embeddings)
        vision_tokens_1 = torch.stack(vision_tokens_1).squeeze(1)
        vision_tokens_2 = torch.stack(vision_tokens_2).squeeze(1)
        llm_tokens_1 = torch.stack(llm_tokens_1).squeeze(1)
        llm_tokens_2 = torch.stack(llm_tokens_2).squeeze(1)

        # Pass embeddings through the custom transformer model
        summary_info = self.custom_transformer_model(vision_tokens_1, vision_tokens_2, llm_tokens_1, llm_tokens_2)

        return summary_info


# Define the CustomTransformerModel as provided
class CustomTransformerModel(nn.Module):
    def __init__(self):
        super(CustomTransformerModel, self).__init__()
        self.vision_proj = nn.Linear(768, 512)  # Project vision token size to 512
        self.llm_proj = nn.Linear(4096, 512)  # Project llm token size to 512
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))  # CLS token
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2, dim_feedforward=512, dropout=0.1,
                                                   activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_proj = nn.Linear(512, 128)  # Project transformer output to summary info size

    def forward(self, vision_tokens_1, vision_tokens_2, llm_tokens_1, llm_tokens_2):
        # Project tokens
        vision_tokens_1 = self.vision_proj(vision_tokens_1)
        vision_tokens_2 = self.vision_proj(vision_tokens_2)
        llm_tokens_1 = self.llm_proj(llm_tokens_1)
        llm_tokens_2 = self.llm_proj(llm_tokens_2)

        # Concatenate all tokens and CLS token
        concatenated_tokens = torch.cat(
            [self.cls_token.repeat(vision_tokens_1.size(0), 1, 1), vision_tokens_1, vision_tokens_2, llm_tokens_1,
             llm_tokens_2], dim=1)

        # Pass through transformer
        transformer_output = self.transformer_encoder(concatenated_tokens)

        # Extract CLS token and project to summary vector
        cls_output = transformer_output[:, 0, :]  # Assuming CLS
