import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from PIL import Image
from lavis.models import load_model_and_preprocess


class CustomVLMEncoderFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int,
                 device: torch.device = torch.device("cuda:2")):
        super().__init__(observation_space, features_dim)

        # Load the pretrained model and preprocessors
        self.device = device
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)

        # Initialize the custom transformer model
        # self.custom_transformer_model = CustomTransformerModel(num_transformer_layer=num_transformer_layer,
        #                                                        out_features=features_dim)

        # Move to the appropriate device
        self.model.to(self.device)
        # self.custom_transformer_model.to(self.device)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Preprocess the observations (images) using the loaded visual preprocessors
        # Assuming observations are already tensors of the correct shape and dtype
        observations=observations["image"][..., -1] * 255
        # processed_images = self.vis_processors["eval"](observations).unsqueeze(0).to(
        #         self.device)
        processed_images = self.vis_processors["eval"](observations).to(
            self.device)
        sample = {"image": processed_images, "text_input": ["Imaging you are the driver, can you safely drive straightforward? Please Explain."]}
        features_image = self.model.extract_features(sample, mode="image")
        proj_image_embed = features_image.image_embeds_proj.view(1, -1)
        # Pass embeddings through the custom transformer model
        # summary_info = self.custom_transformer_model(vision_tokens_1, vision_tokens_2, llm_tokens_1, llm_tokens_2)
        # features_image.image_embeds   features_image.image_embeds_proj
        return proj_image_embed

