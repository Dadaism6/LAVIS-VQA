"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_t5 import Blip2T5


@registry.register_model("blip2_t5_lora")
class Blip2T5Lora(Blip2T5):
    """
    BLIP2 T5 model, with LoRA Applied.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xl_vitL: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xl_vitL": "configs/models/blip2/blip2_pretrain_flant5xl_vitL.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def forward(self, samples):
        image = samples["image"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens = self.t5_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
