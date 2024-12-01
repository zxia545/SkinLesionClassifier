# model.py

import torch
import torch.nn as nn
from transformers import CLIPModel

class CLIPFineTuner(nn.Module):
    def __init__(self, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # Freeze the text encoder and language projection layers
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
        self.clip_model.logit_scale.requires_grad = False
        self.clip_model.text_projection.requires_grad = False

        # Replace the image projection head with a classification head
        self.classifier = nn.Linear(self.clip_model.config.projection_dim, num_classes)

    def forward(self, pixel_values):
        # Get image embeddings
        image_embed = self.clip_model.vision_model(pixel_values=pixel_values).pooler_output
        image_embed = self.clip_model.visual_projection(image_embed)
        # Classification head
        logits = self.classifier(image_embed)
        return logits
