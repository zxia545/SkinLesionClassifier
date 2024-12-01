# model.py

import torch
import torch.nn as nn
from transformers import BlipForConditionalGeneration, BlipConfig

class BLIPFineTuner(nn.Module):
    def __init__(self, num_classes):
        super(BLIPFineTuner, self).__init__()
        # Load pre-trained BLIP model
        self.blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")
        
        # Freeze BLIP parameters if desired
        for param in self.blip.parameters():
            param.requires_grad = False
        
        # Replace the decoder with a classification head
        # BLIP's encoder outputs hidden states which can be used for classification
        self.classifier = nn.Linear(self.blip.config.text_config.hidden_size, num_classes)
        
        # Optionally, unfreeze some layers for fine-tuning
        # Example: Unfreeze the last layer of the encoder
        for param in self.blip.vision_model.encoder.layers[-1].parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        """
        Args:
            pixel_values (Tensor): Input images tensor of shape [batch_size, 3, 224, 224]
        
        Returns:
            logits (Tensor): Classification logits of shape [batch_size, num_classes]
        """
        # Pass images through the BLIP encoder
        encoder_outputs = self.blip.vision_model(pixel_values=pixel_values)
        last_hidden_state = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Pool the encoder outputs (mean pooling)
        pooled_output = last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
        
        # Classification head
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        
        return logits
