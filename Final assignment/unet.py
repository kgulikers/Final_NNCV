# Import all the necessary libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from huggingface_hub import login

# Token used for authentication with Hugging Face
login(token='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')


# Define the UNet class using Segformer as the backbone
class UNet(nn.Module):
    def __init__(
        self,
        pretrained_model: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        num_classes: int = 19,
        image_size: tuple[int, int] = (1024, 1024),
        freeze_backbone: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.model = SegformerForSemanticSegmentation.from_pretrained(pretrained_model)
        self.encoder = self.model.segformer  
        self.decoder = self.model.decode_head  

        # This avoids unnecessary memory usage 
        self.model.config.use_cache = False

        # Freeze the backbone (encoder) if specified
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                param.requires_grad = name.startswith("decode_head")
            if debug:
                trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
                frozen    = [n for n, p in self.model.named_parameters() if not p.requires_grad]
                print(f"trainable ({len(trainable)}):\n  " + "\n  ".join(trainable))
                print(f"frozen   ({len(frozen)}):\n  " + "\n  ".join(frozen))

        # adjust classifier head if num_classes differs
        if self.model.config.num_labels != num_classes:
            self.model.config.num_labels = num_classes
            old_cls = self.model.decode_head.classifier
            self.model.decode_head.classifier = nn.Conv2d(
                in_channels=old_cls.in_channels,
                out_channels=num_classes,
                kernel_size=old_cls.kernel_size,
                stride=old_cls.stride,
                padding=old_cls.padding,
            )

    def unfreeze_backbone(self):
        """
        Unfrezes the encoder parameters for training. 
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen")

    def forward(self, pixel_values: torch.Tensor):
        """
        Forward pass through the model.
        """
        outputs = self.model(pixel_values=pixel_values)
        logits = F.interpolate(
            outputs.logits,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return logits, None