import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class ModifiedViT(nn.Module):
    def __init__(self, pretrained_model_name="google/vit-base-patch16-224", input_size=(1, 1, 64, 256), patch_size=(16, 16), num_output_channels=2, smoothing_kernel_size=3, dropout_rate=0.5):
        super(ModifiedViT, self).__init__()

        self.config = ViTConfig.from_pretrained(pretrained_model_name)
        self.num_output_channels = num_output_channels

        self.config.image_size = (input_size[2], input_size[3])
        self.config.num_channels = input_size[1]
        self.config.patch_size = patch_size
        self.config.num_patches = (input_size[2] // patch_size[0]) * (input_size[3] // patch_size[1])

        self.vit = ViTModel.from_pretrained(
            pretrained_model_name,
            config=self.config,
            ignore_mismatched_sizes=True
        )

        self.fc_out = nn.Linear(self.config.hidden_size, input_size[3] * input_size[0] * input_size[1] * self.num_output_channels)

        self.smoothing = nn.Conv2d(
            in_channels=self.num_output_channels,
            out_channels=self.num_output_channels,
            kernel_size=smoothing_kernel_size,
            padding=smoothing_kernel_size // 2,
            bias=False
        )

        nn.init.constant_(self.smoothing.weight, 1.0 / (smoothing_kernel_size ** 2))

        self.dropout = nn.Dropout(dropout_rate)

        self.input_size = input_size

    def forward(self, x):
        batch_size = x.size(0)
        
        outputs = self.vit(pixel_values=x)
        cls_output = outputs.last_hidden_state[:, 1:, :]

        
        output = self.fc_out(cls_output)
        
        output = self.dropout(output)

        output = output.view(batch_size, self.num_output_channels, self.input_size[2], self.input_size[3])
        output = self.smoothing(output)
        return output
    
    def predict(self, x, device='cpu'):
        """
        descripcion:
        """
        self.eval()
        x = x.to(device) 
        with torch.no_grad():
            output = self.forward(x)
        return output

