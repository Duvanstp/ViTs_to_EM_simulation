import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, num_heads, num_layers, dropout=0.5, smoothing_kernel_size=3):
        super(BasicTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, seq_len)

        self.transformer = nn.Transformer(
            d_model=seq_len,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(seq_len, output_dim * input_dim)
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.input_dim = input_dim

        # Capa de suavizado
        self.smoothing = nn.Conv2d(
            in_channels=self.output_dim,
            out_channels=self.output_dim,
            kernel_size=smoothing_kernel_size,
            padding=smoothing_kernel_size // 2,
            bias=False
        )

        nn.init.constant_(self.smoothing.weight, 1.0 / (smoothing_kernel_size ** 2))


    def forward(self, x):

        x = x.to(next(self.parameters()).device)
        x = x.squeeze(1)

        x = self.embedding(x)

        x = self.transformer(x, x)

        x = self.fc_out(x)
        x = x.view(x.size(0), -1, self.seq_len, self.input_dim)
        x = self.smoothing(x) 
        # x = x.squeeze(1)
        # x = x.view(x.size(0), -1, self.seq_len, self.input_dim)
        return x
    # def forward(self, x):
        # x = x.squeeze(1)

        # x = self.embedding(x) 

        # x = self.transformer(x, x)

        # x = self.fc_out(x)
        # x = x.view(x.size(0), -1, self.seq_len, self.input_dim)
        # return x
    
    def predict(self, x, device='cpu'):
        """
        Realiza predicciones con el modelo.
        
        Args:
            x (torch.Tensor): Tensor de entrada con forma (batch_size, seq_len, input_dim).
            device (str): Dispositivo donde realizar la inferencia ('cpu' o 'cuda').

        Returns:
            torch.Tensor: Predicciones con forma (batch_size, seq_len, output_dim).
        """
        self.eval()
        x = x.to(device) 
        with torch.no_grad():
            output = self.forward(x)
        return output

