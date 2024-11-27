import torch
import torch.nn as nn

class BasicTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, num_heads, num_layers, dropout=0.1):
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

    def forward(self, x):
        x = x.squeeze(1)

        x = self.embedding(x) 

        x = self.transformer(x, x)

        x = self.fc_out(x)
        x = x.view(x.size(0), -1, self.seq_len, self.input_dim)
        return x
    

