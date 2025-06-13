import torch
import torch.nn as nn

class SingleBlockTransformer(nn.Module):
    def __init__(self, vocab_size=10, d_model=32, nhead=4, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed
        x = self.transformer(x)
        logits = self.fc_out(x)
        return logits