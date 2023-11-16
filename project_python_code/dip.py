import torch
import numpy as np
from models import *


class DIP(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, num_hidden_channels=128, use_skip=True, num_encoder_layers = 4):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.use_skip =use_skip
        self.num_encoder_layers = num_encoder_layers
        self.num_hidden_channels = num_hidden_channels

        self.encoder_layers = []
        self.decoder_layers = []
        self.encoder_layers.append(conv_and_down(in_channels=self.num_in_channels, out_channels=self.num_hidden_channels))
        for i in range(self.num_encoder_layers-1):
            self.encoder_layers.append(conv_and_down(in_channels=self.num_hidden_channels, out_channels=self.num_hidden_channels))

        self.decoder_layers.append(conv_and_up(in_channels=self.num_hidden_channels, out_channels=self.num_hidden_channels))
        for i in range(self.num_encoder_layers-1):
            self.decoder_layers.append(conv_and_up(in_channels=self.num_hidden_channels*2, out_channels=self.num_hidden_channels))
        self.encoder = nn.ModuleList(self.encoder_layers)
        self.decoder = nn.ModuleList(self.decoder_layers)

        self.final_layer = conv_bn_relu_block(in_channels=self.num_hidden_channels, out_channels=self.num_out_channels, 
                                              activation="sigmoid", kernel_size=(3, 3), use_bn=False)

       
    
    def forward(self, x):
        encoder_feat = []
        y = x.clone()
        for _layer in self.encoder_layers:
            y = _layer(y)
            encoder_feat.append(y)
        
        encoder_feat = encoder_feat[::-1] # reverse for skip
        for _i, _layer in enumerate(self.decoder):
            if _i == 0:
                y = _layer(y)
            else:
                # print(f"{y.shape=} {encoder_feat[_i].shape=}, {_i=}")
                concat_feat_y = torch.concat([y, encoder_feat[_i]], dim=1)
                y = _layer(concat_feat_y)
        
        out = self.final_layer(y)
        return out


if __name__ == "__main__":
    device = torch.device("cuda")
    dip = DIP(1, 1, num_hidden_channels=128, num_encoder_layers=4).to(device)
    print(dip)

    inp = torch.randn(1, 1, 512,512).to(device)
    out = dip(inp)
    print(out.shape)

