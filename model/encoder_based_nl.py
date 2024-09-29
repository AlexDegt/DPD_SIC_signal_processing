import torch, sys
from .layers import FEAT_EXTR, Encoder
from itertools import chain

class EncoderBasedNL(torch.nn.Module):
    '''
        EncoderBasedNL is a sequence of encoders.
        Takes 2 signals as an input: x_{n}, p_{n}, - pure signal and normalized PA output power and 
        creates input features: Re(x_{n}), Im(x_{n}), |x_{n}|, p_{n}.
        Thus there're 4 input channels. This number of channels corresponds to input embedding size.
        Each self-attention block is followed by linear layer, which transforms embedding into dimensionality, defined by out_embed_size list.
        Output embedding size equals 2, which correspond to Re(x_last_layer) and Im(x_last_layer) part of 
        pre-distorted signal. Output of Encoder is Re(x_last_layer) + 1j * Im(x_last_layer).
    '''
    def __init__(self, interm_embed_size=[12], num_heads=[1], p_drop=None, activate=['sigmoid', 'sigmoid'], 
                layer_norm_mode='nothing', features=['real', 'imag', 'abs'], bias=True, device=None, dtype=torch.float64):
        super().__init__()
        self._dtype = dtype
        self._device = device

        # Feature extractor module is used to create 3 * input_features_number: 
        # Re(x_{0, n}), Im(x_{0, n}), |x_{0, n}|, ..., Re(x_{N-1, n}), Im(x_{N-1, n}), |x_{N-1, n}|. 
        self.feature_extract = FEAT_EXTR(features=features, device=device, dtype=dtype)
        
        # Encoder based on self-attention blocks
        self.nonlin = Encoder(in_embed_size=len(features) + 1, out_embed_size=2, interm_embed_size=interm_embed_size, 
                            num_heads=num_heads, p_drop=p_drop, activate=activate, layer_norm_mode=layer_norm_mode, bias=bias, 
                            device=device, dtype=dtype)
        
    def forward(self, x):
        x_curr = self.feature_extract(x[:, :1, :])
        x_curr = torch.cat([x_curr.real, x[:, 1:, :].real], dim=1)
        x_curr = self.nonlin(x_curr)
        output = x_curr[:, :1, :] + 1j * x_curr[:, 1:2, :]
        return output