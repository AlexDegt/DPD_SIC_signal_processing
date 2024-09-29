import torch
import torch.nn as nn
from .batchnorm import Identity #ComplexBatchNorm1d
from typing import Union, List
from .activation import configure_activates
from collections import OrderedDict
import numpy as np
import sys

ListOfStr = List[str]
ListOfInt = List[int]
ListOfListInt = List[int]
OptionalStr = Union[str, None]
OptionalList = Union[list, None]

class Encoder(nn.Module):
    """
        Class of encoder, which includes sequence of multi-head self-attentions, linear layers, layer-norms and activations.
    """
    def __init__(self, in_embed_size: int, out_embed_size: int, interm_embed_size: ListOfInt=[12], num_heads: ListOfInt=[1], 
                p_drop: OptionalList=None, activate: ListOfStr=['sigmoid'], layer_norm_mode: str='nothing', bias: bool=True, 
                device: OptionalStr=None, dtype: torch.dtype=torch.float64):
        """
        Constructor of the Encoder class. 

        Args:
            in_embed_size (int): Dimensionality of input embeddings.
            out_embed_size (int): Dimensionality of output embeddings.
            interm_embed_size (list of int): List of intermediate embedding sizes, i.e. sizes of embeddings
                at the output of 1-st linear layer (input of second). Defaults to [12].
            num_heads (list of int): List of numbers of heads in each self-attention blocks. Defaults to [1].
            p_drop (list, optional): List of droput parameters. Each elements corresponds to the probability of an 
                element to be zeroed at each convolutional layer. Number of list elements must match number of convolutional layers. 
                If p_drop == None, then dropout of each layer is set to 0. Default to "None".
            activate (list of str): List of activation function names which are used in each conv. layer.
                i-th element of list corresponds to i-th conv. layer. Default to ['sigmoid', 'sigmoid'].
            layer_norm_mode (str): Type of layer normalization which is used in encoder.
                'common' -- Usual layer normalization: calcualtes variance, bias of the input batch and tunes adaptive scale and shift.
                'nothing' -- The is no layer normalization, bypass.
                Default to 'nothing'.
            bias (bool): Parameter shows whether to exploit bias in convolutional layers or not. Default to True.
            device (str, optional): Parameter shows which device to use for calculation on.
                'cpu', None -- CPU usage.
                'cuda' -- GPU usage.
            dtype (torch.complex64 or torch.complex128): Parameter type. Default to torch.complex128.
        """
        super().__init__()
        assert len(activate) == len(interm_embed_size), \
            "Number of activation functions must be the same as number of output embedding sizes."
        if p_drop is None:
            p_drop = [0 for i in range(len(interm_embed_size))]
        assert len(p_drop) == len(interm_embed_size), \
            "Number of dropout parameters must be the same as number of output embedding sizes."
        self.device = device
        self.dtype = dtype
        self.activate = activate
        # FC-layers initialization
        self.num_layers = len(interm_embed_size)
        self.in_embed_size = in_embed_size
        self.out_embed_size = out_embed_size
        self.interm_embed_size = interm_embed_size
        self.num_heads = num_heads
        self.p_drop = p_drop
        # List of encoder components
        self.encoder = nn.Sequential()
        self.attention, self.dp1, self.norm1, self.lin1, self.activation, \
        self.lin2, self.dp2, self.norm2 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), \
                                        nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for layer_i in range(self.num_layers):

            self.attention.append(nn.MultiheadAttention(embed_dim=self.in_embed_size, num_heads=num_heads[layer_i], bias=True, 
                                            batch_first=True, device=device, dtype=dtype))

            self.dp1.append(nn.Dropout(self.p_drop[layer_i]))

            if layer_norm_mode == 'common':
                self.norm1.append(nn.LayerNorm(normalized_shape=(self.in_embed_size,), device=device, dtype=dtype))
            if layer_norm_mode == 'nothing':
                self.norm1.append(Identity())

            self.lin1.append(nn.Linear(in_features=self.in_embed_size, out_features=self.interm_embed_size[layer_i], 
                            bias=True, device=device, dtype=dtype))

            self.activation.append(configure_activates(activate[layer_i], channel_num=self.interm_embed_size[layer_i], 
                                             dtype=dtype))

            self.lin2.append(nn.Linear(in_features=self.interm_embed_size[layer_i], out_features=self.in_embed_size, 
                            bias=True, device=device, dtype=dtype))

            self.dp2.append(nn.Dropout(self.p_drop[layer_i]))

            if layer_norm_mode == 'common':
                self.norm2.append(nn.LayerNorm(normalized_shape=(self.in_embed_size,), device=device, dtype=dtype))
            if layer_norm_mode == 'nothing':
                self.norm2.append(Identity())
        self.lin_out = nn.Linear(in_features=self.in_embed_size, out_features=self.out_embed_size, 
                                bias=True, device=device, dtype=dtype)

    def forward(self, x_in):
        # Input has dims (batch_size, embed_size, seq_len)
        x_in = torch.permute(x_in, (0, 2, 1))
        # Copy of model input to implement residual connection
        x_curr = torch.clone(x_in)
        for layer_i in range(self.num_layers):
            # Self-attention takes dims (batch_size, seq_len, embed_size) with batch_first=True
            x_curr, _ = self.attention[layer_i](x_curr, x_curr, x_curr)
            # Dropout input can have any shape, current: (batch_size, seq_len, embed_size)
            x_curr = self.dp1[layer_i](x_curr)
            # Add residual connection
            x_curr += x_in
            # LayerNorm deals with last embed_size dimensions, thus dims: (batch_size, seq_len, embed_size)
            x_curr = self.norm1[layer_i](x_curr)
            # Copy 1-st linear layer input to implement residual connection
            x_lin1_in = torch.clone(x_curr)
            # Output of LayerNorm and input of linear layer are the same: (batch_size, seq_len, embed_size)
            x_curr = self.activation[layer_i](self.lin1[layer_i](x_curr))
            # 2-nd linear layer (batch_size, seq_len, embed_size_interm)
            x_curr = self.lin2[layer_i](x_curr)
            # Dropout input can have any shape, current: (batch_size, seq_len, embed_size)
            x_curr = self.dp2[layer_i](x_curr)
            # Add residual connection
            x_curr += x_lin1_in
            # LayerNorm deals with last embed_size dimensions, thus dims: (batch_size, seq_len, embed_size)
            x_curr = self.norm2[layer_i](x_curr)
        # Output linear layer to transform embedding into desired dimensionality
        x_curr = self.lin_out(x_curr)
        # Output has dimensionality: (batch_size, output_embed_size, seq_len)
        return torch.permute(x_curr, (0, 2, 1))