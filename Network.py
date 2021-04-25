import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
import math, copy, time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout=0.2):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:

            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None):
        self.src = src
        self.src_mask = None
        if trg is not None:
            self.trg = trg
            self.trg_mask = \
                self.make_std_mask(self.trg)

    @staticmethod
    def make_std_mask(tgt):
        "Create a mask to hide padding and future words."
        tgt_mask =Variable(
            subsequent_mask(tgt.size(-1)))
        tgt_mask=tgt_mask.to(device)
        return tgt_mask

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and many other models.
    """
    def __init__(self, encoder, decoder,
                 src_embed, tgt_embed, input_enc_len, input_dec_len, out_seq_len, d_model):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        # Encoder对象
        self.decoder = decoder
        # Decoder对象
        self.src_embed = src_embed
        # 源语言序列的编码，包括词嵌入和位置编码
        self.tgt_embed = tgt_embed
        #1 is the dimension
        self.enc_input_fc = nn.Linear(1, d_model)
        self.dec_input_fc = nn.Linear(1, d_model)
        self.out_fc = nn.Linear(d_model, 1)

        # 目标语言序列的编码，包括词嵌入和位置编码

    def forward(self, src, tgt, src_mask, tgt_mask):
        out=self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)
        return out.to(device)

    def encode(self, src, src_mask):
        in_enc = self.enc_input_fc(src)
        in_enc=in_enc.to(device)
        encode=self.encoder(self.src_embed(in_enc), src_mask)
        return encode.to(device)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        in_dec = self.dec_input_fc(tgt)
        result=self.decoder(self.tgt_embed(in_dec),
                            memory, src_mask, tgt_mask)
        out=self.out_fc(result)

        return out.to(device)


def make_model(input_enc_len,input_dec_len, out_seq_len,N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.2):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    attn = MultiHeadedAttention(h, d_model)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(c(position)),
        nn.Sequential(c(position)), input_enc_len, input_dec_len, out_seq_len,d_model)
    model=model.to(device)
    return model
