import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def create_pad_mask(t, pad):
    mask = (t == pad)
    return mask.to(t.device)


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2),
                 padding=(0, 0), dilation=(1, 1), bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = Linear(filter_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, nhead, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(hidden_size, nhead, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, padding_mask):
        y = self.self_attention_norm(x)
        y, self_att_weights = self.self_attention(y, y, y, attn_mask=None, key_padding_mask=padding_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x, self_att_weights


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, nhead, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(hidden_size, nhead, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.enc_dec_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.enc_dec_attention = nn.MultiheadAttention(hidden_size, nhead, dropout_rate)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, padding_mask, attn_mask, memory_mask):
        y = self.self_attention_norm(x)
        y, self_att_weights = self.self_attention(y, y, y, attn_mask=attn_mask, key_padding_mask=padding_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.enc_dec_attention_norm(x)
        y, mutihead_att_weights = self.enc_dec_attention(y, enc_output, enc_output, attn_mask=None,
                                                         key_padding_mask=memory_mask)
        y = self.enc_dec_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x, self_att_weights, mutihead_att_weights


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, nhead):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(hidden_size, filter_size, nhead, dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, inputs, padding_mask):
        encoder_output = inputs
        self_attentions = []
        encoder_output = encoder_output.permute([1, 0, 2])
        for enc_layer in self.layers:
            encoder_output, self_att_weights = enc_layer(encoder_output, padding_mask)
            self_attentions.append(self_att_weights)
        encoder_output = encoder_output.permute([1, 0, 2])
        return self.last_norm(encoder_output), self_attentions


class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, nhead):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(hidden_size, filter_size, nhead, dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(decoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, targets, enc_output, padding_mask, attn_mask, memory_mask):
        decoder_output = targets
        self_attentions = []
        mutihead_attentions = []
        decoder_output = decoder_output.permute([1, 0, 2])
        enc_output = enc_output.permute([1, 0, 2])
        for i, dec_layer in enumerate(self.layers):
            decoder_output, self_att_weights, mutihead_att_weights = dec_layer(decoder_output, enc_output,
                                                                               padding_mask, attn_mask, memory_mask)
            self_attentions.append(self_att_weights)
            mutihead_attentions.append(mutihead_att_weights)
        decoder_output = decoder_output.permute([1, 0, 2])
        return self.last_norm(decoder_output), self_attentions, mutihead_attentions


class Transformer(nn.Module):
    def __init__(self, encoder_layers=6, decoder_layers=6,
                 hidden_size=512, filter_size=2048,
                 encoder_nhead=8, decoder_nhead=8, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5

        self.decoder = Decoder(hidden_size, filter_size,
                               dropout_rate, decoder_layers, encoder_nhead)

        self.encoder = Encoder(hidden_size, filter_size,
                               dropout_rate, encoder_layers, decoder_nhead)

    def forward(self, inputs, targets, i_padding_mask, t_padding_mask, t_attn_mask):
        enc_output, en_self_attns = self.encoder(inputs, i_padding_mask)
        dec_output, de_self_attns, mutihead_attns = self.decoder(targets, enc_output, t_padding_mask, t_attn_mask,
                                                                 i_padding_mask)
        return enc_output, dec_output, en_self_attns, de_self_attns, mutihead_attns


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Speech_Transformer(nn.Module):
    def __init__(self, d_model, vocal_size,
                 nhead=4, num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 conv_kenel_size=3, rate=0.1):
        super(Speech_Transformer, self).__init__()

        self.d_model = d_model

        self.trg_pad_idx = 0

        self.conv1 = Conv(in_channels=1,
                          out_channels=d_model,
                          kernel_size=(conv_kenel_size, conv_kenel_size),
                          w_init='tanh')
        self.conv2 = Conv(in_channels=d_model,
                          out_channels=d_model,
                          kernel_size=(conv_kenel_size, conv_kenel_size),
                          w_init='tanh')
        self.conv3 = Conv(in_channels=d_model,
                          out_channels=d_model,
                          kernel_size=(conv_kenel_size, conv_kenel_size),
                          w_init='tanh')
        self.conv4 = Conv(in_channels=d_model,
                          out_channels=d_model,
                          kernel_size=(conv_kenel_size, conv_kenel_size),
                          w_init='tanh')

        self.batch_norm1 = nn.BatchNorm2d(d_model)
        self.batch_norm2 = nn.BatchNorm2d(d_model)
        self.batch_norm3 = nn.BatchNorm2d(d_model)
        self.batch_norm4 = nn.BatchNorm2d(d_model)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)

        self.embed = nn.Embedding(vocal_size, d_model, padding_idx=0)
        self.embed_dropout = nn.Dropout(p=0.1)

        self.pos_encoder = PositionalEncoding(d_model, dropout=rate)
        self.pos_decoder = PositionalEncoding(d_model, dropout=rate)

        self.transfomer = Transformer(encoder_layers=num_encoder_layers,
                                      decoder_layers=num_decoder_layers,
                                      hidden_size=d_model, filter_size=dim_feedforward,
                                      encoder_nhead=nhead, decoder_nhead=nhead,
                                      dropout_rate=rate)

        self.t_encoder = Linear(d_model, vocal_size)

    def forward(self, input_, target_):
        input_ = input_.permute(0, 3, 1, 2)
        input_ = self.dropout1(torch.tanh(self.batch_norm1(self.conv1(input_))))
        input_ = self.dropout2(torch.tanh(self.batch_norm2(self.conv2(input_))))
        input_ = self.dropout3(torch.tanh(self.batch_norm3(self.conv3(input_))))
        input_ = self.dropout4(torch.tanh(self.batch_norm4(self.conv4(input_))))
        input_ = input_.flatten(2)
        input_ = input_.transpose(-1, -2)
        input_ = self.pos_encoder(input_)

        i_padding_mask = None
        t_padding_mask = create_pad_mask(target_, self.trg_pad_idx)

        target_size = target_.size()[1]
        t_attn_mask = generate_square_subsequent_mask(target_size, target_.device)

        target_embedded = self.embed(target_)

        target_embedded = target_embedded[:, :-1]
        target_embedded = F.pad(target_embedded, (0, 0, 1, 0))

        target_embedded *= math.sqrt(self.d_model)
        target_embedded = self.pos_decoder(target_embedded)
        target_embedded = self.embed_dropout(target_embedded)

        enc_output, dec_output, en_self_attns, de_self_attns, mutihead_attns = self.transfomer(input_, target_embedded,
                                                                                               i_padding_mask,
                                                                                               t_padding_mask,
                                                                                               t_attn_mask)

        dec_output = self.t_encoder(dec_output)

        return dec_output, en_self_attns, de_self_attns, mutihead_attns