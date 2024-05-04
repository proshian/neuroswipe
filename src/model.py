from typing import Callable, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwipeCurveTransformerEncoderv1(nn.Module):
    """
    Transformer-based Curve encoder takes in a sequence of vectors and creates a representation
    of a swipe gesture on a samrtphone keyboard.
    Each vector contains information about finger trajectory at a time step.
    It contains:
    * x coordinate
    * y coordinate
    * Optionally: t
    * Optionally: dx/dt
    * Optionally: dy/dt
    * Optionally: keyboard key that has x and y coordinates within its boundaries
    """

    def __init__(self,
                 input_size: int,
                 d_model: int,
                 dim_feedforward: int,
                 num_layers: int,
                 num_heads_first: int,
                 num_heads_other: int,
                 dropout: float = 0.1,
                 device = None):
        """
        Arguments:
        ----------
        input_size: int
            Size of input vectors.
        d_model: int
            Size of the embeddings (output vectors).
            Should be equal to char embedding size of the decoder.
        dim_feedforward: int
        num_layers: int
            Number of encoder layers including the first layer.

        """
        super().__init__()

        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.first_encoder_layer = nn.TransformerEncoderLayer(
            input_size, num_heads_first, dim_feedforward, dropout, device=device)
        self.liner = nn.Linear(input_size, d_model, device=device)  # to convert embedding to d_model size
        num_layer_after_first = num_layers - 1
        if num_layer_after_first > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, num_heads_other, dim_feedforward, dropout, device=device)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layer_after_first)
        else:
            self.transformer_encoder = None
    

    def forward(self, x, pad_mask: torch.Tensor):
        x = self.first_encoder_layer(x, src_key_padding_mask=pad_mask)
        x = self.liner(x)
        if self.transformer_encoder is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        return x



class SwipeCurveTransformerDecoderv1(nn.Module):
    """
    Decodes a swipe gesture representation into a sequence of characters.

    Uses decoder transformer with masked attention to prevent the model from cheating.
    """

    def __init__(self,
                 char_emb_size,
                 n_classes,
                 nhead,
                 num_decoder_layers,
                 dim_feedforward,
                 dropout,
                 activation = F.relu,
                 device = None):
        super().__init__()

        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.decoder_layer = nn.TransformerDecoderLayer(
            char_emb_size, nhead, dim_feedforward, dropout, activation, device = device)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        self.out = nn.Linear(char_emb_size, n_classes, device = device)
    
    def forward(self, x, memory, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask):
        x = self.transformer_decoder(x,
                                     memory,
                                     tgt_mask=tgt_mask,
                                     memory_key_padding_mask=memory_key_padding_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask)
        x = self.out(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.to(device=device)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
        ----------
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SwipeCurveTransformer(nn.Module):
    """
    Seq2seq model. Encodes a sequence of points of a
    swipe-keyboard-typing gesture into a sequence of characters.

    n_output_classes = char_vocab_size - 2 because <pad> and <sos>
    tokens are never predicted.
    """

    def _get_mask(self, max_seq_len: int):
        """
        Returns a mask for the decoder transformer.
        """
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def __init__(self,
                 n_coord_feats: int,
                 char_emb_size: int,
                 char_vocab_size: int,
                 key_emb_size: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dim_feedforward: int,
                 num_heads_encoder_1: int,
                 num_heads_encoder_2: int,
                 num_heads_decoder: int,
                 dropout:float,
                 char_embedding_dropout: float,
                 key_embedding_dropout: float,
                 max_out_seq_len: int,
                 max_curves_seq_len: int,
                 activation: Callable = F.relu,
                 device: Optional[str] = None):
        super().__init__()

        self.device = torch.device(
            device 
            or 'cuda' if torch.cuda.is_available() else 'cpu')

        input_feats_size = n_coord_feats + key_emb_size

        d_model = char_emb_size

        self.char_embedding_dropout = nn.Dropout(char_embedding_dropout)
        self.key_embedding_dropout = nn.Dropout(key_embedding_dropout)
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_size, device=device)
        self.key_embedding = nn.Embedding(char_vocab_size, key_emb_size, device=device)

        self.encoder = SwipeCurveTransformerEncoderv1(
            input_feats_size, d_model, dim_feedforward,
            num_encoder_layers, num_heads_encoder_1,
            num_heads_encoder_2, dropout, device=device)
        
        self.char_pos_encoder = PositionalEncoding(
            char_emb_size, max_out_seq_len, device=device)
        
        self.key_pos_encoder = PositionalEncoding(
            key_emb_size, max_curves_seq_len, device=device)
        
        n_classes = char_vocab_size - 2  # <sos> and <pad> are not predicted
        self.decoder = SwipeCurveTransformerDecoderv1(
            char_emb_size, n_classes, num_heads_decoder,
            num_decoder_layers, dim_feedforward, dropout, activation, device=device)

        self.mask = self._get_mask(max_out_seq_len).to(device=device)

    # def forward_old(self, x, kb_tokens, y, x_pad_mask, y_pad_mask):
    #     # Differs from forward(): uses self.mask instead of generating it.
    #     kb_k_emb = self.key_embedding(kb_tokens)  # keyboard key
    #     kb_k_emb = self.key_embedding_dropout(kb_k_emb)
    #     kb_k_emb = self.key_pos_encoder(kb_k_emb)
    #     x = torch.cat((x, kb_k_emb), dim = -1)
    #     x = self.encoder(x, x_pad_mask)
    #     y = self.char_embedding(y)
    #     y = self.char_embedding_dropout(y)
    #     y = self.char_pos_encoder(y)
    #     y = self.decoder(y, x, self.mask, x_pad_mask, y_pad_mask)
    #     return y
    
    def encode(self, x, kb_tokens, x_pad_mask):
        kb_k_emb = self.key_embedding(kb_tokens)  # keyboard key
        kb_k_emb = self.key_embedding_dropout(kb_k_emb)
        kb_k_emb = self.key_pos_encoder(kb_k_emb)
        x = torch.cat((x, kb_k_emb), dim = -1)
        x = self.encoder(x, x_pad_mask)
        return x
    
    def decode(self, x_encoded, y, x_pad_mask, y_pad_mask):
        y = self.char_embedding(y)
        y = self.char_embedding_dropout(y)
        y = self.char_pos_encoder(y)
        mask = self._get_mask(len(y)).to(device=self.device)
        y = self.decoder(y, x_encoded, mask, x_pad_mask, y_pad_mask)
        return y

    def forward(self, x, kb_tokens, y, x_pad_mask, y_pad_mask):
        x_encoded = self.encode(x, kb_tokens, x_pad_mask)
        return self.decode(x_encoded, y, x_pad_mask, y_pad_mask)




def get_m1_model(device = None, weights_path = None):
    CHAR_VOCAB_SIZE = 37  # = len(word_char_tokenizer.char_to_idx)
    MAX_CURVES_SEQ_LEN = 299
    MAX_OUT_SEQ_LEN = 35  # word_char_tokenizer.max_word_len - 1

    model = SwipeCurveTransformer(
        n_coord_feats=6,
        char_emb_size=128,
        char_vocab_size=CHAR_VOCAB_SIZE,
        key_emb_size=54,
        num_encoder_layers=4,
        num_decoder_layers=3,
        dim_feedforward=128,
        num_heads_encoder_1=4,
        num_heads_encoder_2=4,
        num_heads_decoder=4,
        dropout=0.1,
        char_embedding_dropout=0.1,
        key_embedding_dropout=0.1,
        max_out_seq_len=MAX_OUT_SEQ_LEN,
        max_curves_seq_len=MAX_CURVES_SEQ_LEN,
    device = device)

    if weights_path:
        model.load_state_dict(
            torch.load(weights_path,
                    map_location = device))
    
    model = model.to(device)
        
    model = model.eval()

    return model


def get_m1_bigger_model(device = None, weights_path = None):
    CHAR_VOCAB_SIZE = 37  # = len(word_char_tokenizer.char_to_idx)
    MAX_CURVES_SEQ_LEN = 299
    MAX_OUT_SEQ_LEN = 35  # word_char_tokenizer.max_word_len - 1

    model = SwipeCurveTransformer(
        n_coord_feats=6,
        char_emb_size=128,
        char_vocab_size=CHAR_VOCAB_SIZE,
        key_emb_size=66,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=128,
        num_heads_encoder_1=4,
        num_heads_encoder_2=4,
        num_heads_decoder=4,
        dropout=0.1,
        char_embedding_dropout=0.1,
        key_embedding_dropout=0.1,
        max_out_seq_len=MAX_OUT_SEQ_LEN,
        max_curves_seq_len=MAX_CURVES_SEQ_LEN,
        device = device)

    if weights_path:
        model.load_state_dict(
            torch.load(weights_path,
                    map_location = device))
    
    model = model.to(device)
        
    model = model.eval()

    return model


def get_m1_smaller_model(device = None, weights_path = None):
    CHAR_VOCAB_SIZE = 37  # = len(word_char_tokenizer.char_to_idx)
    MAX_CURVES_SEQ_LEN = 299
    MAX_OUT_SEQ_LEN = 35  # word_char_tokenizer.max_word_len - 1

    model = SwipeCurveTransformer(
        n_coord_feats=6,
        char_emb_size=128,
        char_vocab_size=CHAR_VOCAB_SIZE,
        key_emb_size=54,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=128,
        num_heads_encoder_1=4,
        num_heads_encoder_2=4,
        num_heads_decoder=4,
        dropout=0.1,
        char_embedding_dropout=0.1,
        key_embedding_dropout=0.1,
        max_out_seq_len=MAX_OUT_SEQ_LEN,
        max_curves_seq_len=MAX_CURVES_SEQ_LEN,
        device = device)

    if weights_path:
        model.load_state_dict(
            torch.load(weights_path,
                    map_location = device))
    
    model = model.to(device)
        
    model = model.eval()

    return model














# ##########################################
# import torch
# import math
# from torch import nn
# from torchaudio.models import Conformer


    




# class ConformerEncoderTransformerDecoderWithPos(nn.Module):

       
#     def __init__(self, 
#                  n_coord_feats,  # 6
#                  key_emb_size,  # 122
#                  char_vocab_size: int,
#                  num_encoder_layers,  # 6
#                  num_decoder_layers,  # 6
#                  dim_feedforward,  # 128 у меня. Но надо бы 2048 
#                  dropout:float,
#                  char_embedding_dropout: float,
#                  key_embedding_dropout: float,
#                  num_heads_encoder,  # 4 
#                  num_heads_decoder,  # 4
#                  max_out_seq_len: int,
#                  max_curves_seq_len: int,
#                  activation: Callable = F.relu,
#                  device = None,) -> None:
#         super().__init__()

#         self.device = torch.device(
#             device 
#             or 'cuda' if torch.cuda.is_available() else 'cpu')
        
#         d_model = n_coord_feats + key_emb_size

#         self.char_embedding_dropout = nn.Dropout(char_embedding_dropout)
#         self.key_embedding_dropout = nn.Dropout(key_embedding_dropout)
        
#         self.char_embedding = nn.Embedding(char_vocab_size, d_model)
#         self.key_embedding = nn.Embedding(char_vocab_size, key_emb_size)


#         # self.encoder = SwipeCurveTransformerEncoderv1(
#         #     input_feats_size, d_model, dim_feedforward,
#         #     num_encoder_layers, num_heads_encoder_1,
#         #     num_heads_encoder_2, dropout, device=device)

#         self.encoder = Conformer(input_dim=d_model,
#                                  num_heads=num_heads_encoder,
#                                  ffn_dim=dim_feedforward,
#                                  num_layers=num_encoder_layers,
#                                  depthwise_conv_kernel_size=32,
#                                  dropout=dropout)
        
#         self.char_pos_encoder = PositionalEncoding(
#             d_model, max_out_seq_len, device=device)
        
#         self.key_pos_encoder = PositionalEncoding(
#             key_emb_size, max_curves_seq_len, device=device)
        
#         n_classes = char_vocab_size - 2  # <sos> and <pad> are not predicted
#         self.decoder = SwipeCurveTransformerDecoderv1(
#             d_model, n_classes, num_heads_decoder,
#             num_decoder_layers, dim_feedforward, dropout, activation, device=device)

#         self.mask = self._get_mask(max_out_seq_len).to(device=device)


#     # def encode(self, x, kb_tokens, x_pad_mask):
#     #     kb_k_emb = self.key_embedding(kb_tokens)  # keyboard key
#     #     kb_k_emb = self.key_embedding_dropout(kb_k_emb)
#     #     kb_k_emb = self.key_pos_encoder(kb_k_emb)
#     #     x = torch.cat((x, kb_k_emb), dim = -1)
#     #     x = self.encoder(x, src_key_padding_mask = x_pad_mask)
#     #     return x
    
#     # def decode(self, x_encoded, y, x_pad_mask, y_pad_mask):
#     #     y = self.char_embedding(y)
#     #     y = self.char_embedding_dropout(y)
#     #     y = self.char_pos_encoder(y)
#     #     mask = self._get_mask(len(y)).to(device=self.device)
#     #     y = self.decoder(y, x_encoded, mask, x_pad_mask, y_pad_mask)
#     #     return y

#     # def forward(self, x, kb_tokens, y, x_pad_mask, y_pad_mask):
#     #     x_encoded = self.encode(x, kb_tokens, x_pad_mask)
#     #     return self.decode(x_encoded, y, x_pad_mask, y_pad_mask)







#############################################



class TransformerEncoderTransformerDecoderWithPos(nn.Module):

    def _get_mask(self, max_seq_len: int):
        """
        Returns a mask for the decoder transformer.
        """
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def __init__(self, 
                 n_coord_feats,  # 6
                 key_emb_size,  # 122
                 char_vocab_size: int, 
                 num_encoder_layers,  # 6
                 num_decoder_layers,  # 6
                 dim_feedforward,  # 128 у меня. Но надо бы 2048 
                 dropout:float,
                 char_embedding_dropout: float,
                 key_embedding_dropout: float,
                 num_heads_encoder,  # 4 
                 num_heads_decoder,  # 4
                 max_out_seq_len: int,
                 max_curves_seq_len: int,
                 activation: Callable = F.relu,
                 device = None,) -> None:
        super().__init__()

        self.device = torch.device(
            device 
            or 'cuda' if torch.cuda.is_available() else 'cpu')
        
        d_model = n_coord_feats + key_emb_size

        self.char_embedding_dropout = nn.Dropout(char_embedding_dropout)
        self.key_embedding_dropout = nn.Dropout(key_embedding_dropout)
        
        self.char_embedding = nn.Embedding(char_vocab_size, d_model)
        self.key_embedding = nn.Embedding(char_vocab_size, key_emb_size)


        # self.encoder = SwipeCurveTransformerEncoderv1(
        #     input_feats_size, d_model, dim_feedforward,
        #     num_encoder_layers, num_heads_encoder_1,
        #     num_heads_encoder_2, dropout, device=device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads_encoder,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device=device)
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        self.char_pos_encoder = PositionalEncoding(
            d_model, max_out_seq_len, device=device)
        
        self.key_pos_encoder = PositionalEncoding(
            key_emb_size, max_curves_seq_len, device=device)
        
        n_classes = char_vocab_size - 2  # <sos> and <pad> are not predicted
        self.decoder = SwipeCurveTransformerDecoderv1(
            d_model, n_classes, num_heads_decoder,
            num_decoder_layers, dim_feedforward, dropout, activation, device=device)

        self.mask = self._get_mask(max_out_seq_len).to(device=device)


    def encode(self, x, kb_tokens, x_pad_mask):
        kb_k_emb = self.key_embedding(kb_tokens)  # keyboard key
        kb_k_emb = self.key_embedding_dropout(kb_k_emb)
        kb_k_emb = self.key_pos_encoder(kb_k_emb)
        x = torch.cat((x, kb_k_emb), dim = -1)
        x = self.encoder(x, src_key_padding_mask = x_pad_mask)
        return x
    
    def decode(self, x_encoded, y, x_pad_mask, y_pad_mask):
        y = self.char_embedding(y)
        y = self.char_embedding_dropout(y)
        y = self.char_pos_encoder(y)
        mask = self._get_mask(len(y)).to(device=self.device)
        y = self.decoder(y, x_encoded, mask, x_pad_mask, y_pad_mask)
        return y

    def forward(self, x, kb_tokens, y, x_pad_mask, y_pad_mask):
        x_encoded = self.encode(x, kb_tokens, x_pad_mask)
        return self.decode(x_encoded, y, x_pad_mask, y_pad_mask)





def get_transformer_bigger_model(device = None, weights_path = None):
    CHAR_VOCAB_SIZE = 37  # = len(word_char_tokenizer.char_to_idx)
    MAX_CURVES_SEQ_LEN = 299
    MAX_OUT_SEQ_LEN = 35  # word_char_tokenizer.max_word_len - 1

    model = TransformerEncoderTransformerDecoderWithPos(
        n_coord_feats=6,
        key_emb_size=122,
        char_vocab_size=CHAR_VOCAB_SIZE,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=128,
        num_heads_encoder=4,
        num_heads_decoder=4,
        dropout=0.1,
        char_embedding_dropout=0.1,
        key_embedding_dropout=0.1,
        max_out_seq_len=MAX_OUT_SEQ_LEN,
        max_curves_seq_len=MAX_CURVES_SEQ_LEN,
        device = device)

    if weights_path:
        model.load_state_dict(
            torch.load(weights_path,
                    map_location = device))
    
    model = model.to(device)
        
    model = model.eval()

    return model




def get_transformer_bb_model(device = None, weights_path = None):
    CHAR_VOCAB_SIZE = 37  # = len(word_char_tokenizer.char_to_idx)
    MAX_CURVES_SEQ_LEN = 299
    MAX_OUT_SEQ_LEN = 35  # word_char_tokenizer.max_word_len - 1

    model = TransformerEncoderTransformerDecoderWithPos(
    n_coord_feats=6,
    key_emb_size=506,
    char_vocab_size=CHAR_VOCAB_SIZE,
    num_encoder_layers=8,
    num_decoder_layers=8,
    dim_feedforward=2048,
    num_heads_encoder=8,
    num_heads_decoder=8,
    dropout=0.1,
    char_embedding_dropout=0.1,
    key_embedding_dropout=0.1,
    max_out_seq_len=MAX_OUT_SEQ_LEN,
    max_curves_seq_len=MAX_CURVES_SEQ_LEN,
    device = device)

    if weights_path:
        model.load_state_dict(
            torch.load(weights_path,
                    map_location = device))
    
    model = model.to(device)
        
    model = model.eval()

    return model





###############################################################################
###############################################################################




# Новые модели будут представлены так:
# Функция - model_getter (значение в словаре MODEL_GETTERS_DICT_V2)
# будет иметь среди прочих аргументы encoder_input_dim и embedding_model_ctor: Callable. 
# Все наследники класса EmbeddingModel будут в __init__ иметь аргумент out_dim.
# В __init__ model_getter'а будет создаваться embedding_model_ctor c аргументом...




# class EmbeddingModel(nn.Module):
#     def __init__(self, out_dim):
#         super().__init__()
#         self.out_dim = out_dim

#     def forward(self, x):
#         raise NotImplementedError("Method forward() is not implemented.")



class WeightedSumEmbedding(nn.Module):
    """
    Computes embedding as a weighted sum of embeddings

    Is used as a swipe dot embedding: the embedding is
    a weighted sum of embeddings of all key on keyboard
    """
    def __init__(self, n_elements, dim) -> None:
        """
        Arguments:
        ----------
        
        """
        super().__init__()
        # Using linear is same as using Linear(Embedding.get_matrix * weights)
        # Embedding is same as Linear(one_hot)
        # weights is generalization of one-hot
        # Linear(weights) = Linear(Linear(one_hot) * weights) 
        self.embeddings = nn.Linear(n_elements, dim)

    def forward(self, weights):
        return self.embeddings(weights)


class WeightsSumEmbeddingWithPos(WeightedSumEmbedding):
    def __init__(self, n_elements, dim, max_len, device) -> None:
        super().__init__(n_elements, dim)
        self.pos_encoder = PositionalEncoding(dim, max_len, device)

    def forward(self, weights):
        emb = super().forward(weights)
        emb = self.pos_encoder(emb)
        return emb



class SeparateTrajAndWEightedEmbeddingWithPos(nn.Module):
    def __init__(self, n_keys, key_emb_size, max_len, device, dropout = 0.1) -> None:
        super().__init__()
        self.weighted_sum_emb = WeightsSumEmbeddingWithPos(n_keys, key_emb_size, max_len, device)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_tuple: Tuple[torch.Tensor, torch.Tensor]):
        traj_feats, kb_key_weights = input_tuple
        kb_k_emb = self.weighted_sum_emb(kb_key_weights)
        kb_k_emb = self.dropout(kb_k_emb)
        x = torch.cat((traj_feats, kb_k_emb), dim = -1)
        return x
    

#############
# !!!!!!!!!!!!!!! It may be a bad practice. This
# usage of args and kwargs prevents contracts
# and any checks on input validity.


class EncoderDecoderAbstract(nn.Module):
    def __init__(self, enc_in_emb_model, dec_in_emb_model, encoder, decoder, out):
        super().__init__()
        self.enc_in_emb_model = enc_in_emb_model
        self.dec_in_emb_model = dec_in_emb_model
        self.encoder = encoder
        self.decoder = decoder
        self.out = out  # linear

    # x can be a tuple (ex. traj_feats, kb_tokens) or a single tensor
    # (ex. just kb_tokens).
    def encode(self, x, *encoder_args, **encoder_kwargs):
        x = self.enc_in_emb_model(x)
        return self.encoder(x, *encoder_args, **encoder_kwargs)
    
    def decode(self, x_encoded, y, *decoder_args, **decoder_kwargs):
        y = self.dec_in_emb_model(y)
        dec_out = self.decoder(x_encoded, y, *decoder_args, **decoder_kwargs)
        return self.out(dec_out)
    
    def forward(self, x, y, 
                encoder_args: list = None, 
                encoder_kwargs: dict = None,
                decoder_args: list = None,
                decoder_kwargs: dict = None):
        if encoder_args is None:
            encoder_args = []
        if decoder_args is None:
            decoder_args = []
        if encoder_kwargs is None:
            encoder_kwargs = {}
        if decoder_kwargs is None:
            decoder_kwargs = {}

        x_encoded = self.encode(x, *encoder_args, **encoder_kwargs)
        return self.decode(x_encoded, y, *decoder_args, **decoder_kwargs)




class EncoderDecoderAbstractLegacyDSFormat(nn.Module):
    def __init__(self, enc_in_emb_model, dec_in_emb_model, encoder, decoder, out):
        super().__init__()
        self.enc_in_emb_model = enc_in_emb_model
        self.dec_in_emb_model = dec_in_emb_model
        self.encoder = encoder
        self.decoder = decoder
        self.out = out  # linear

    # x can be a tuple (ex. traj_feats, kb_tokens) or a single tensor
    # (ex. just kb_tokens).
    def encode(self, traj_feats, kb_tokens, *encoder_args, **encoder_kwargs):
        x = self.enc_in_emb_model((traj_feats, kb_tokens))
        return self.encoder(x, *encoder_args, **encoder_kwargs)
    
    def decode(self, x_encoded, y, *decoder_args, **decoder_kwargs):
        y = self.dec_in_emb_model(y)
        dec_out = self.decoder(x_encoded, y, *decoder_args, **decoder_kwargs)
        return self.out(dec_out)
    
    def forward(self, traj_feats, kb_tokens, y, 
                encoder_args: list = None, 
                encoder_kwargs: dict = None,
                decoder_args: list = None,
                decoder_kwargs: dict = None):
        if encoder_args is None:
            encoder_args = []
        if decoder_args is None:
            decoder_args = []
        if encoder_kwargs is None:
            encoder_kwargs = {}
        if decoder_kwargs is None:
            decoder_kwargs = {}

        x_encoded = self.encode(traj_feats, kb_tokens, *encoder_args, **encoder_kwargs)
        return self.decode(x_encoded, y, *decoder_args, **decoder_kwargs)






def get_transformer_bigger_weighted(device = None, weights_path = None, legacy_ds: bool = True):
    CHAR_VOCAB_SIZE = 37  # = len(word_char_tokenizer.char_to_idx)
    MAX_CURVES_SEQ_LEN = 299
    MAX_OUT_SEQ_LEN = 35  # word_char_tokenizer.max_word_len - 1

    device = torch.device(
        device 
        or 'cuda' if torch.cuda.is_available() else 'cpu')

    n_coord_feats = 6
    key_emb_size = 122
    d_model = n_coord_feats + key_emb_size

    input_embedding_dropout = 0.1

    n_word_chars = CHAR_VOCAB_SIZE
    
    # Actually, n_keys != n_word_chars. n_keys = 36. 
    # It's legacy. It should not affect the model though.
    n_keys = CHAR_VOCAB_SIZE  

    input_embedding = SeparateTrajAndWEightedEmbeddingWithPos(
        n_keys=n_keys, key_emb_size=key_emb_size, 
        max_len=MAX_CURVES_SEQ_LEN, device = device, dropout=input_embedding_dropout)


    transformer_dropout = 0.1

    transformer = nn.Transformer(
        d_model,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=128,
        dropout=transformer_dropout,
        device = device
    )
    

    word_char_embedding = nn.Embedding(n_word_chars, d_model)
    word_char_emb_dropout_val = 0.1
    word_char_emb_dropout = nn.Dropout(word_char_emb_dropout_val)
    word_char_pos_encoder = PositionalEncoding(d_model, MAX_OUT_SEQ_LEN, device=device)

    word_char_embedding_model = nn.Sequential(
        word_char_embedding,
        word_char_emb_dropout,
        word_char_pos_encoder
    )

    n_classes = CHAR_VOCAB_SIZE - 2  # <sos> and <pad> are not predicted


    out = nn.Linear(d_model, n_classes, device = device)


    encoder_decoder_ctor = (
        EncoderDecoderAbstractLegacyDSFormat if legacy_ds 
        else EncoderDecoderAbstract
    )

    model = encoder_decoder_ctor(
        input_embedding, word_char_embedding_model, 
        transformer.encoder, transformer.decoder, out)
    
    if weights_path:
        model.load_state_dict(
            torch.load(weights_path,
                    map_location = device))
        

    model = model.to(device)

    model = model.eval()

    return model





###############################################################################
###############################################################################





MODEL_GETTERS_DICT = {
    "m1": get_m1_model,
    "m1_bigger": get_m1_bigger_model,
    "m1_smaller": get_m1_smaller_model,

    "transformer_m1_bigger": get_transformer_bigger_model,
    "transformer_bb_model": get_transformer_bb_model,

    "weighted_transformer_bigger": get_transformer_bigger_weighted,
}
