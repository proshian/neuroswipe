import torch

from model import (SwipeCurveTransformerEncoderv1,
                   SwipeCurveTransformerDecoderv1)


def test_encoder_out_shape_and_no_nans(device):
    seq_len = 32
    batch_size = 10
    in_features = 40
    d_model = 128

    encoder = SwipeCurveTransformerEncoderv1(
        input_size=in_features,
        d_model=d_model,
        dim_feedforward=128,
        num_layers=1,
        num_heads_first=2,
        num_heads_other=4,
        dropout=0.1,
        device=device)


    pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    # as if each batch contains 22 actual sequnce elements and 10 padding elements
    pad_mask[:, 10:] = True
    pad_mask = pad_mask.to(device)


    encoded = encoder(torch.rand(seq_len, batch_size, in_features).to(device), pad_mask)
    torch.set_printoptions(threshold=100_000)


    if True in torch.isnan(encoded):
        print(encoded)
        raise ValueError("encoded contains NaNs")
    
    expected_shape = (seq_len, batch_size, d_model)

    if encoded.shape == expected_shape:
        print("Encoder out shape correct")




def test_decoder_out_shape_and_no_nans(device):
    curves_seq_len = 20
    chars_seq_len = 14
    batch_size = 10
    char_emb_size = 32
    n_classes = 5

    decoder = SwipeCurveTransformerDecoderv1(
        char_emb_size=char_emb_size,
        n_classes=n_classes,
        nhead=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        device = device)

    def get_mask(max_seq_len: int):
        """
        Returns a mask for the decoder transformer.
        """
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    x = torch.rand(chars_seq_len, batch_size, char_emb_size).to(device)
    memory = torch.rand(curves_seq_len, batch_size, char_emb_size).to(device)
    tgt_mask = get_mask(chars_seq_len).to(device)
    memory_key_padding_mask = torch.zeros(batch_size, curves_seq_len, dtype=torch.bool).to(device)
    memory_key_padding_mask[:, 15:] = True
    tgt_key_padding_mask = torch.zeros(batch_size, chars_seq_len, dtype=torch.bool).to(device)
    tgt_key_padding_mask[:, 10:] = True


    decoded = decoder(
        x,
        memory,
        tgt_mask=tgt_mask,
        memory_key_padding_mask=memory_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask)

    if True in torch.isnan(decoded):
        print(decoded)
        # raise ValueError("decoded contains NaNs")

    expected_shape = (chars_seq_len, batch_size, n_classes)

    if decoded.shape == expected_shape:
        print("Decoder out shape correct")




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_encoder_out_shape_and_no_nans(device)
    test_decoder_out_shape_and_no_nans(device)