import torch

def truncate_padding(seq, mask):
    max_curve_len = int(torch.max(torch.sum(~mask, dim = 1)))
    seq = seq[:, :max_curve_len]
    mask = mask[:, :max_curve_len]
    return seq, mask


def turncate_traj_batch(xyt, kb_tokens, traj_pad_mask):
    max_curve_len = int(torch.max(torch.sum(~traj_pad_mask, dim = 1)))
    xyt = xyt[:, :max_curve_len]
    kb_tokens = kb_tokens[:, :max_curve_len]
    traj_pad_mask = traj_pad_mask[:, :max_curve_len]
    return xyt, kb_tokens, traj_pad_mask



def prepare_batch_with_pad_truncation(x, y, device):
    (xyt, kb_tokens, dec_in_char_seq, traj_pad_mask, word_pad_mask), dec_out_char_seq = x, y

    xyt, traj_pad_mask = truncate_padding(xyt, traj_pad_mask)
    kb_tokens, traj_pad_mask = truncate_padding(kb_tokens, traj_pad_mask)
    dec_in_char_seq, word_pad_mask = truncate_padding(kb_tokens, traj_pad_mask)
    dec_out_char_seq, word_pad_mask = truncate_padding(kb_tokens, traj_pad_mask)

    # print(max_curve_len)

    xyt = xyt.transpose_(0, 1).to(device)  # (curves_seq_len, batch_size, n_coord_feats)
    kb_tokens = kb_tokens.transpose_(0, 1).to(device) # (curves_seq_len, batch_size)
    dec_in_char_seq = dec_in_char_seq.transpose_(0, 1).to(device)  # (chars_seq_len - 1, batch_size)
    dec_out_char_seq = dec_out_char_seq.transpose_(0, 1).to(device)  # (chars_seq_len - 1, batch_size)

    traj_pad_mask = traj_pad_mask.to(device)  # (batch_size, max_curve_len)
    # traj_pad_mask = torch.zeros_like(kb_tokens, dtype = torch.bool).transpose_(0, 1).to(device)
    word_pad_mask = word_pad_mask.to(device)  # (batch_size, chars_seq_len - 1)

    return (xyt, kb_tokens, dec_in_char_seq, traj_pad_mask, word_pad_mask), dec_out_char_seq


def prepare_batch_without_truncation(x, y, device):
    (xyt, kb_tokens, dec_in_char_seq, traj_pad_mask, word_pad_mask), dec_out_char_seq = x, y
    
    if xyt is not None:
        xyt = xyt.transpose_(0, 1).to(device)  # (curves_seq_len, batch_size, n_coord_feats)
    if kb_tokens is not None:
        kb_tokens = kb_tokens.transpose_(0, 1).to(device) # (curves_seq_len, batch_size)
    if dec_in_char_seq is not None:
        dec_in_char_seq = dec_in_char_seq.transpose_(0, 1).to(device)  # (chars_seq_len - 1, batch_size)
    if dec_out_char_seq is not None:
        dec_out_char_seq = dec_out_char_seq.transpose_(0, 1).to(device)  # (chars_seq_len - 1, batch_size)

    if traj_pad_mask is not None:
        traj_pad_mask = traj_pad_mask.to(device)  # (batch_size, curves_seq_len)
    if word_pad_mask is not None:
        word_pad_mask = word_pad_mask.to(device)  # (batch_size, chars_seq_len - 1)

    return (xyt, kb_tokens, dec_in_char_seq, traj_pad_mask, word_pad_mask), dec_out_char_seq


def prepare_batch(x, y, device):
    return prepare_batch_without_truncation(x, y, device)