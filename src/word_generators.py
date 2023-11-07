import torch

from utils import prepare_batch


class GreedyGenerator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.eos_token_id = tokenizer.char_to_idx['<eos>'] 

    def __call__(self, xyt, kb_tokens, traj_pad_mask, max_steps_n=35):
        tokens = [self.tokenizer.char_to_idx['<sos>']]

        # We don't have to put everything to device because it's done in prepare_batch.

        with torch.no_grad():
            for _ in range(max_steps_n):
                dec_in_char_seq = torch.tensor(tokens).to(self.device)
                word_pad_mask = torch.zeros_like(dec_in_char_seq, dtype=torch.bool, device=self.device)
                # dummy_y is any tensor with n_dims = 2 (chars_seq_len - 1, batch_size).
                dummy_y = torch.tensor([[1]])
                x = [xyt, kb_tokens, dec_in_char_seq, traj_pad_mask, word_pad_mask]
                x = [el.unsqueeze(0) for el in x]
                model_input, dummy_y = prepare_batch(x, dummy_y, self.device)
                best_next_token = self.model(*model_input).transpose_(0, 1)
                best_next_token = best_next_token[0, -1].argmax()  # batch_i = 0, decoder_out_onehot_vector_seq_i = -1 
                if best_next_token == self.eos_token_id:
                    break

                tokens.append(int(best_next_token))
        
        return self.tokenizer.decode(tokens[1:])