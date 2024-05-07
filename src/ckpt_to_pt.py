import argparse

import torch


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def ckpt_to_torch_state(ckpt):
    return {remove_prefix(k, 'model.'): v for k, v in ckpt['state_dict'].items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    state_dict = ckpt_to_torch_state(ckpt)
    torch.save(state_dict, args.out_path)





### Alternative script below

# MODEL_NAME = 
# CPT_PATH = 

# import torch

# from model import MODEL_GETTERS_DICT
# from pl_module import LitNeuroswipeModel



# def cross_entropy_with_reshape(pred, target, ignore_index=-100, label_smoothing=0.0):
#     """
#     pred - BatchSize x TargetLen x VocabSize
#     target - BatchSize x TargetLen
#     """
#     pred_flat = pred.view(-1, pred.shape[-1])  # BatchSize*TargetLen x VocabSize
#     target_flat = target.reshape(-1)  # BatchSize*TargetLen
#     return F.cross_entropy(pred_flat,
#                            target_flat,
#                            ignore_index=ignore_index,
#                            label_smoothing=label_smoothing)


# pl_model = LitNeuroswipeModel.load_from_checkpoint(
#     CPT_PATH, model_name = MODEL_NAME, 
#     criterion = cross_entropy_with_reshape, 
#     num_classes = 35)

# model = pl_model.model

# torch.save(model.state_dict(), CPT_PATH + ".pt")