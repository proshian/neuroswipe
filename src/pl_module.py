import torch
from lightning import LightningModule
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
import torchmetrics


from model import MODEL_GETTERS_DICT
from metrics import get_word_level_accuracy

# ! Make sure:
# * Add metrics

#! Maybe store:
# * batch_size
# * early_stopping_patience

#! Maybe:
# * Checpointing by condition: if model improved on val_loss and val_loss < max_val_loss_to_save


class LitNeuroswipeModel(LightningModule):
    def __init__(self, model_name: str, criterion, 
                 num_classes: int,
                 train_batch_size: int = None,
                 criterion_ignore_index: int = -100, optim_kwargs = None, 
                 optimizer_ctor=None, lr_scheduler_ctor=None, label_smoothing=0.0,
                 ) -> None:
        super().__init__()

        self.optim_kwargs = optim_kwargs or dict(lr=1e-4, weight_decay=0)
        
        self.model_name = model_name
        self.train_batch_size = train_batch_size
        self.label_smoothing = label_smoothing
        self.criterion_ignore_index = criterion_ignore_index

        self.optimizer_ctor = optimizer_ctor
        self.lr_scheduler_ctor = lr_scheduler_ctor

        self.model = MODEL_GETTERS_DICT[model_name]()
        self.criterion = criterion
        
        self.train_token_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=criterion_ignore_index)
        self.val_token_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=criterion_ignore_index)
        self.train_token_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes, ignore_index=criterion_ignore_index)
        self.val_token_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes, ignore_index=criterion_ignore_index)

    def forward(self, traj_feats, kb_tokens, y, x_pad_mask, y_pad_mask):
        return self.model.forward(traj_feats, kb_tokens, y, x_pad_mask, y_pad_mask)
    
    def configure_optimizers(self):
        optimizer = self.optimizer_ctor(self.parameters(), **self.optim_kwargs)
        
        optimizers_configuration = {'optimizer': optimizer}

        if self.lr_scheduler_ctor:
            lr_scheduler = self.lr_scheduler_ctor(optimizer)
            optimizers_configuration['lr_scheduler'] = lr_scheduler
            optimizers_configuration['monitor'] = 'val_loss'

        return optimizers_configuration


    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        
        batch_size = batch_y.shape[-1]

        # batch_x, batch_y = move_all_to_device(batch_x, batch_y, self.device)

        # * batch_x is a Tuple of (curve_traj_feats, curve_kb_tokens,
        #   decoder_in, curve_pad_mask, dec_seq_pad_mask).
        # * batch_y is decoder_out.
        
        # preds.shape = (chars_seq_len, batch_size, n_classes)
        
        curve_traj_feats, curve_kb_tokens, decoder_in, curve_pad_mask, dec_seq_pad_mask = batch_x

        pred = self.forward(*batch_x)
        
        loss = self.criterion(pred, batch_y, ignore_index=self.criterion_ignore_index,
                              label_smoothing=self.label_smoothing)
        
        
        argmax_pred = torch.argmax(pred, dim=2)
        wl_acccuracy = get_word_level_accuracy(
            argmax_pred.T, batch_y.T, pad_token = self.criterion_ignore_index, mask = dec_seq_pad_mask)
        
        
        flat_y = batch_y.reshape(-1)
        n_classes = pred.shape[-1]
        flat_preds = pred.reshape(-1, n_classes)
        
        self.train_token_acc(flat_preds, flat_y)
        self.log('train_token_level_accuracy', self.train_token_acc, on_step=True, on_epoch=False)
        
        self.train_token_f1(flat_preds, flat_y)
        self.log('train_token_level_f1', self.train_token_f1, on_step=True, on_epoch=False)
        
        
        self.log("train_word_level_accuracy", wl_acccuracy, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size = batch_size)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size = batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_size = batch_y.shape[-1]
        # batch_x, batch_y = move_all_to_device(batch_x, batch_y, self.device)
        curve_traj_feats, curve_kb_tokens, ecoder_in, curve_pad_mask, dec_seq_pad_mask = batch_x
        pred = self.forward(*batch_x)
        loss = self.criterion(pred, batch_y, ignore_index=self.criterion_ignore_index,
                              label_smoothing=self.label_smoothing)
        argmax_pred = torch.argmax(pred, dim=2)
        wl_acccuracy = get_word_level_accuracy(
            argmax_pred.T, batch_y.T, pad_token = self.criterion_ignore_index, mask = dec_seq_pad_mask)
        
        
        flat_y = batch_y.reshape(-1)
        n_classes = pred.shape[-1]
        flat_preds = pred.reshape(-1, n_classes)
        
        
        self.val_token_acc(flat_preds, flat_y)
        self.log('val_token_level_accuracy', self.train_token_acc, on_step=False, on_epoch=True)
        
        self.val_token_f1(flat_preds, flat_y)
        self.log('val_token_level_f1', self.train_token_f1, on_step=False, on_epoch=True)
        
        
        
        self.log("val_word_level_accuracy", wl_acccuracy, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size = batch_size)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, 
                 logger=True, batch_size = batch_size)
        return loss
    