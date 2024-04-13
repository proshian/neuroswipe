import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers


from model import MODEL_GETTERS_DICT

# ! Make sure:
# * Add metrics

#! Maybe store:
# * batch_size
# * early_stopping_patience

#! Maybe:
# * Checpointing by condition: if model improved on val_loss and val_loss < max_val_loss_to_save


class LitNeuroswipeModel(L.LightningModule):
    def __init__(self, model_name: str, criterion, 
                 batch_size: int = None,
                 criterion_ignore_index: int = -100, optim_kwargs = None, 
                 optimizer_ctor=None, lr_scheduler_ctor=None, label_smoothing=0.0
                 ) -> None:
        super().__init__()

        self.optim_kwargs = optim_kwargs or dict(lr=1e-4, l2_reg_alpha=0)
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.criterion_ignore_index = criterion_ignore_index

        self.optmizer_ctor = optimizer_ctor
        self.lr_scheduler_ctor = lr_scheduler_ctor

        self.model = MODEL_GETTERS_DICT[model_name]()
        self.criterion = criterion

    def forward(self, x, kb_tokens, y, x_pad_mask, y_pad_mask):
        x_encoded = self.model.encode(x, kb_tokens, x_pad_mask)
        return self.model.decode(x_encoded, y, x_pad_mask, y_pad_mask)
    
    def configure_optimizers(self):
        optimizer = self.optimizer_ctor(self.parameters(), **self.optim_kwargs)

        if self.lr_scheduler_ctor:
            lr_scheduler = self.lr_scheduler_ctor(optimizer)
            return optimizer, lr_scheduler

        return optimizer


    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        # batch_x, batch_y = move_all_to_device(batch_x, batch_y, self.device)

        # * batch_x is a Tuple of (curve_traj_feats, curve_kb_tokens,
        #   decoder_in, curve_pad_mask, dec_seq_pad_mask).
        # * batch_y is decoder_out.

        pred = self.forward(*batch_x)

        loss = self.criterion(pred, batch_y, ignore_index=self.criterion_ignore_index,
                              label_smoothing=self.label_smoothing)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        # batch_x, batch_y = move_all_to_device(batch_x, batch_y, self.device)
        pred = self.forward(batch_x, batch_y)
        loss = self.criterion(pred, batch_y, ignore_index=self.criterion_ignore_index,
                              label_smoothing=self.label_smoothing)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")

early_stopping_cb = EarlyStopping(
    monitor='val_loss', mode = 'min', patience=20)

model_checkpoint_cb = ModelCheckpoint(
    monitor='val_loss', mode = 'min', save_top_k=50, 
    filename='{epoch}-{val_loss:.2f}')

trainer = Trainer(
    num_sanity_val_steps=0,
    gpus=1,
    max_epochs=1000,
    callbacks=[early_stopping_cb, model_checkpoint_cb],
    logger=tb_logger
)

# trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")