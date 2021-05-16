from model import *
config = {
    'lr': 1e-5,
    'optimizer': "Adam",
    'batch_size': 4,
    'gradient_clip_val': 1.0,
    'num_workers': 4,
    'pin_memory': True,
    'subset': 0.1,
    'nhead': 1,
    'dropout': 0.,
    'max_len': 277,
    'train_trans': {
      'Resize': {
        'width': 320,
        'height': 160,
      }
    },
    'val_trans': {
      'Resize': {
        'width': 320,
        'height': 160,
      }
    },
    'gpus': 1,
    'precision': 16,
    'max_epochs': 5
}
from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(
    dirpath='ckp', 
    filename=f'MeiYun-{{val_loss:.4f}}',
    save_top_k=1, 
    monitor='val_loss', 
    mode='min'
)
dm = DataModule(
    data_file = 'train_labels_tokenized.csv', 
    path=Path(''), 
    **config
)
model = MeiYun(config)
trainer = pl.Trainer(
    gpus=config['gpus'],
    precision=config['precision'],
    max_epochs=config['max_epochs'],
    gradient_clip_val=config['gradient_clip_val'],
    callbacks=[checkpoint]
)
if __name__ == '__main__':
    trainer.fit(model, dm)