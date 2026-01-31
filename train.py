import os 
import pytorch_lightning as pl  
from pytorch_lightning.callbacks import ModelCheckpoint  

from comer.datamodule import CROHMEDatamodule  
from comer.lit_comer import LitCoMER

import torch
torch.set_float32_matmul_precision('high')

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Khởi tạo model  
model = LitCoMER(  
    d_model=256,  
    growth_rate=24,  
    num_layers=16,  
    nhead=8,  
    num_decoder_layers=3,  
    dim_feedforward=1024,  
    dropout=0.3,  
    dc=32,  
    cross_coverage=True,  
    self_coverage=True,  
    beam_size=10,  
    max_len=100,  
    alpha=1.0,  
    early_stopping=True,  
    temperature=1.0,  
    learning_rate=0.07,  
    patience=20  
)

# Khởi tạo datamodule  
datamodule = CROHMEDatamodule(  
    zipfile_path="comer_data/data.zip",  
    test_year="val_300",  
    train_batch_size=64,  
    eval_batch_size=16,
    num_workers=16,  
    scale_aug=True  
)

# Tạo checkpoint callback  
checkpoint_callback = ModelCheckpoint(  
    save_top_k=30,  
    monitor='val_ExpRate',
    mode='max',  
    filename='{epoch}-{step}-{val_ExpRate:.4f}'  
) 

trainer = pl.Trainer(   
    max_epochs=300, 
    check_val_every_n_epoch=10,
    gradient_clip_val=1.0,
    deterministic=False ,
    precision="bf16-mixed",
    callbacks=[checkpoint_callback]
)

trainer.fit(model, 
            datamodule,
            # ckpt_path="lightning_logs/version_49/checkpoints/epoch=79-step=20960-val_ExpRate=0.3556.ckpt"
)