
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import clip
import tensorflow as tf
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from os.path import join
import clip
from PIL import Image

#################################
### adapted from https://github.com/christophschuhmann/improved-aesthetic-predictor 
#################################
def init_laion(device = "cpu"):
    print("loading laion model")
    aes_model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    s = torch.load("models/sac+logos+ava1-l14-linearMSE.pth", map_location=device)   
    aes_model.load_state_dict(s)
    aes_model.to(device)
    aes_model.eval()
    print("loading clip model")
    vit_clip, clip_preprocess = clip.load("models/ViT-L-14.pt", device=device)   
    print("done!")
    return aes_model, vit_clip, clip_preprocess


# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalizer(a, axis=-1, order=2):
	l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
	l2[l2 == 0] = 1
	return a / np.expand_dims(l2, axis)