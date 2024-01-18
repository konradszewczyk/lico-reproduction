import pytorch_lightning as pl
import torch
import torch.nn as nn
import clip
from typing import List

from models.LICO_loss import LICOLoss
from models.cosine_lr_scheduler import CosineLRScheduler
from training_utils import accuracy


class LICOModel(pl.LightningModule):
    def __init__(self, image_model, target_names: torch.tensor, alpha, beta, train_mm_temp):
        """
        :param image_model: the image model to use for feature extraction nad classification
            has to implement features_forward() method that gives features and logits
        :param target_names: a tensor where at index i there is tokenized name of the ith class
        """
        super().__init__()

        self.image_model = image_model
        self.text_model, _ = clip.load("ViT-B/32")
        self.register_buffer('target_names', target_names)

        self.clip_tokenizer_dim = 77
        self.clip_text_dim = 512

        # todo: im not sure if this one acts independently on each token or not
        self.projection_mlp = nn.Sequential(
            nn.Linear(self.clip_text_dim, 768, dtype=torch.float16),
            nn.ReLU(),
            nn.Linear(768, self.image_model.get_feature_dim(), dtype=torch.float16)
        )  # h

        self.criterion = LICOLoss(alpha, beta, reduction='mean', train_mm_temperature=train_mm_temp)
        
        # self.criterion = LICOLoss(reduction='mean')
        # hyperparameter
        self.M = 10

        # how can the prompts be learnable? they are discrete, they are text prompts!
        # learnable_prompt = torch.zeros((self.M, self.tokenizer_dim), dtype=torch.long)
        # self.register_parameter('learnable_prompt', nn.Parameter(learnable_prompt))

        # self.save_hyperparameters()

    def get_learnable_prompts(self):
        return self.learnable_prompt

    def training_step(self, batch, batch_idx):
        """
        :param batch: tuple of (images, target)
        """
        images, target = batch
        img_logits, img_features = self.image_model.features_forward(images)
        # full_prompt = torch.cat([
        #     torch.broadcast_to(self.get_learnable_prompts(), (target.shape[0], self.M, self.tokenizer_dim)),
        #     self.target_names[target]
        # ], dim=1)
        full_prompt = self.target_names[target].view(target.shape[0], self.clip_tokenizer_dim)
        with torch.no_grad():
            text_features = self.text_model.encode_text(full_prompt)

        # Apply projection MLP (independently on each token embedding)
        projected_text_features = self.projection_mlp(text_features)

        # 1 here is number of tokens in the prompt
        projected_text_features = projected_text_features.view(projected_text_features.shape[0], 1, projected_text_features.shape[1])
        # if torch.isnan(img_features).any():
            # raise ValueError("Image features contain NaN values.")
        # if torch.isnan(projected_text_features).any():
            # raise ValueError("Text features contain NaN values.")
        loss, mm_part, ot_part = self.criterion(img_logits, target, img_features, projected_text_features)
        # if torch.isnan(projected_text_features).any():
            # raise ValueError("Loss output contains NaN values.")

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mm_part', mm_part)
        self.log('train_ot_part', ot_part)
        self.log('mm_loss_temperature', self.criterion.mm_loss.temperature)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch
        img_logits, img_features = self.image_model.features_forward(images)
        full_prompt = self.target_names[target].view(target.shape[0], self.clip_tokenizer_dim)
        with torch.no_grad():
            text_features = self.text_model.encode_text(full_prompt)

        projected_text_features = self.projection_mlp(text_features)

        projected_text_features = projected_text_features.view(projected_text_features.shape[0], 1,
                                                               projected_text_features.shape[1])
        loss, mm_part, ot_part = self.criterion(img_logits, target, img_features, projected_text_features)
        acc1, acc5 = accuracy(img_logits, target, topk=(1, 5))
        self.log('val_loss', loss)
        self.log('val_mm_part', mm_part)
        self.log('val_ot_part', ot_part)
        self.log('val_acc1', acc1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc5', acc5)
        return {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}

    # todo: add test_step

    def configure_optimizers(self):
        # todo: add the trainable prompts to the optimizer
        optimizer = torch.optim.SGD(
            list(self.image_model.parameters()) + list(self.projection_mlp.parameters()) + list(self.criterion.parameters()),
            self.image_model.lr,
            momentum=self.image_model.momentum,
            weight_decay=self.image_model.weight_decay
        )
        lr_scheduler = CosineLRScheduler(optimizer, T_max=self.image_model.total_steps)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def on_save_checkpoint(self, checkpoint):
        # Identify the keys associated with the text_model, which we dont want to save
        clip_keys = [key for key in checkpoint['state_dict'].keys() if key.startswith('text_model')]

        # Remove these keys from the checkpoint
        for key in clip_keys:
            del checkpoint['state_dict'][key]

    def on_load_checkpoint(self, checkpoint):
        self.text_model, _ = clip.load("ViT-B/32")


def tokenize_targets(targets: List[str]) -> torch.tensor:
    """
    :param targets: list of strings with class names
    :return: tensor of tokenized class names
    """
    tokenized = []
    for i, target in enumerate(targets):
        tokenized.append(clip.tokenize(target))
    return torch.stack(tokenized).to(dtype=torch.long)