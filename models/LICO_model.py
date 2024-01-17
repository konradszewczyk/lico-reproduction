import pytorch_lightning as pl
import torch
import torch.nn as nn
import clip
from typing import List

from models.LICO_loss import LICOLoss
from training_utils import accuracy


class LICOModel(pl.LightningModule):
    def __init__(self, image_model, target_names: torch.tensor):
        """
        :param image_model: the image model to use for feature extraction nad classification
            has to implement features_forward() method that gives features and logits
        :param target_names: a tensor where at index i there is tokenized name of the ith class
        """
        super().__init__()

        self.image_model = image_model
        # todo: remove that from saving and from parameters in general
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

        self.criterion = LICOLoss(reduction='mean')
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
        img_features, img_logits = self.image_model.features_forward(images)
        # full_prompt = torch.cat([
        #     torch.broadcast_to(self.get_learnable_prompts(), (target.shape[0], self.M, self.tokenizer_dim)),
        #     self.target_names[target]
        # ], dim=1)
        full_prompt = self.target_names[target].view(target.shape[0], self.clip_tokenizer_dim)
        with torch.no_grad():
            text_features = self.text_model.encode_text(full_prompt)

        # Apply projection MLP (independently on each token embedding)
        projected_text_features = self.projection_mlp(text_features)

        # todo: i have no idea what the num_channels and prompt_size are for the loss
        # so for now its 1
        img_features = img_features.view(img_features.shape[0], 1, img_features.shape[1])
        projected_text_features = projected_text_features.view(projected_text_features.shape[0], 1, projected_text_features.shape[1])
        loss = self.criterion(img_logits, target, img_features, projected_text_features)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch
        img_features, img_logits = self.image_model.features_forward(images)
        full_prompt = self.target_names[target].view(target.shape[0], self.clip_tokenizer_dim)
        with torch.no_grad():
            text_features = self.text_model.encode_text(full_prompt)

        projected_text_features = self.projection_mlp(text_features)

        img_features = img_features.view(img_features.shape[0], 1, img_features.shape[1])
        projected_text_features = projected_text_features.view(projected_text_features.shape[0], 1,
                                                               projected_text_features.shape[1])
        loss = self.criterion(img_logits, target, img_features, projected_text_features)
        acc1, acc5 = accuracy(img_logits, target, topk=(1, 5))
        self.log('val_loss', loss)
        self.log('val_acc1', acc1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc5', acc5)
        return {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}

    # todo: add test_step

    def configure_optimizers(self):
        # todo: add the trainable prompts to the optimizer
        optimizer = torch.optim.SGD(
            list(self.image_model.parameters()) + list(self.projection_mlp.parameters()),
            self.image_model.lr,
            momentum=self.image_model.momentum,
            weight_decay=self.image_model.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [lr_scheduler]


def tokenize_targets(targets: List[str]) -> torch.tensor:
    """
    :param targets: list of strings with class names
    :return: tensor of tokenized class names
    """
    tokenized = []
    for i, target in enumerate(targets):
        tokenized.append(clip.tokenize(target))
    return torch.stack(tokenized).to(dtype=torch.long)
