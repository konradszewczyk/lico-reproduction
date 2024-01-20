import pytorch_lightning as pl
import torch
import torch.nn as nn
import clip
from typing import List

from models.LICO_loss import LICOLoss
from models.cosine_lr_scheduler import CosineLRScheduler
from training_utils import accuracy


class LICOModel(pl.LightningModule):
    def __init__(self, image_model, target_names: torch.tensor, alpha: float, beta: float, context_tokens: int,
                 learnable_context: bool, dynamic_context: bool, train_mm_temp: bool):
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

        self.projection_mlp = nn.Sequential(
            nn.Linear(self.clip_text_dim, 768, dtype=torch.float16),
            nn.ReLU(),
            nn.Linear(768, self.image_model.get_feature_dim(), dtype=torch.float16)
        )  # h

        self.criterion = LICOLoss(alpha, beta, reduction='mean', train_mm_temperature=train_mm_temp)

        # hyperparameter
        self.M = 10

        self.dynamic_context = dynamic_context
        self.context_tokens = context_tokens
        self.output_tokens = torch.count_nonzero(target_names, dim=(1, 2)).max() + self.context_tokens
        self.learnable_prompts = nn.Parameter(
            torch.randn(1, self.context_tokens, self.clip_text_dim).type(self.text_model.dtype),
            requires_grad=learnable_context
        )

        # self.save_hyperparameters()

    def get_learnable_prompts(self):
        return self.learnable_prompts

    def encode_text_tokens(self, target):
        label_prompt = self.target_names[target].view(target.shape[0], self.clip_tokenizer_dim)
        with torch.no_grad():
            label_features = self.text_model.token_embedding(label_prompt).type(self.text_model.dtype)

        prefix_length = 1
        label_length = self.text_model.positional_embedding.shape[0] - self.context_tokens
        label_prefix = label_features[:, :prefix_length, :]
        label_suffix = label_features[:, prefix_length:label_length, :]

        context_features = self.learnable_prompts.expand(target.shape[0], -1, -1).to(device=target.device)
        if self.dynamic_context:
            context_order = torch.randperm(self.context_tokens)
            context_features = context_features[:, context_order, :]

        text_features = torch.concat([label_prefix, context_features, label_suffix], dim=1)

        text_features = text_features + self.text_model.positional_embedding.type(self.text_model.dtype)
        text_features = text_features.permute(1, 0, 2)  # NLD -> LND
        text_features = self.text_model.transformer(text_features)
        text_features = text_features.permute(1, 0, 2)  # LND -> NLD
        text_features = self.text_model.ln_final(text_features).type(self.text_model.dtype)

        return text_features[:, :self.output_tokens, :]

    def training_step(self, batch, batch_idx):
        """
        :param batch: tuple of (images, target)
        """
        images, target = batch
        img_logits, img_features = self.image_model.features_forward(images)

        text_features = self.encode_text_tokens(target)

        # Apply projection MLP (independently on each token embedding)
        projected_text_features = self.projection_mlp(text_features)

        loss, mm_part, ot_part = self.criterion(img_logits, target, img_features, projected_text_features)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mm_part', mm_part)
        self.log('train_ot_part', ot_part)
        self.log('mm_loss_temperature', self.criterion.mm_loss.temperature)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch
        img_logits, img_features = self.image_model.features_forward(images)
        text_features = self.encode_text_tokens(target)

        projected_text_features = self.projection_mlp(text_features)

        loss, mm_part, ot_part = self.criterion(img_logits, target, img_features, projected_text_features)
        acc1, acc5 = accuracy(img_logits, target, topk=(1, 5))
        self.log('val_loss', loss)
        self.log('val_mm_part', mm_part)
        self.log('val_ot_part', ot_part)
        self.log('val_acc1', acc1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc5', acc5)
        return {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}

    # todo: add test_step

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        # clamping of the temperature to log(-100) and log(100)
        self.criterion.mm_loss.temperature.data = torch.clamp(
            self.criterion.mm_loss.temperature.data, -4.6052, 4.6052
        )

    def configure_optimizers(self):
        # todo: make this a little nicer - don't really like how it looks now
        optimizer = torch.optim.SGD(
            list(self.image_model.parameters()) + list(self.projection_mlp.parameters()) + list(
                self.criterion.parameters()) + [self.learnable_prompts] if self.learnable_prompts.requires_grad else [],
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
