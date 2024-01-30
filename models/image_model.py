import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

from training_utils import accuracy
from models.cosine_lr_scheduler import CosineLRScheduler


def features_forward(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    features = model.layer4(x)

    x = model.avgpool(features)
    x = torch.flatten(x, 1)
    x = model.fc(x)

    return x, features


class ImageClassificationModel(pl.LightningModule):
    def __init__(self, pretrained, arch, lr, momentum, weight_decay, num_classes, total_steps):
        super().__init__()

        print(f"=> {'using pre-trained' if pretrained else 'creating'} model {arch}")
        self._model = models.__dict__[arch](pretrained=pretrained)

        assert 'resnet' in arch, 'Only resnet architectures are supported'
        if arch == 'resnet18':
            self.num_channels = 512
        elif arch == 'resnet50':
            self.num_channels = 2048
        self.feature_dim = 49

        if not pretrained or num_classes != 1000:
            self._model.fc = nn.Linear(self.num_channels, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.total_steps = total_steps

        self.save_hyperparameters()

    def forward(self, x):
        return self._model(x)

    def features_forward(self, x) -> (torch.Tensor, torch.Tensor):
        """
        Returns the features of the penultimate layer and the logits
        """
        logits, features = features_forward(self._model, x)
        return logits, features.view(logits.shape[0], self.num_channels, self.feature_dim)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss = self.criterion(output, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, logging_prefix='val'):
        images, target = batch
        output = self(images)
        loss = self.criterion(output, target)
        # acc1, acc5 = self.metric1(output, target), self.metric5(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.log(f'{logging_prefix}_loss', loss)
        self.log(f'{logging_prefix}_acc1', acc1, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{logging_prefix}_acc5', acc5)
        return {f'{logging_prefix}_loss': loss, f'{logging_prefix}_acc1': acc1, f'{logging_prefix}_acc5': acc5}
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, logging_prefix='test')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self._model.parameters(), self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        lr_scheduler = CosineLRScheduler(optimizer, T_max=self.total_steps)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def get_feature_dim(self):
        return self.feature_dim