import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy

from training_utils import accuracy


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
    def __init__(self, pretrained, arch, lr, momentum, weight_decay, num_classes):
        super().__init__()

        print(f"=> {'using pre-trained' if pretrained else 'creating'} model {arch}")
        self._model = models.__dict__[arch](pretrained=pretrained)

        assert 'resnet' in arch, 'Only resnet architectures are supported'
        # and this one will be for classification
        if arch == 'resnet18':
            self.num_channels = 512
        elif arch == 'resnet50':
            self.num_channels = 2048
        self.feature_dim = 49

        self._model_fc = nn.Linear(self.num_channels, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # idk why but these just give straight up wrong values
        # https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#multiclassaccuracy
        # self.metric1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        # self.metric5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.save_hyperparameters()

    def forward(self, x):
        return self._actual_fc(self._model(x))

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
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss = self.criterion(output, target)
        # acc1, acc5 = self.metric1(output, target), self.metric5(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.log('val_loss', loss)
        self.log('val_acc1', acc1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc5', acc5)
        return {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self._model.parameters(), self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [lr_scheduler]

    def get_feature_dim(self):
        return self.feature_dim
