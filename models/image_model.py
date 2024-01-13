import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.classification import MulticlassAccuracy

from training_utils import accuracy


class ImageClassificationModel(pl.LightningModule):
    def __init__(self, pretrained, arch, logger, lr, momentum, weight_decay, num_classes):
        super().__init__()
        if pretrained:
            # print("=> using pre-trained model '{}'".format(args.arch))
            logger.info("=> using pre-trained model '{}'".format(arch))
            self.model = models.__dict__[arch](pretrained=True)
        else:
            # print("=> creating model '{}'".format(args.arch))
            logger.info("=> creating model '{}'".format(arch))
            self.model = models.__dict__[arch]()

        # Modify the final layer based on the architecture
        if arch == 'resnet18':
            self.model.fc = nn.Linear(512, num_classes)
        elif arch == 'resnet50':
            self.model.fc = nn.Linear(2048, num_classes)

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
        return self.model(x)

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
        optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [lr_scheduler]
        # {"scheduler": scheduler, "interval": "epoch"}
