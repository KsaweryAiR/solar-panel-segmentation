import torch
import torchmetrics
import lightning.pytorch as pl
import segmentation_models_pytorch as smp

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
        self.loss_function = smp.losses.DiceLoss(mode='binary')

        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task='binary'),
            torchmetrics.Precision(task='binary'),
            torchmetrics.Recall(task='binary'),
            torchmetrics.F1Score(task='binary'),
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        self.train_metrics.update(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log_dict(self.train_metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        self.val_metrics.update(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.val_metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        self.test_metrics.update(outputs, labels)
        self.log('test_loss', loss)
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
