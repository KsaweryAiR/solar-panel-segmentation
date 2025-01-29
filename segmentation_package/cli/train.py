import lightning.pytorch as pl
from segmentation_package.models.segmentation import LitModel
from segmentation_package.data_module.datamodulepanels import PanelsDataModule

if __name__ == "__main__":
    data_module = PanelsDataModule()
    data_module.setup()

    model = LitModel()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_BinaryF1Score',
        mode='max',
        verbose=True,
        filename='{epoch}-{val_loss:.2f}-{val_BinaryF1Score:.2f}'
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=[checkpoint_callback],
        max_epochs=100
    )

    trainer.fit(model, datamodule=data_module)
