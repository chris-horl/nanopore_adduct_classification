from lightning.pytorch.cli import LightningCLI
from model.classifier_head import AdductClassificationModel
from data.data import NanoporeDataModule

def main():
    cli = LightningCLI(
        model_class = AdductClassificationModel,
        datamodule_class = NanoporeDataModule,
    )

if __name__ == "__main__":
    main()