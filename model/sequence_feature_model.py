import lightning as L
import torch
from torch import nn
from torchmetrics.classification import BinaryAccuracy
from model.input_embedding import InputEmbedding
from model.positional_encoding import PositionalEncoding

# For testing purposes
from data.sequence_data import SequenceDataModule

class SequenceFeatureModel(nn.Module):
    """Submodel for sequence features extraction based on base-called sequence.

    Builds on Transformer architecture using Transformer Encoder only.

    Parameters:
        See PositionalEncoding module and PyTorch documentation
        for nn.TransformerEncoderLayer
    """
    def __init__(self,
                # <<<TransformerEncoderLayer>>>
                d_model=68,
                nhead=1, # original paper: 8
                dropout=0.1,
                activation="relu",
                layer_norm_eps=1e-5,
                batch_first=True, # default: False
                norm_first=False,
                bias=True,
                # <<<TransformerEncoder>>>
                num_layers=6, # from original paper
                enable_nested_tensor=False,
                # <<<PositionalEncoding>>>
                max_len=17
                ):
        super().__init__()

        self.input_embedding = InputEmbedding(d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        self.encoder_layer = nn.TransformerEncoderLayer(
                                              d_model=d_model,
                                              nhead=nhead,
                                              dim_feedforward=d_model * 4,
                                              dropout=dropout,
                                              activation=activation,
                                              layer_norm_eps=layer_norm_eps,
                                              batch_first=batch_first, 
                                              norm_first=norm_first,
                                              bias=bias
                                              ) 
        
        self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                             num_layers=num_layers, 
                                             enable_nested_tensor=enable_nested_tensor
                                             )

        self.head = nn.Sequential(
            nn.Flatten()
        )

        self.net = nn.Sequential(
                                 self.input_embedding,
                                 self.positional_encoding,
                                 self.encoder,
                                 self.head
                                 )

    def forward(self, x):
        """
        Args:
            x: source tensor of size (N, S, E) for batch_first=True
                N := batch size
                S := source sequence length
                E := feature number

        Returns:
            tensor of size (N, T, E)
            T := target sequence length
        """
        return self.net(x.type(torch.long))

class LitSequenceFeatureModel(L.LightningModule):
    """Lightning wrapper for SequenceFeatureModel.
    
        Used for testing purposes.
    """

    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = SequenceFeatureModel()
        self.loss_fn = nn.BCELoss()
        self.acc = BinaryAccuracy(threshold=0.5)
        self.lr = lr

    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        X, y, _ = batch
        pred = self.model(X)
        loss = self.loss_fn(pred, y.unsqueeze(-1))
        acc = self.acc(pred, y.unsqueeze(-1))
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, _ = batch
        pred = self.model(X)
        loss = self.loss_fn(pred, y.unsqueeze(-1))
        acc = self.acc(pred, y.unsqueeze(-1))
        self.log("val_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx): 
        X, y, idx = batch

        # Perform test step logic
        pred = self.model(X)
        # <<<Debugging>>>
        # print("\n pred: {}".format(torch.max(pred)))
        # print("\n y: {}".format(torch.max(y.unsqueeze(-1))))
        loss = self.loss_fn(pred, y.unsqueeze(-1))
        acc = self.acc(pred, y.unsqueeze(-1))
        self.log("test_acc", acc)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


#<<<<< <<<<< <<<<< <<<<< <<<<< <<<<< Testing >>>>> >>>>> >>>>> >>>>> >>>>> >>>>>

def test_forward_pass():
    """Technical test of the sequence feature module"""

    L.seed_everything(44)
    model = LitSequenceFeatureModel()
    datamodule = SequenceDataModule()
    trainer = L.Trainer()
    trainer.test(model=model, datamodule=datamodule, verbose=True)

def main():
    test_forward_pass()

if __name__ == "__main__":
    main()