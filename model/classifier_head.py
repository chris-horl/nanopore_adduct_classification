import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
import lightning as L
from model.signal_feature_model import SignalFeatureModel
from model.sequence_feature_model import SequenceFeatureModel
from data.data import NanoporeDataModule

class ClassifierHead(nn.Module):
    """Combined signal and sequence features for classification

    Calls SignalFeatureModel and SequenceFeatureModel and uses the concatenated
    output for classification in fully connected layers. Outputs are raw logits
    as required by torch.nn.CrossEntropyLoss.
    
    Attributes:
        num_classes: number of adduct classes to be classified
        dropout_classifier: dropout rate of layer between the final two fully
            connected layers
        linear_dimensions: number of neurons in the two fully connected layers

        See SignalFeatureModel and SequenceFeatureModel for ulterior attributes
    """
    def __init__(self,
                num_classes=6,
                dropout_classifier=0.5,
                linear_dimensions=(4756, 4756),
                # <<<SignalFeatureModel>>>
                signal_output_channels=64,
                bias_signal=False,
                # <<<TransformerEncoderLayer>>>
                d_model=68,
                nhead=1, # original paper: 8
                dropout_transformer=0.1,
                activation="relu",
                layer_norm_eps=1e-5,
                batch_first=True, # default: False
                norm_first=False,
                bias_transformer=True,
                # <<<TransformerEncoder>>>
                num_layer_transformer=6, # from original paper
                enable_nested_tensor=False,
                # <<<PositionalEncoding>>>
                max_len=17
                 ):      
        super().__init__()

        # Hyperparameters of classifier
        self.num_classes = num_classes
        self.dropout_classifier = dropout_classifier
        self.linear_dimensions = linear_dimensions

        # Hyperparameters of SignalFeatureModel
        self.c_out = signal_output_channels
        self.bias_signal = bias_signal

        # Hyperparameters of SequenceFeatureModel
        # <<<TransformerEncoderLayer>>>
        self.d_model=d_model
        self.nhead=nhead # original paper: 8
        self.dropout_transformer=dropout_transformer
        self.activation=activation
        self.layer_norm_eps=layer_norm_eps
        self.batch_first=batch_first # default: False
        self.norm_first=norm_first
        self.bias_transformer=bias_transformer
        # <<<TransformerEncoder>>>
        self.num_layers_transformer=num_layer_transformer # from original paper
        self.enable_nested_tensor=enable_nested_tensor
        # <<<PositionalEncoding>>>
        self.max_len=max_len
        

        # output of signal_model: vector of len 3600 for c_out=64
        self.signal_model = SignalFeatureModel(c_out=self.c_out, bias_mode=self.bias_signal)

        # output: vector of len 1156 for 68-embedding, 68 for 4-embedding
        self.sequence_model = SequenceFeatureModel(d_model=self.d_model,
                                                    nhead=self.nhead,
                                                    dropout=self.dropout_transformer,
                                                    activation=self.activation,
                                                    layer_norm_eps=self.layer_norm_eps,
                                                    batch_first=self.batch_first,
                                                    norm_first=self.norm_first,
                                                    bias=self.bias_transformer,
                                                    num_layers=self.num_layers_transformer,
                                                    enable_nested_tensor=self.enable_nested_tensor,
                                                    max_len=self.max_len) 

        self.classifier = nn.Sequential(
            nn.Linear(self.linear_dimensions[0], self.linear_dimensions[1], bias=False),
            nn.Dropout(p=self.dropout_classifier),
            nn.Linear(self.linear_dimensions[1], self.num_classes, bias=False),
        )

    def forward(self, x_sig, x_seq):
        signal_features_flat = self.signal_model(x_sig)
        sequence_features_flat = self.sequence_model(x_seq)
        x = torch.cat((signal_features_flat, sequence_features_flat), dim=1)
        return self.classifier(x)

class AdductClassificationModel(L.LightningModule):
    """LightningModule wrapper for ClassifierHead.
     
    Exposing parameters of the model to the client. All parameters are modifiable
    using yaml configuration files that can be used with the Lightning CLI.

    Args:
        Refer to base classes of the model's part:
            ClassifierHead
            SignalFeatureModel
            SequenceFeatureModel
    """
    def __init__(self,
                lr=1e-3,
                num_classes=6,
                dropout_classifier=0.5,
                linear_dimensions=(4756, 4756),
                # <<<SignalFeatureModel>>>
                signal_output_channels=64,
                bias_signal=False,
                # <<<TransformerEncoderLayer>>>
                d_model=68,
                nhead=1, # original paper: 8
                dropout_transformer=0.1,
                activation="relu",
                layer_norm_eps=1e-5,
                batch_first=True, # default: False
                norm_first=False,
                bias_transformer=True,
                # <<<TransformerEncoder>>>
                num_layer_transformer=6, # from original paper
                enable_nested_tensor=False,
                # <<<PositionalEncoding>>>
                max_len=17):
        super().__init__()
        self.model = ClassifierHead(
                num_classes=num_classes,
                dropout_classifier=dropout_classifier,
                linear_dimensions=linear_dimensions,
                # <<<SignalFeatureModel>>>
                signal_output_channels=signal_output_channels,
                bias_signal=bias_signal,
                # <<<TransformerEncoderLayer>>>
                d_model=d_model,
                nhead=nhead, # original paper: 8
                dropout_transformer=dropout_transformer,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first, # default: False
                norm_first=norm_first,
                bias_transformer=bias_transformer,
                # <<<TransformerEncoder>>>
                num_layer_transformer=num_layer_transformer, # from original paper
                enable_nested_tensor=enable_nested_tensor,
                # <<<PositionalEncoding>>>
                max_len=max_len
                )
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = MulticlassAccuracy(num_classes = 6)
        self.lr = lr

    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        X_sig, X_seq, y, _ = batch

        pred = self.model(X_sig.unsqueeze(1), X_seq)
        loss = self.loss_fn(pred, y)
        acc = self.acc(pred, y)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_sig, X_seq, y, _ = batch

        pred = self.model(X_sig.unsqueeze(1), X_seq)
        loss = self.loss_fn(pred, y)
        acc = self.acc(pred, y)
        self.log("val_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        X_sig, X_seq, y, idx = batch

        # Perform test step logic
        pred = self.model(X_sig.unsqueeze(1), X_seq)
        loss = self.loss_fn(pred, y)
        acc = self.acc(pred, y)
        self.log("test_acc", acc)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

def test_forward_pass():
    """Technical test of the model"""
    L.seed_everything(44)
    print("Init. model...")
    model = AdductClassificationModel()
    print("Init. datamodule...")
    datamodule = NanoporeDataModule(sample_size=1000, batch_size=512)
    print("Init. trainer...")
    trainer = L.Trainer()
    print("Starting forward pass...")
    trainer.validate(model=model, datamodule=datamodule, verbose=True)

def main():
    test_forward_pass()

if __name__ == "__main__":
    main()