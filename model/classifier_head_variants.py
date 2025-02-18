from model.classifier_head import AdductClassificationModel
from data.data import NanoporeDataModule
import lightning as L
import numpy as np
import time

class HookExtractionModel(AdductClassificationModel):
    """ Child of AdductClassificationModel including extraction of activations.

    Extraction of activations is only performed during testing phase using data
    not included in training and validation.
    
    Attributes:
        reset_activation_history: Bool flag triggering initialization of data 
            structures to save activations
        activations: Dict saving activations. Keys are layer aliases.
        labels: List for saving labels
        data_indices: List saving data indices
        hook_handle_first_linear: handle for hook function attached to first
            linear layer of the classifier head.
    """
    def __init__(self):
        super().__init__()
        self.reset_activation_history = True

    def __save_activations(self, layer_alias: str):
        """Hook function to save activations and corresponding labels"""
        def hook_fn(model, input, output):
            # module: nn.Module, input: Tensor, output: Tensor
            self.activations[layer_alias].append(output.detach().numpy(force=True))
        return hook_fn
    
    def remove_hooks(self):
        self.hook_handle_first_linear.remove()

    def test_step(self, batch, batch_idx):
        X_sig, X_seq, y, idx = batch

        # Initialize dict and register hook only during first test step
        if self.reset_activation_history == True:
            self.labels = []
            self.data_indices = []
            self.activations = {"first_linear": []}
            self.hook_handle_first_linear = self.model.classifier[0].register_forward_hook(self.__save_activations("first_linear"))
            self.reset_activation_history = False

        # Save labels and data indices
        self.labels.append(y.numpy(force=True))
        self.data_indices.append(idx.numpy(force=True))
        # Perform test step logic
        pred = self.model(X_sig.unsqueeze(1), X_seq)
        loss = self.loss_fn(pred, y.unsqueeze(-1))
        acc = self.acc(pred, y.unsqueeze(-1))
        self.log("test_acc", acc)
        self.log("test_loss", loss)

def retrieve_activations(checkpoint_path: str="foo"):
    """Function to extract activations from the first linear layer of classifier.
    
    Uses HookExtractionModel. Saves activations, labels and data indices as npy.

    Args:
        checkpoint_path: .ckpt file generated by Lightning to load trained model
    """
    timestr = time.strftime("%Y%m%d_%H%M%S")
    L.seed_everything(44)
    checkpoint_path = "/checkpoints/my_checkpoint.ckpt"
    # Load best model from ckpt and test to retrieve activations
    model = HookExtractionModel.load_from_checkpoint(checkpoint_path)
    datamodule = NanoporeDataModule(sample_size=20_000)
    trainer = L.Trainer()
    trainer.test(model=model, datamodule=datamodule, verbose=True)
    model.remove_hooks()
    activations_ls_1st_lin = model.activations["first_linear"]
    np.save(f"./results/{timestr}_activations_1st_lin.npy", np.concatenate(activations_ls_1st_lin), allow_pickle=True)
    labels = model.labels
    np.save(f"./results/{timestr}_labels.npy", np.concatenate(labels), allow_pickle=True)
    indices = model.data_indices
    np.save(f"./results/{timestr}_indices.npy", np.concatenate(indices), allow_pickle=True)


if __name__ == "__main__":
    retrieve_activations()
