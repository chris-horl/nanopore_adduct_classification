import torch
from torch import nn
import math
from data.data import NanoporeDataModule

class InputEmbedding(nn.Module):
    """InputEmbedding based on The Annotated Transformer v2022
       https://nlp.seas.harvard.edu/annotated-transformer/#embeddings-and-softmax

    Attributes:
        d_model: model dimension of transformer
        vocab_size: length of one-hot vector encoding nucleobases
    """

    def __init__(self, d_model=68, vocab_size=4):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size,
                                            embedding_dim=self.d_model)

    def forward(self, x):
        """
        Input: one-hot vector of length 4 representing a nucleobase
        """
        x = torch.argmax(x, dim=-1) # Convert one-hot to indices / nucleobase
        return self.embedding_layer(x) * math.sqrt(self.d_model)

#<<<<< <<<<< <<<<< <<<<< <<<<< <<<<< Testing >>>>> >>>>> >>>>> >>>>> >>>>> >>>>>

def main():
    dm = NanoporeDataModule()
    dm.setup("fit")
    X_sig, X_seq, y, idx = next(iter(dm.test_dataloader()))
    ie = InputEmbedding()
    example_embedding = ie(X_seq.type(torch.long))

if __name__ == "__main__":
    main()