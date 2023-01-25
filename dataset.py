'''Make a PyTorch Dataset for the Home Data Eval Subset.'''

import os
import pandas as pd

from torchvision.io import read_image
from torch.utils.data import Dataset

class HomeDataEvalSubset(Dataset):
    def __init__(self, root_dir, annotations, ontology):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass