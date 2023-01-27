'''Make a PyTorch Dataset for the Home Data Eval Subset.'''

import os
import pandas as pd # type: ignore

from torchvision.io import read_image # type: ignore
from torch.utils.data import Dataset # type: ignore

class HomeDataEvalSubset(Dataset):
    def __init__(self, root_dir, annotations, ontology):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass