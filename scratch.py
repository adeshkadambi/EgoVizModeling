"""This is a scratch file for testing out code snippets."""

import torch  # type: ignore
import numpy as np  # type: ignore

print(torch.__version__)
print(np.__version__)


x = np.nan

print(x)

if np.nan_to_num(x) == 0:
    print("yes")


print(x + x)