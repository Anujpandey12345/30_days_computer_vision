# 




"""Polling"""

import torch
import torch.nn as nn
 

image = torch.rand(1, 1, 28, 28)


pool = nn.MaxPool2d(
    kernel_size=2,
)


output = pool(image)

print("Before Pooling: ", image.shape)
print("After Pooling: ", output.shape)