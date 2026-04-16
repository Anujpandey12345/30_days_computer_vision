import torch
import torch.nn as nn


# class MyModule(nn.module):
#     def __init__():
#         super().__init__()


    

layer = nn.Linear(3, 2)
x = torch.tensor([1.0, 2.0, 3.0])
y = layer(x)
print(y)