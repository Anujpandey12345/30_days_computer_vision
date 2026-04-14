import torch
import numpy as np
# # print(torch.__version__)

# # One D tensor
# x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
# print(x)


# #Two D tensor
# y  = torch.tensor([
#     [1, 2],
#     [3, 4],
# ])
# print(y)



# # Make tensor using numpy
# arr = np.array([1,2 ,3, 4, 5])
# n = torch.from_numpy(arr)
# print(n)







"""A small linear algebra project"""

# Step1 -- innput data x
x = torch.tensor([1., 2., 3., 4.])


#Step2 - y

y = torch.tensor([2., 4., 6., 8.])

#Step3 -- wrong guesses for w and b
w = torch.tensor([0.0])
b = torch.tensor([0.0])

# step 4 -- 
y_pred = w * x + b
loss = ((y_pred - y) ** 2).mean()

print("Predicted Value : ", y_pred)
print("Actuall Value : ", y)
print("Loss : ", loss)

