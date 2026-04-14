import torch

# w = torch.tensor(3.0, requires_grad=True)
# x = torch.tensor(4.0)
# y_pred = x * w
# y_true = torch.tensor(10.0)
# loss = (y_pred - y_true) ** 2
# print(loss)
# loss.backward()
# print(w.grad)

# Input Fixed
x = torch.tensor(2.0)
# learnable number
w = torch.tensor(3.0, requires_grad=True)
#Currect answer
y_true = torch.tensor(10.0)
# Learning rate
lr = 0.1
y_pred = w * x
loss = (y_pred - y_true) ** 2
loss.backward()

print("Before Update")
print("W: ", w.item())
print("loss: ", loss.item())
print("gradiant: ", w.grad.item())

with torch.no_grad():
    w -= lr * w.grad
w.grad.zero_()
print("After Update")
print("W: ", w.item())
print("loss: ", loss.item())
print("gradiant: ", w.grad.item())
