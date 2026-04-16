import torch

x = torch.tensor([1., 2., 3., 4.])
y_true = torch.tensor([2., 4., 6., 8.]) #Correct answer
w = torch.tensor(0.0, requires_grad=True)
lr = 0.1
epochs = 10

for epoch in range(epochs):
    #Step1 prediction
    y_pred = w * x

    #Step2 - loss
    loss = ((y_pred - y_true) ** 2).mean()
    loss.backward()


    with torch.no_grad():
        w -= lr*w.grad
    
    print(f"Epoch {epoch+1}: w={w.item():.4f}, loss={loss.item():.4f}")
