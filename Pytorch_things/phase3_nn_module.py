import torch
import torch.nn as nn
import torch.optim as optim


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = nn.Linear(1,1)
        
    def forward(self, x):
        return self.Linear(x)
    
# Data
x = torch.tensor([[1.], [2.], [3.], [4.]])
y = torch.tensor([[2.], [4.], [6.], [8.]])
# Model
model = MyModel()
# Loss & optimizer
loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(10):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    print(f"EPOCH {epoch + 1}, Loss : {loss.item():.4f}")




#  Save the Pytorch model 

torch.save(model.state_dict(), "Linear_model.pth")