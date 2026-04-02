#lstm_basics

import torch
import torch.nn as nn


# ==== data =====


def create_sequences(data , seq_length):
    x , y = [] , []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return torch.tensor(x , dtype = torch.float32),\
           torch.tensor(y , dtype = torch.float32)

#generate sequence data

data = list(range(1,101))
data_normalized = [x/100.0 for x in data]
seq_length = 10

x , y =create_sequences(data_normalized, seq_length)
x = x.unsqueeze(-1) # add feature dimension
y = y.unsqueeze(-1)

print(f" x shape :{x.shape}")
print(f" y shape :{y.shape}")
print(f"sample : {x[0].squeeze()[:5]} - {y[0].item():.3f}")

#====BUILD LSTM =====

class sequenceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size =1,hidden_size = 64, num_layers = 2, batch_first = True,dropout = 0.2)
        self.fc = nn.Linear(64 , 1)

    def forward(self , x):
        out ,(hidden , cell) =self.lstm(x)
        out = self.fc(out[:, -1 ,:])
        return out
model = sequenceLSTM()
print(f"\n LSTM architecture:")
print(model)

#==== TRAIN =====
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters() , lr=0.01)

print("\n training LSTM")
for epoch in range(500):
    prediction = model(x)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)% 100 == 0:
        print(f"epoch{epoch+1}/500 - loss :{loss.item():.6f}")

#==== predict=======
model.eval()
with torch.no_grad():
    test_seq = torch.tensor([[91/100],[92/100],[93/100],[94/100],[95/100],
         [96/100],[97/100],[98/100],[99/100],[100/100]],dtype = torch.float32).unsqueeze(0)
    pred = model(test_seq)
    predicted_value = pred.item()*100
    print(f"\n Input sequence : 91,92,93,94,95,96,97,98,99,100")
    print(f"prediction next:{predicted_value:.1f}")
    print(f"actual answer : 101")