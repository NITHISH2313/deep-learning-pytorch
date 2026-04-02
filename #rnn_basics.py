#rnn_basics

import torch
import torch.nn as nn
import numpy as np

# ==== data =====
# predict next number in sequence
# input : [1,2,3,4,5] - output : 6

def create_sequences(data , seq_length):
    x , y = [] , []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return torch.tensor(x , dtype = torch.float32),\
           torch.tensor(y , dtype = torch.float32)

#generate sequence data

data = list(range(1,51))
data_normalized = [x/50.0 for x in data]
seq_length = 5

x , y =create_sequences(data_normalized, seq_length)
x = x.unsqueeze(-1) # add feature dimension
y = y.unsqueeze(-1)

print(f" x shape :{x.shape}")
print(f" y shape :{y.shape}")
print(f"sample : {x[0].squeeze()} - {y[0].item():.3f}")

#====BUILD RNN =====

class sequencernn(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size =1,hidden_size = 64, num_layers = 2, batch_first = True)
        self.fc = nn.Linear(64 , 1)

    def forward(self , x):
        out ,_ =self.rnn(x)
        out = self.fc(out[:, -1 ,:])
        return out
model = sequencernn()
print(f"\n RNN architecture:")
print(model)

#==== TRAIN =====
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters() , lr=0.01)

print("\n training rnn")
for epoch in range(500):
    prediction = model(x)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)% 20 == 0:
        print(f"epoch{epoch+1}/100 - loss :{loss.item():.4f}")

#==== predict=======
model.eval()
with torch.no_grad():
    test_seq = torch.tensor([[46/50], [47/50], [48/50], [49/50], [50/50]],dtype = torch.float32).unsqueeze(0)
    pred = model(test_seq)
    predicted_value = pred.item()*50
    print(f"\n Input sequence : 46, 47 ,48 ,49, 50")
    print(f"prediction next:{predicted_value:.1f}")
    print(f"actual answer : 51")