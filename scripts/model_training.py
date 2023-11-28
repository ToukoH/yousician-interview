import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# FROM OPTUNA
#########################################
learning_rate = 0.0010663269850729943   #
layer_size = 256                        #
#########################################

num_epochs = 20
target_class_size = 24
input_size = 13

data = "data/labeled_chord_data.csv"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def init(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.ff_stack = nn.Sequential(
          nn.Linear(input_size, layer_size),
          nn.ReLU(),
          nn.Linear(layer_size, target_class_size),
          nn.ReLU(),
          nn.Linear(target_class_size, 1),
          nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.ff_stack(self.flatten(x))
        return x

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.l1 = nn.Linear(input_size, layer_size)
#         self.l2 = nn.Linear(layer_size, target_class_size)
#         self.l3 = nn.Linear(target_class_size, 1)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.l1(x))
#         x = self.relu(self.l2(x))
#         x = self.l3(x)
#         return x

df = pd.read_csv(data)

X = df.drop("chord_type", axis=1)
y = df["chord_type"]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=314)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train.values))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

model = NeuralNetwork().to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'model.pth')
