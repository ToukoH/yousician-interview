import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

learning_rate = 0.0010663269850729943
inner_dim = 256
num_epochs = 20
target_class_dim = 24
input_dim = 13

data = "data/labeled_chord_data.csv"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_dim, inner_dim)
        self.l2 = nn.Linear(inner_dim, target_class_dim)
        self.l3 = nn.Linear(target_class_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x
    
df = pd.read_csv(data)
X = df.drop("chord_type", axis=1)
y = df["chord_type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=314)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train.values))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test.values))
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = NeuralNetwork().to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'model.pth')

model.load_state_dict(torch.load("model.pth"))
model.eval()

correct = 0
total = len(y_test)

for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs).squeeze()
    predicted = (outputs > 0.5).long()
    
    correct += (predicted == labels.long()).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.1f}%")
