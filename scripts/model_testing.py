import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

inner_dim = 256
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
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, target_class_dim),
            nn.ReLU(),
            nn.Linear(target_class_dim, 1),
        )

    def forward(self, x):
        return self.stack(self.flatten(x))
"""

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_dim, inner_dim)
        self.l2 = nn.Linear(inner_dim, target_class_dim)
        self.l3 = nn.Linear(target_class_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Tanh()

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

test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test.values))
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

correct = 0
total = len(y_test)

for inputs, labels in test_loader:
    outputs = model(inputs)
    predicted = (outputs.squeeze() > 0.5).long()

    correct_predictions = (predicted == labels)

    batch_correct = correct_predictions.sum()
    correct_predictions = batch_correct.item()

    correct += correct_predictions
    
print(f"Accuracy: {100 * correct / total}%")
