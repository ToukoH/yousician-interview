import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

data = "data/labeled_chord_data.csv"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(13, 64)
        self.l2 = nn.Linear(64, 24)
        self.l3 = nn.Linear(24, 1)
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

test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test.values))
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = NeuralNetwork()
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
