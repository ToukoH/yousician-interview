import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

learning_rate = 0.001
inner_dim = 256
training_epochs = 5
target_class_dim = 24
input_dim = 12
batch_size = 32

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

data = "data/combined_label_chord_data.csv"

data = pd.read_csv(data)
X = data.drop(["combined_label"], axis=1).values
y = data["combined_label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=314)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=314)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train, y_train = torch.Tensor(X_train), torch.LongTensor(y_train)
X_val, y_val = torch.Tensor(X_val), torch.LongTensor(y_val)
X_test, y_test = torch.Tensor(X_test), torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_dim, inner_dim)
        self.l2 = nn.Linear(inner_dim, inner_dim)
        self.l3 = nn.Linear(inner_dim, target_class_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x

model = NeuralNetwork().to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in val_loader:
            output = model(X)
            val_loss += criterion(output, y).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.0f}%)')

print("Executing training")
print("_________________________________________")
for epoch in range(training_epochs):
    model.train()
    for batch_idx, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch + 1} [{batch_idx * len(X)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    validate(model, val_loader, criterion)

torch.save(model.state_dict(), 'model.pth')

model.load_state_dict(torch.load("model.pth"))
model.eval()

correct = 0
total = len(y_test)

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    correct_predictions = (predicted == labels)

    batch_correct = correct_predictions.sum()
    correct_predictions = batch_correct.item()

    correct += correct_predictions

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.1f}%")
