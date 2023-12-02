import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

learning_rate = 0.001
inner_dim = 256
training_epochs = 10
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

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
    total_correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions = (predicted == labels)

            correct_in_batch = correct_predictions.sum()

            total_correct_predictions += correct_in_batch

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100 * (total_correct_predictions / len(val_loader.dataset))


with open("output.txt", "w") as file:

    for epoch in range(training_epochs):
        model.train()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validate(model, val_loader, criterion)


    torch.save(model.state_dict(), 'model.pth')

    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    total_correct_predictions = 0
    labels_total = len(y_test)

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct_predictions = (predicted == labels)

        correct_in_batch = correct_predictions.sum()

        total_correct_predictions += correct_in_batch

    accuracy = 100 * total_correct_predictions / labels_total



chroma_row = [0.1, 1.65, 0.0177, 1.25, 0.00512, 1.03, 0.222, 0.0274, 2.31, 0.0441, 1.08, 0.0391]

model = NeuralNetwork()
model.load_state_dict(torch.load('model.pth'))
model.eval()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
scaled_input = scaler.transform([chroma_row])

input_tensor = torch.Tensor(scaled_input).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    print(predicted_class)
