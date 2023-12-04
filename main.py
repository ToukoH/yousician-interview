##################################
# Yousician Interview Assignment #
#        Touko Haapanen          #
#           2.12.2023            #
##################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# HYPERPARAMETERS
learning_rate = 0.001
inner_dim = 256
training_epochs = 10
target_class_dim = 24
input_dim = 12
batch_size = 32

# NAMING THE DEVICE TO RUN EFFICIENT INFERENCE ON CUDA OR MPS CAPABLE HARDWARE
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# LOADING AND PREPARING THE DATA
data = "data/combined_label_chord_data.csv"
df = pd.read_csv(data)
X = df.drop(["combined_label"], axis=1).values
y = df["combined_label"].values

# DATASET SIZES ARE:
# 70% FOR TRAINING
# 15% FOR VALIDATION
# 15% FOR TESTING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

# FITTING THE SCALER AND STANDARDIZING THE DATA
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# CONVERTING DATA INTO TESORS
X_train, y_train = torch.Tensor(X_train), torch.LongTensor(y_train)
X_val, y_val = torch.Tensor(X_val), torch.LongTensor(y_val)
X_test, y_test = torch.Tensor(X_test), torch.LongTensor(y_test)

# CREATING SEPARATE TORCH DATASETS
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# WRAPPING AN ITERABLE AROUND THE DATASETS
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# DEFINING OUR NEURAL NETWORK
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

# INITIALIZING MODEL, OPTIMIZER AND LOSS FUNCTION (WE USE CRITERION HERE FOR CLARITY)
model = NeuralNetwork().to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# VALIDATION FUNCTION, USED IN TRAINING LOOP
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

    file.write(f"Validation: Average loss: {val_loss:.4f}, Accuracy: {val_accuracy:.1f}%\n")
    print(f"Validation: Average loss: {val_loss:.4f}, Accuracy: {val_accuracy:.1f}%")

# TRAINING METRICS IS LOGGED INTO A SEPARATE FILE
with open("output.txt", "w") as file:
    file.write("Executing training\n")
    file.write("_________________________________________\n")
    
    # TRAINING LOOP
    for epoch in range(training_epochs):
        print(f"Starting epoch {epoch + 1}")
        file.write((f"Starting epoch {epoch + 1}\n"))
        model.train()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                file.write(f"Training Epoch {epoch + 1}: [{100 * batch_idx / len(train_loader):.0f}%   Loss: {loss.item():.6f}]\n")
                print(f"Training Epoch {epoch + 1}: [{100 * batch_idx / len(train_loader):.0f}%   Loss: {loss.item():.6f}]")

        print(f"Training epoch {epoch + 1} finished")
        validate(model, val_loader, criterion)

        file.write(f"Training epoch {epoch + 1} finished!\n")
        file.write("_________________________________________\n")
        file.write("")

    torch.save(model.state_dict(), 'model.pth')

    # TESTING PHASE
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

    print(f"Testing accuracy: {accuracy:.1f}%")
    file.write("\n")
    file.write(f"Testing accuracy: {accuracy:.1f}%\n")
