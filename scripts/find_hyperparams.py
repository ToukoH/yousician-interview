import optuna
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layer_size = trial.suggest_categorical('layer_size', [32, 64, 128, 256])
    
    layers = []
    input_size = 13
    for _ in range(n_layers):
        layers.append(nn.Linear(input_size, layer_size))
        layers.append(nn.ReLU())
        input_size = layer_size
    layers.append(nn.Linear(layer_size, 1))
    model = nn.Sequential(*layers)

    df = pd.read_csv("data/labeled_chord_data.csv")
    X = df.drop("chord_type", axis=1)
    y = df["chord_type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train.values))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(10):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

    model.eval()
    test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test.values))
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    correct = 0
    total = len(y_test)
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs.squeeze() > 0.5).long()
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print(f" Value: {trial.value}")
print(" Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
