import torch
import os
import torch.nn as nn
import torch.optim as optim
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, train_data, train_labels, test_data, test_labels, epochs, batch_size, lr, device, save_path, optimizer = None):
    """
    Entrena el modelo en los datos dados y evalúa el desempeño en el conjunto de prueba, mostrando progreso con TQDM.

    :param model: El modelo a entrenar
    :param train_data: Tensores de estructuras de entrenamiento
    :param train_labels: Tensores de etiquetas (Hy fields) de entrenamiento
    :param test_data: Tensores de estructuras de prueba
    :param test_labels: Tensores de etiquetas (Hy fields) de prueba
    :param epochs: Número de épocas de entrenamiento
    :param batch_size: Tamaño del lote
    :param lr: Tasa de aprendizaje
    :param device: Dispositivo ("cuda" o "cpu")
    """
    # Mover el modelo al dispositivo
    model = model.to(device)

    if optimizer is None:
    # Definir el optimizador y la función de pérdida
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()

    # Crear DataLoaders para el entrenamiento y prueba
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    epoch_losses = []

    # Ciclo de entrenamiento con barra de progreso
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Configurar TQDM para el ciclo de entrenamiento
        with tqdm(total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch") as pbar:
            for inputs, labels in train_loader:
                # Mover los datos al dispositivo
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward y optimización
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({"Loss": f"{train_loss / (pbar.n + 1):.4f}"})
                pbar.update(1)

        # Promedio de pérdida por época
        train_loss /= len(train_loader)
        epoch_losses.append(train_loss)
        print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {train_loss:.4f}")

        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(save_path, f"model_epoch_50_to_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")

       # Evaluación en datos de prueba
        evaluate_model(model, test_loader, criterion, device)
        csv_path = os.path.join(save_path, "train_losses.csv")
        with open(csv_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Epoch", "Train Loss"])
            for epoch, loss in enumerate(epoch_losses, 1):
                csv_writer.writerow([epoch, loss])

        print(f"Training losses saved to {csv_path}")

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
