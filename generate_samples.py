import torch
import torch.optim as optim
import os
import time
from utils.data_load import data_import
from utils.plots import plot_structures_and_field
from utils.train import train_model
from utils.transformer_preentrened import ModifiedViT

from utils.transformer_preentrened import ModifiedViT


def load_model(checkpoint_path, input_size=(1, 1, 64, 256), device='cpu'):
    model = ModifiedViT(input_size=input_size, patch_size=(16, 16), num_output_channels=2, smoothing_kernel_size=3, dropout_rate=0.4).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Modelo cargado desde {checkpoint_path}")
    print(f"Época: {epoch}, Pérdida: {loss}")

    return model, optimizer, epoch, loss


if __name__ == '__main__':
    checkpoint_path = r'checkpoint/model_epoch_50.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, optimizer, epoch, loss = load_model(checkpoint_path, device=device)

    path_data_test = r"C:\Folder_Personal\Física\Trabajo de grado\Data_WaveYNet\test_ds.npz"
    path_data_train = r"C:\Folder_Personal\Física\Trabajo de grado\Data_WaveYNet\train_ds.npz"
    start_time = time.time()
    train_structures, train_Hy_fields, train_dielectric_permittivities, test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities = data_import(path_data_train, path_data_test)

    device = torch.device("cpu")
    train_data = torch.tensor(train_structures[:1, :, :, :], dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_Hy_fields[:1, :, :, :], dtype=torch.float32).to(device)

    test_data = torch.tensor(test_structures[:1, :, :, :], dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_Hy_fields[:1, :, :, :], dtype=torch.float32).to(device)

    plot_structures_and_field(model.predict(test_data[:1, :, :, :]), 0, 1, 'Campo Generado muestra test 1', 'Tamaño horizontal', 'Tamaño vertical')
    plot_structures_and_field(test_labels[:1, :, :, :], 0, 1, 'Campo Imaginario muestra test 1', 'Tamaño horizontal', 'Tamaño vertical')