import torch
import os
import argparse
import matplotlib.pyplot as plt
import time
from utils.data_load import data_import
from utils.plots import plot_structures_and_field
from utils.train import train_model
from utils.transformer_preentrened import ModifiedViT
from generate_samples import load_model

def main(args):
    print('Iniciando')
    path_data_test = r"C:\Folder_Personal\Física\Trabajo de grado\Data_WaveYNet\train_ds.npz"
    path_data_train = r"C:\Folder_Personal\Física\Trabajo de grado\Data_WaveYNet\train_ds.npz"
    start_time = time.time()
    train_structures, train_Hy_fields, train_dielectric_permittivities, test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities = data_import(path_data_train, path_data_test)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Import data time: {execution_time:.2f} seconds")

    print(train_structures.shape)
    print(train_Hy_fields.shape)

    num_sample = args.num_sample # 27000 maximo
    epochs = args.epochs
    batch_size_2 = args.batch_size

    seq_len = train_structures.shape[2] #64
    num_features = train_structures.shape[3] #256
    output_channels = train_Hy_fields.shape[1]
    input_dim = num_features
    output_dim = output_channels # 2

    # Parámetros de entrenamiento changes

    lr = args.lr
    # train 27000 samples
    # test 3000 samples
    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, "checkpoint")

#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    train_data = torch.tensor(train_structures[:num_sample, :, :, :], dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_Hy_fields[:num_sample, :, :, :], dtype=torch.float32).to(device)
    # test_data = torch.tensor(train_structures[:batch_size, :, :, :], dtype=torch.float32)
    # test_labels = torch.tensor(train_Hy_fields[:batch_size, :, :, :], dtype=torch.float32)

    test_data = torch.tensor(test_structures[:, :, :, :], dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_Hy_fields[:, :, :, :], dtype=torch.float32).to(device)

    print('Implementacion de ModifiedViT')

    checkpoint_path = r'checkpoint/model_epoch_50.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, optimizer, epoch, loss = load_model(checkpoint_path, device=device)

    start_time = time.time()
    train_model(model, train_data, train_labels, test_data, test_labels, epochs, batch_size_2, lr, device, save_path, optimizer)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training time: {execution_time:.2f} seconds")

    plot_structures_and_field(model.predict(train_data[:1, :, :, :]), 0, 0, 'Campo Generado', 'Tamaño horizontal', 'Tamaño vertical')
    plot_structures_and_field(train_labels[:1, :, :, :], 0, 0, 'Campo Real', 'Tamaño horizontal', 'Tamaño vertical') # plot magnetic field

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de entrenamiento para inference de magnetic field")

    parser.add_argument("--num_sample", type=int, required=True, help="Tamaño de la muestra de train")
    parser.add_argument("--epochs", type=int, required=True, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, required=True, help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Tasa de aprendizaje")
    parser.add_argument("--dropout_rate", type=float, default=0.4, help="Tasa de dropout")

    args = parser.parse_args()

    main(args)
