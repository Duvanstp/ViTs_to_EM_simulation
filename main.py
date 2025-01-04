import torch
import matplotlib.pyplot as plt

from utils.data_load import data_import
from utils.plots import plot_structures_and_field
from utils.transformers import BasicTransformer
from utils.train import train_model


if __name__ == "__main__":
    print('Iniciando')
    path_data_test = r"C:\Folder_Personal\Física\Trabajo de grado\Data_WaveYNet\test_ds.npz"
    path_data_train = r"C:\Folder_Personal\Física\Trabajo de grado\Data_WaveYNet\train_ds.npz"
    train_structures, train_Hy_fields, train_dielectric_permittivities, test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities = data_import(path_data_train, path_data_test)

    print(train_structures.shape)
    print(train_Hy_fields.shape)

    #plot_structures_and_field(train_structures, 100, 0, 'Estructuras', 'Tamaño horizontal', 'Tamaño vertical')

    #print(train_structures.shape[1])
    #print(train_structures.shape[2])
    #print(train_structures.shape[3])

    seq_len = train_structures.shape[2] #64
    num_features = train_structures.shape[3] #256
    output_channels = train_Hy_fields.shape[1]
    input_dim = num_features
    output_dim = output_channels # 2

    # Parámetros de entrenamiento changes
    batch_size = 300 # 27000 maximo
    num_heads = 8 # num par [2^n]
    num_layers = 2
    epochs = 3
    lr = 0.001
    # train 27000 samples
    # test 3000 samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicTransformer(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len, num_heads=num_heads, num_layers=num_layers)


    train_data = torch.tensor(train_structures[:batch_size, :, :, :], dtype=torch.float32)
    train_labels = torch.tensor(train_Hy_fields[:batch_size, :, :, :], dtype=torch.float32)
    # test_data = torch.tensor(train_structures[:batch_size, :, :, :], dtype=torch.float32)
    # test_labels = torch.tensor(train_Hy_fields[:batch_size, :, :, :], dtype=torch.float32)

    test_data = torch.tensor(test_structures[:batch_size, :, :, :], dtype=torch.float32)
    test_labels = torch.tensor(test_Hy_fields[:batch_size, :, :, :], dtype=torch.float32)


    # print(model(train_data).shape)

    train_model(model, train_data, train_labels, test_data, test_labels, epochs, batch_size, lr, device)

    plot_structures_and_field(model.predict(train_data[:1, :, :, :]), 0, 0, 'Campo Generado', 'Tamaño horizontal', 'Tamaño vertical')
    plot_structures_and_field(train_labels[:1, :, :, :], 0, 0, 'Campo Real', 'Tamaño horizontal', 'Tamaño vertical') # plot magnetic field

    print('Finalizado')