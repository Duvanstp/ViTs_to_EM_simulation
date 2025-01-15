import torch
import time
import matplotlib.pyplot as plt

from utils.data_load import data_import
from utils.plots import plot_structures_and_field
from utils.basic_transformer import BasicTransformer
from utils.train import train_model
from utils.transformer_preentrened import ModifiedViT
from transformers import ViTFeatureExtractor


if __name__ == "__main__":
    print('Iniciando')
    path_data_test = r"./data/test_ds.npz"
    path_data_train = r"./data/train_ds.npz"
    start_time = time.time()
    train_structures, train_Hy_fields, train_dielectric_permittivities, test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities = data_import(path_data_train, path_data_test)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Import data time: {execution_time:.2f} seconds")

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
    num_sample = 10000
    #int(input('Tamaño de la muestra de train: ')) # 27000 maximo
    epochs = 3
    #int(input('numero de epochs: '))
    lr = 0.001
    # train 27000 samples
    # test 3000 samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = torch.tensor(train_structures[:num_sample, :, :, :], dtype=torch.float32)
    train_labels = torch.tensor(train_Hy_fields[:num_sample, :, :, :], dtype=torch.float32)
    # test_data = torch.tensor(train_structures[:batch_size, :, :, :], dtype=torch.float32)
    # test_labels = torch.tensor(train_Hy_fields[:batch_size, :, :, :], dtype=torch.float32)

    test_data = torch.tensor(test_structures[:num_sample, :, :, :], dtype=torch.float32)
    test_labels = torch.tensor(test_Hy_fields[:num_sample, :, :, :], dtype=torch.float32)

    model_select = 1
#int(input('Que modelo desea usar, seleccione el numero: \n 1. basic_trasformer\n 2. ModifiedViT\n'))

    if model_select == 1:
        num_heads = 8 # num par [2^n]
        num_layers = 2
        model = BasicTransformer(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len, num_heads=num_heads, num_layers=num_layers)

        batch_size_2 = 1000
#int(input(f'Tamaño del batch, no puede ser mayor a: {num_sample}: '))

        train_model(model, train_data, train_labels, test_data, test_labels, epochs, batch_size_2 , lr, device)

        plot_structures_and_field(model.predict(train_data[:1, :, :, :]), 0, 0, 'Campo Generado', 'Tamaño horizontal', 'Tamaño vertical')
        plot_structures_and_field(train_labels[:1, :, :, :], 0, 0, 'Campo Real', 'Tamaño horizontal', 'Tamaño vertical') # plot magnetic field

        print('Finalizado')
    elif model_select == 2:
        print('Implementacion de ModifiedViT')
        from PIL import Image
        import requests

        feature_extractor = ViTFeatureExtractor(size=(64, 256), do_resize=False)

        model = ModifiedViT(pretrained_model_name="google/vit-base-patch16-224", input_size=(1, 1,64, 256), patch_size=(16, 16), num_output_channels=2, smoothing_kernel_size=3, dropout_rate=0.2)

        inputs = torch.tensor(train_structures[:10, :, :, :], dtype=torch.float32)
        batch_size_2 = 1000
#int(input(f'Tamaño del batch, no puede ser mayor a: {num_sample}: '))
        train_model(model, train_data, train_labels, test_data, test_labels, epochs, batch_size_2, lr, device)

        plot_structures_and_field(model.predict(train_data[:1, :, :, :]), 0, 0, 'Campo Generado', 'Tamaño horizontal', 'Tamaño vertical')
        plot_structures_and_field(train_labels[:1, :, :, :], 0, 0, 'Campo Real', 'Tamaño horizontal', 'Tamaño vertical') # plot magnetic field
    else:
        print('No ha seleccionado un modelo valido')
