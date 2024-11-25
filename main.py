import matplotlib.pyplot as plt

from utils.data_load import data_import
from utils.plots import plot_structures_and_field


if __name__ == "__main__":
    print('Iniciando')
    path_data_test = r"C:\Folder_Personal\Física\Trabajo de grado\Data_WaveYNet\test_ds.npz"
    path_data_train = r"C:\Folder_Personal\Física\Trabajo de grado\Data_WaveYNet\train_ds.npz"
    train_structures, train_Hy_fields, train_dielectric_permittivities, test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities = data_import(path_data_train, path_data_test)

    print(train_structures.shape)
    print(train_Hy_fields.shape)

    plot_structures_and_field(train_structures, 100, 0, 'Estructuras', 'Tamaño horizontal', 'Tamaño vertical')

    print('Finalizado')