import torch
from torchsummary import summary
from utils.transformer_preentrened import ModifiedViT
from utils.plots import plot_structures_and_field
from utils.data_load import data_import
import time
from generate_samples import load_model
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance

def to_numpy(tensor):
    """ Convierte tensores de PyTorch a numpy si es necesario. """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def mean_squared_error(real, generated):
    """ Calcula el Error Cuadrático Medio (MSE) entre los datos reales y generados. """
    real, generated = to_numpy(real), to_numpy(generated)
    return np.mean((real - generated) ** 2)

def mean_absolute_error(real, generated):
    """ Calcula el Error Absoluto Medio (MAE) entre los datos reales y generados. """
    real, generated = to_numpy(real), to_numpy(generated)
    return np.mean(np.abs(real - generated))

def pearson_correlation(real, generated):
    """ Calcula el coeficiente de correlación de Pearson para cada muestra y promedia el resultado. """
    real, generated = to_numpy(real), to_numpy(generated)
    real = real.reshape(real.shape[0], -1)  # Aplanar cada muestra
    generated = generated.reshape(generated.shape[0], -1)
    correlations = [pearsonr(real[i], generated[i])[0] for i in range(real.shape[0])]
    return np.mean(correlations)

def ssim_score(real, generated):
    """ Calcula el Índice de Similaridad Estructural (SSIM) y promedia sobre todas las muestras. """
    real, generated = to_numpy(real), to_numpy(generated)
    ssim_values = []
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):  # 2 canales
            ssim_values.append(ssim(real[i, j], generated[i, j], data_range=real.max() - real.min()))
    return np.mean(ssim_values)

def wasserstein_distance_metric(real, generated):
    """ Calcula la Distancia de Wasserstein (EMD) para cada muestra y promedia el resultado. """
    real, generated = to_numpy(real), to_numpy(generated)
    real = real.reshape(real.shape[0], -1)  # Aplanar cada muestra
    generated = generated.reshape(generated.shape[0], -1)
    distances = [wasserstein_distance(real[i], generated[i]) for i in range(real.shape[0])]
    return np.mean(distances)

# Ejemplo de uso
if __name__ == "__main__":
    path_data_test = r"C:\Folder_Personal\Física\Trabajo de grado\Data_WaveYNet\test_ds.npz"
    path_data_train = r"C:\Folder_Personal\Física\Trabajo de grado\Data_WaveYNet\train_ds.npz"
    start_time = time.time()
    train_structures, train_Hy_fields, train_dielectric_permittivities, test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities = data_import(path_data_train, path_data_test)


    checkpoint_path = r'checkpoint/model_epoch_50_to_120.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, optimizer, epoch, loss = load_model(checkpoint_path, device=device)

    N = 3000
    real_data = torch.tensor(test_Hy_fields[:N,:,:,:])
    generated_data = model.predict(torch.tensor(test_structures[:N,:,:,:]))

    print("MSE:", mean_squared_error(real_data, generated_data))
    print("MAE:", mean_absolute_error(real_data, generated_data))
    print("Pearson Correlation:", pearson_correlation(real_data, generated_data))
    print("SSIM:", ssim_score(real_data, generated_data))
    print("Wasserstein Distance:", wasserstein_distance_metric(real_data, generated_data))


