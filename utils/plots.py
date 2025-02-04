import torch
import matplotlib.pyplot as plt

def plot_structures_and_field(data, i, j, name, xname, yname, label='Intensidad'):
# Ensure the data is on the CPU and converted to NumPy
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.imshow(data[i,j,:,:], cmap='Blues', aspect='auto') # j es campo real o campo imaginario
    plt.colorbar(label=label)
    plt.title(name)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(f'{name}.pdf')
    plt.show()
