import matplotlib.pyplot as plt

def plot_structures_and_field(data, i, j, name, xname, yname):
    plt.figure(figsize=(10, 5))
    plt.imshow(data[i,j,:,:], cmap='Blues', aspect='auto') # j es campo real o campo imaginario
    plt.colorbar(label='Intensidad')
    plt.title(name)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(f'{name}.png')
    plt.show()
