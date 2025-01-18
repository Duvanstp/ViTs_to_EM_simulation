import torch
import torch.optim as optim


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
    checkpoint_path = 'checkpoints/model_epoch_200.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, optimizer, epoch, loss = load_model(checkpoint_path, device=device)