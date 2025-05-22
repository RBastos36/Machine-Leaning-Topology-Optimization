import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import PoissonDataset
from model_Unet_node_level import TopologyOptimizationCNN  # Adjust if your model class has a different name
import os


def compute_laplacian(u):
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = kernel.to(u.device)
    return F.conv2d(u, kernel, padding=1)


def compute_residual(u_pred, f_input):
    laplace_u = compute_laplacian(u_pred)
    return laplace_u - f_input


def visualize_sample(input_tensor, u_true, u_pred, residual):
    domain_mask = input_tensor[0].cpu().numpy()
    f = input_tensor[1].cpu().numpy()
    u_true = u_true.squeeze().cpu().numpy()
    u_pred = u_pred.squeeze().detach().cpu().numpy()
    residual = residual.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(f, cmap='coolwarm');
    axs[0].set_title("f(x, y)")
    axs[1].imshow(u_true, cmap='viridis');
    axs[1].set_title("u true")
    axs[2].imshow(u_pred, cmap='viridis');
    axs[2].set_title("u predicted")
    axs[3].imshow(residual, cmap='inferno');
    axs[3].set_title("Residual (∇²û - f)")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.show()


def test_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load model
    model = TopologyOptimizationCNN()  # Adjust if yours is different
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load test data
    test_dataset = PoissonDataset(num_samples=10, grid_size=64)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Inference
    for i, (input_tensor, u_true) in enumerate(test_loader):
        input_tensor = input_tensor.to(device)
        u_true = u_true.to(device)
        f_input = input_tensor[:, 1:2, :, :]  # Extract f(x, y)

        with torch.no_grad():
            u_pred = model(input_tensor)
            residual = compute_residual(u_pred, f_input)

        print(f"Sample {i + 1}")
        visualize_sample(input_tensor[0], u_true[0], u_pred[0], residual[0])

        if i >= 4:  # limit to 5 samples
            break


if __name__ == '__main__':
    # Path to your pre-trained model
    model_file = 'checkpoints/best_model.pth'  # Change if needed

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at {model_file}")

    test_model(model_file)
