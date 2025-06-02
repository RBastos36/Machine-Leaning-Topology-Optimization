# Author: Ricardo A. O. Bastos
# Created: June 2025


import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import spsolve


class PoissonDataset(Dataset):
    """
    Dataset for synthetic 2D Poisson equation: ∇²u(x, y) = f(x, y)
    Each sample has:
      - input: [2, H, W] → [domain_mask, source_term f(x,y)]
      - target: [1, H, W] → solution u(x,y)
    """

    def __init__(self, num_samples=1000, grid_size=64, stats=None):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.stats = stats or {"mean": 0.0, "std": 1.0}
        self.samples = [self.generate_sample() for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        domain_mask, f, u = self.samples[idx]
        input_tensor = torch.tensor(np.stack([domain_mask, f]), dtype=torch.float32)  # [2, H, W]
        output_tensor = torch.tensor(u[None, :, :], dtype=torch.float32)              # [1, H, W]
        return input_tensor, output_tensor

    def generate_sample(self):
        N = self.grid_size
        h = 1.0 / (N + 1)

        # Random source term f(x, y)
        f = np.random.randn(N, N)

        # Solve the Poisson equation
        A = self._laplace_matrix(N) / h**2
        u = spsolve(A, f.flatten()).reshape(N, N)

        # Normalize u and f
        u = (u - self.stats.get("mean", 0.0)) / self.stats.get("std", 1.0)
        f = (f - self.stats.get("mean", 0.0)) / self.stats.get("std", 1.0)

        # Domain mask is all ones
        domain_mask = np.ones((N, N), dtype=np.float32)

        return domain_mask, f.astype(np.float32), u.astype(np.float32)

    def _laplace_matrix(self, N):
        e = np.ones(N)
        T = diags([e, -4 * e, e], [-1, 0, 1], shape=(N, N))
        I = identity(N)
        A = kron(I, T) + kron(diags([e], [-1], shape=(N, N)), I) + kron(diags([e], [1], shape=(N, N)), I)
        return A.tocsr()
