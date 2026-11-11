import torch

from dual_prior_rul import DualPriorReconstructor


def main():
    torch.manual_seed(1)
    torch.set_num_threads(2)
    signal = torch.rand(1, 30, 14)
    mask = torch.rand_like(signal) > 0.4
    model = DualPriorReconstructor(curvature_variance=torch.full((14,), 0.02)).eval()
    with torch.no_grad():
        restored = model(signal, mask)
    print(f"Reconstructed window shape: {tuple(restored.shape)}")
    print("Synthetic input and untrained denoiser; demonstration only.")


if __name__ == "__main__":
    main()
