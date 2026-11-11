"""Run dual-prior reconstruction and RUL prediction on synthetic input."""

import torch

from dual_prior_rul import DualPriorRUL, RUL_SCALE


def main():
    torch.manual_seed(1)
    torch.set_num_threads(2)
    # Two overlapping 30-cycle windows from one synthetic 33-cycle trajectory.
    trajectory = torch.rand(1, 33, 14)
    observed = torch.rand_like(trajectory) > 0.4
    model = DualPriorRUL(curvature_variance=torch.full((14,), 0.02)).eval()

    with torch.no_grad():
        restored = model.reconstruct(
            trajectory[:, 3:], observed[:, 3:],
            history=trajectory[:, :-3],
            history_mask=observed[:, :-3],
            guidance_strength=1.0,
        )
        prediction = RUL_SCALE * model.predictor(restored).flatten()

    print(f"Reconstructed window shape: {tuple(restored.shape)}")
    print(f"Example RUL output (cycles): {prediction.item():.3f}")
    print("Synthetic input and untrained networks; this is a demo, not an experiment result.")


if __name__ == "__main__":
    main()
