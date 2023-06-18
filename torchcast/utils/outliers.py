import torch


def mahalanobis_dist(diff: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    cholesky = torch.linalg.cholesky(covariance)
    y = torch.cholesky_solve(diff.unsqueeze(-1), cholesky).squeeze(-1)
    return torch.sqrt(torch.sum(diff * y, -1))
