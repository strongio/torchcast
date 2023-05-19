import torch


def get_inlier_mask(resid: torch.Tensor,
                    system_covariance: torch.Tensor,
                    outlier_threshold: torch.Tensor) -> torch.Tensor:
    if outlier_threshold > 0:
        mdist = mahalanobis_dist(resid, system_covariance)
        return mdist <= outlier_threshold
    else:
        return torch.ones(len(resid), dtype=torch.bool, device=resid.device)


def mahalanobis_dist(diff: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    cholesky = torch.linalg.cholesky(covariance)
    y = torch.cholesky_solve(diff.unsqueeze(-1), cholesky).squeeze(-1)
    mahalanobis_dist = torch.sqrt(torch.sum(diff * y, -1))
    return mahalanobis_dist
