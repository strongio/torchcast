import torch
from torch.linalg import LinAlgError


def get_outlier_multi(resid: torch.Tensor,
                      cov: torch.Tensor,
                      outlier_threshold: torch.Tensor) -> torch.Tensor:
    if len(outlier_threshold) == 2:
        if resid.shape[-1] != 1:
            raise NotImplementedError
        neg_mask = (resid < 0).squeeze(-1)
        mdist_neg = mahalanobis_dist(resid[neg_mask], cov[neg_mask])
        mdist_pos = mahalanobis_dist(resid[~neg_mask], cov[~neg_mask])
        multi = torch.ones_like(resid).squeeze(-1)
        neg_thresh, pos_thresh = outlier_threshold.abs()
        multi[neg_mask] = (mdist_neg - neg_thresh).clamp(min=0) + 1
        multi[~neg_mask] = (mdist_pos - pos_thresh).clamp(min=0) + 1
    else:
        assert outlier_threshold.numel()
        mdist = mahalanobis_dist(resid, cov)
        multi = (mdist - outlier_threshold).clamp(min=0) + 1
    return multi


def mahalanobis_dist(diff: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    try:
        cholesky = torch.linalg.cholesky(covariance)
    except LinAlgError as e:
        cholesky = None
        for i in [-8, -7, -6, -5, -4, -3, -2]:
            try:
                cholesky = torch.linalg.cholesky(covariance + torch.eye(covariance.shape[-1]) * 10 ** -i)
                break
            except LinAlgError:
                continue
        if cholesky is None:
            raise e

    y = torch.cholesky_solve(diff.unsqueeze(-1), cholesky).squeeze(-1)
    return torch.sqrt(torch.sum(diff * y, -1))
