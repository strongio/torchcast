from scipy import stats


def conf2bounds(mean, std, conf) -> tuple:
    assert conf >= .50
    multi = -stats.norm.ppf((1 - conf) / 2)
    lower = mean - multi * std
    upper = mean + multi * std
    return lower, upper
