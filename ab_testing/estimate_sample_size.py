import numpy as np
import pandas as pd
from scipy import stats


def get_sample_size(mu, std, effect, alpha, beta):
    epsilon = np.abs(1 - effect) * mu

    t_alpha = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)
    t_beta = stats.norm.ppf(1 - beta, loc=0, scale=1)

    square_stats = (t_alpha + t_beta) ** 2

    sample_size = ((square_stats * (2 * std ** 2)) // (epsilon ** 2)) + 1

    return sample_size


def estimate_sample_size(df, metric_name, effects, alpha=0.05, beta=0.2):
    """Оцениваем sample size для списка эффектов.

    df - pd.DataFrame, датафрейм с данными
    metric_name - str, название столбца с целевой метрикой
    effects - List[float], список ожидаемых эффектов. Например, [1.03] - увеличение на 3%
    alpha - float, ошибка первого рода
    beta - float, ошибка второго рода

    return - pd.DataFrame со столбцами ['effect', 'sample_size']
    """

    metric = df[metric_name].values

    mu = np.mean(metric)
    std = np.std(metric)

    sample_size = list()

    for effect in effects:
        sample_size.append(get_sample_size(mu, std, effect, alpha, beta))

    df_res = pd.DataFrame({'effect': effects, 'sample_size': sample_size})

    df_res['sample_size'] = df_res['sample_size'].astype('int')

    return df_res
