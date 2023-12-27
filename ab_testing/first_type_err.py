import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def estimate_first_type_error(df_pilot_group, df_control_group, metric_name, alpha=0.05, n_iter=10000, seed=None):
    """Оцениваем ошибку первого рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, считаем долю случаев с значимыми отличиями.

    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел.

    return - float, ошибка первого рода
    """
    if seed is not None:
        np.random.seed(seed)
    pilot = df_pilot_group[metric_name].values
    control = df_control_group[metric_name].values

    results = list()

    for i in range(n_iter):
        pilot_sample = np.random.choice(pilot, size=len(pilot))
        control_sample = np.random.choice(control, size=len(control))
        _, p_value = ttest_ind(pilot_sample, control_sample)
        results.append(p_value)

    p_values = np.asarray(results)

    first_type_errors = np.mean(p_values < 0.05)

    return first_type_errors