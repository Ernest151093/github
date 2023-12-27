import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def estimate_second_type_error(df_pilot_group, df_control_group, metric_name, effects, alpha=0.05, n_iter=10000,
                               seed=None):
    """Оцениваем ошибки второго рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, добавляем эффект к пилотной группе,
    считаем долю случаев без значимых отличий.

    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    effects - List[float], список размеров эффектов ([1.03] - увеличение на 3%).
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел

    return - dict, {размер_эффекта: ошибка_второго_рода}
    """

    df_pilot = df_pilot_group[metric_name].values
    df_control = df_control_group[metric_name].values
    
    if seed is not None:
        np.random.seed(seed)

    estimate_second_type_error = dict()
    
    for effect in effects:
        p_values = list()
        for _ in range(n_iter):
            pilot_sample = np.random.choice(df_pilot, size=len(df_pilot))
            pilot_sample = pilot_sample * effect
            control_sample = np.random.choice(df_control, size=len(df_control))
            _, p_value = ttest_ind(pilot_sample, control_sample)
            
            p_values.append(p_value)

        second_type_error = np.mean(np.array(p_values) > alpha)
        estimate_second_type_error[effect] = second_type_error

    return estimate_second_type_error
