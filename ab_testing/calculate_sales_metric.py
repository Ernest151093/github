import numpy as np
import pandas as pd

def calculate_sales_metrics(df, cost_name, date_name, sale_id_name, period, filters=None):
    """Вычисляет метрики по продажам.

    df - pd.DataFrame, датафрейм с данными. Пример
        pd.DataFrame(
            [[820, '2021-04-03', 1, 213]],
            columns=['cost', 'date', 'sale_id', 'shop_id']
        )
    cost_name - str, название столбца с стоимостью товара
    date_name - str, название столбца с датой покупки
    sale_id_name - str, название столбца с идентификатором покупки (в одной покупке может быть несколько товаров)
    period - dict, словарь с датами начала и конца периода пилота.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}.
        Дата начала периода входит в полуинтервал, а дата окончания нет,
        то есть '2020-01-01' <= date < '2020-01-08'.
    filters - dict, словарь с фильтрами. Ключ - название поля, по которому фильтруем, значение - список значений,
        которые нужно оставить. Например, {'user_id': [111, 123, 943]}.
        Если None, то фильтровать не нужно.

    return - pd.DataFrame, в индексах все даты из указанного периода отсортированные по возрастанию, 
        столбцы - метрики ['revenue', 'number_purchases', 'average_check', 'average_number_items'].
        Формат данных столбцов - float, формат данных индекса - datetime64[ns].
    """
    dict_cols = {cost_name: {'sum': 'revenue',
                             'mean': 'average_check'
                             },
                 sale_id_name: {'count': 'number_purchases',
                                'mean': 'average_number_items'
                                }
                 }

    df = df[(df[date_name] < period['end']) & (df[date_name] >= period['begin'])]
    if filters:
        for col, values in filters.items():
            df = df[df[col].isin(values)]

    i = 0
    for col_name, dict_ in dict_cols.items():
        for agg_func, new_col_name in dict_.items():
            if col_name == sale_id_name and agg_func == 'mean':
                df0 = (df
                       .groupby(by=[date_name, sale_id_name])[cost_name]
                       .count()
                       .reset_index()
                       .groupby(by=[date_name])[cost_name]
                       .agg(agg_func)
                       .reset_index()
                       )
            else:
                df0 = df.groupby(by=[date_name])[col_name].agg(agg_func).reset_index()
                df0 = df0.rename(columns={col_name: new_col_name})

            if i == 0:
                df_tot = df0
            else:
                df_tot = df_tot.merge(df0, on=date_name, how='outer')

            i += 1

    df_tot = df_tot.set_index('date', drop=True)
    df_tot = df_tot.astype('float')

    return df_tot
