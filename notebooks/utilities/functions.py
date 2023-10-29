import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def castTypeCols(data: pd.DataFrame, dtype: dict) -> pd.DataFrame:
    df = data.copy(deep=True)

    TYPES = {
        'categorical': 'str',
        'continuos': 'float',
        'discrete': 'float',
    }
    
    for type in list(dtype.keys()):
        for col in dtype[type]:
            try:
                if type != 'categorical':
                    df.replace('nan', np.NaN, inplace=True)
                    
                df.loc[:, col] = df.loc[:, col].astype(TYPES[type])

            except:
                print(f'La variable {col} no puede convertirse a {TYPES[type]}')
                df.loc[:, col] = df.loc[:, col].astype('str')
    
    return df


def statisticsNulls(df: pd.DataFrame) -> None:
    n_records = df.shape[0] * df.shape[1]
    n_nulls = df.isnull().sum().sum()
    n_rows = df.shape[0]
    n_rows_nulls = df.isnull().any(axis=1).sum()
    p_records_nulls = "{:.2f}".format(n_nulls / n_records * 100)
    p_rows_nulls = "{:.2f}".format(n_rows_nulls / n_rows * 100)

    print(f'""" Estadísticas de Nulos: """')
    print(f'Hay {n_nulls} / {n_records} de registros nulos, un {p_records_nulls}% sobre el total.')
    print(f'Hay {n_rows_nulls} / {n_rows} filas con al menos un nulo, un {p_rows_nulls}% sobre el total.')

    if n_nulls > 0:
        print(f'\n""" Top 10 variables con nulos """')
        print(df.isnull().sum().sort_values(ascending=False)[:10])

def instanceSimplePlot(texts: dict, colors: dict, fig_size: tuple = (8, 5)) -> tuple:
    fig, ax = plt.subplots(figsize=fig_size)

    fig.set_facecolor(color=colors.get('fig_face_color'))
    ax.set_facecolor(color=colors.get('ax_face_color'))
    ax.set_title(texts.get('title'), pad=15, color=colors.get('title_color'))
    ax.tick_params(axis='y', labelcolor=colors.get('ticks_color'))
    ax.tick_params(axis='x', labelcolor=colors.get('ticks_color'))
    ax.set_xlabel(texts.get('xlabel'), color=colors.get('labels_color'))
    ax.set_ylabel(texts.get('ylabel'), color=colors.get('labels_color'))
    ax.spines['bottom'].set_color(colors.get('spines_color'))
    ax.spines['top'].set_color(colors.get('spines_color'))
    ax.spines['left'].set_color(colors.get('spines_color'))
    ax.spines['right'].set_color(colors.get('spines_color'))

    fig.tight_layout()

    return fig, ax