import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression 
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from data_science.constants import StyleColor, TypeCols
from ..constants import Texts


def castTypeCols(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy(deep=True)

    cols_to_cast = TypeCols.CONTINUOS.value    
    for col_name in cols_to_cast:
        try:
            df[col_name] = pd.to_numeric(df[col_name], downcast="float")

        except:
            print(f'La variable {col_name} no puede convertirse a decimal.')
    
    cols_to_cast = TypeCols.DISCRETE.value    
    for col_name in cols_to_cast:
        try:
            df[col_name] = pd.to_numeric(df[col_name], downcast="integer")

        except:
            print(f'La variable {col_name} no puede convertirse a entero.')
    
    df[TypeCols.CATEGORICAL.value] = df[TypeCols.CATEGORICAL.value].astype("str")
        
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

def instanceSimplePlot(texts: Texts, fig_size: tuple = (8, 5)) -> tuple:
    fig, ax = plt.subplots(figsize=fig_size)

    fig.set_facecolor(color=StyleColor.FIG_FACE.value)
    ax.set_facecolor(color=StyleColor.AX_FACE.value)

    ax.set_title(texts.title, pad=15, color=StyleColor.TITL.value)

    ax.tick_params(axis='y', labelcolor=StyleColor.TICKS.value)
    ax.tick_params(axis='x', labelcolor=StyleColor.TICKS.value)
    
    ax.set_xlabel(texts.xlabel, color=StyleColor.LABELS.value)
    ax.set_ylabel(texts.xlabel, color=StyleColor.LABELS.value)

    ax.spines['bottom'].set_color(StyleColor.SPINES.value)
    ax.spines['top'].set_color(StyleColor.SPINES.value)
    ax.spines['left'].set_color(StyleColor.SPINES.value)
    ax.spines['right'].set_color(StyleColor.SPINES.value)

    fig.tight_layout()

    return fig, ax

def calculateVif(data:pd.DataFrame) -> pd.DataFrame:
    features = data.columns.to_list()
    
    mdl = LinearRegression()
    
    result = pd.DataFrame(index = ['VIF'], columns = features)
    result.fillna(0, inplace=True)
    
    for col in range(len(features)):
        x_features = features[:]
        y_feature = features[col]
        x_features.remove(y_feature)
        
        mdl.fit(data[x_features], data[y_feature])

        r_squared = mdl.score(data[x_features], data[y_feature])
        
        if abs(1 - r_squared) < 1e-10:
            result[y_feature] = float('inf')
            
        else:
            result[y_feature] = 1 / (1 - r_squared)
    
    return result

def calculateVariance(data: pd.DataFrame, threshold_value: float = 0.1) -> pd.DataFrame:
    features = data.columns.tolist()
    n_features = len(features)

    mdl = VarianceThreshold(threshold=threshold_value)
    X = mdl.fit_transform(data)

    selected_features = [features[i] for i, is_selected in enumerate(mdl.get_support()) if is_selected]
    n_selected_features = len(selected_features)

    df = pd.DataFrame(data=X, columns=selected_features)

    print(f'{n_features} Variables originales:')
    print(f'{n_selected_features} Variables tras la selección:')

    for drop_feature in selected_features: features.remove(drop_feature)
    print(f'Variables eliminadas: {features}')

    return df

def selectKbest(data: pd.DataFrame, target: pd.Series, k_value: int, type_classification: bool = False) -> pd.DataFrame:
    features = data.columns.tolist()
    n_features = len(features)

    if type_classification:
        mdl = SelectKBest(f_regression, k=k_value)

    else:
        mdl = SelectKBest(chi2, k=k_value)
    
    X = mdl.fit_transform(data, target)

    selected_features = [features[i] for i, is_selected in enumerate(mdl.get_support()) if is_selected]
    n_selected_features = len(selected_features)

    df = pd.DataFrame(data=X, columns=selected_features)

    print(f'{n_features} Variables originales:')
    print(f'{n_selected_features} Variables tras la selección:')

    for drop_feature in selected_features: features.remove(drop_feature)
    print(f'Variables eliminadas: {features}')

    return df

def stepWise(data: pd.DataFrame, target: pd.Series, n_variables: int = 0) -> pd.DataFrame:
    feature_order = []
    feature_error = []

    X_train, X_test, y_train, y_test = train_test_split(data.values, target.values, random_state=0)
    mdl = LinearRegression()

    for _ in range(X_train.shape[1]):
        col_try = [val for val in range(X_train.shape[1]) if val not in feature_order]
        iter_error = []
        
        for i_try in col_try:
            use_col = feature_order[:]
            use_col.append(i_try)
            
            use_train = X_train[:, use_col]
            use_test = X_test[:, use_col]
            
            mdl.fit(use_train, y_train)
            y_pred = mdl.predict(use_test)
            iter_error.append(mean_squared_error(y_test, y_pred))
            
        pos_best = np.argmin(iter_error)
        feature_order.append(col_try[pos_best])
        feature_error.append(iter_error[pos_best])

    if n_variables == 0: n_variables = len(data.columns)
    data = data.iloc[:, feature_order[:n_variables]]
        
    plt.plot(range(1, len(feature_error) + 1), feature_error)
    plt.xlabel('Características')
    plt.ylabel('Error (MSE)')

    return data

def lassoRegression(data: pd.DataFrame, target: pd.Series, alpha: float = 0.1, n_rounded: int = 3) -> pd.DataFrame:
    mdl = Lasso(alpha=alpha)
    mdl.fit(data, target)
    
    rounded_coefs = [round(coef, n_rounded) for coef in mdl.coef_]
    _ = [print(coe) for coe in rounded_coefs]

    selected_variables = [True if coef != 0.0 else False for coef in rounded_coefs]
    data = data.loc[:, selected_variables]

    return data
