from enum import Enum

class Path(Enum):
    PATH = '../datasets/'
    TEST = '../datasets/test.csv'
    TRAIN = '../datasets/train.csv'

class Color(Enum):
    DARK = '#161b22'
    WHITE ='#D1D4C9'
    BLUE ='#2f81f7'

STYLE_COLOR = {
    'fig_face_color': Color.DARK.value,
    'ax_face_color': Color.DARK.value,
    'title_color': Color.WHITE.value,
    'labels_color': Color.WHITE.value,
    'ticks_color': Color.WHITE.value,
    'spines_color': Color.WHITE.value
}

# Dicciconario para poder seleccionar los datos en función del tipo de característica
TYPE_COLS = {
    'discrete': [
        'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
        'BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
        'MoSold', 'YrSold'
    ],
    'continuos':[
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
    ],
    'categorical': [
        'MSSubClass', 'MSZoning', 'Street','Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual',
        'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual',
        'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual','Functional', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
        'MiscFeature', 'SaleType', 'SaleCondition'
    ]
}
