from enum import Enum

class Path(Enum):
    PATH = 'data_science//datasets//'
    TEST = 'data_science//datasets//test.csv'
    TRAIN = 'data_science//datasets//train.csv'

class Color(Enum):
    DARK = '#161b22'
    WHITE ='#D1D4C9'
    BLUE ='#2f81f7'

class StyleColor(Enum):
    FIG_FACE = Color.DARK.value
    AX_FACE = Color.DARK.value
    TITL = Color.WHITE.value
    LABELS = Color.WHITE.value
    TICKS = Color.WHITE.value
    SPINES = Color.WHITE.value

# Dicciconario para poder seleccionar los datos en función del tipo de característica
class TypeCols(Enum):
    DISCRETE = [
        'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
        'BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
        'MoSold', 'YrSold'
    ]
    CONTINUOS = [
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
    ]
    CATEGORICAL = [
        'MSSubClass', 'MSZoning', 'Street','Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual',
        'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual',
        'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual','Functional', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
        'MiscFeature', 'SaleType', 'SaleCondition'
    ]

class Texts():
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
