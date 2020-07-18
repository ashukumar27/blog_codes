from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso

from regression_model.config import config
import regression_model.processing.preprocessors as pp

price_pipe = Pipeline(
    [
        ('Numerical Imputer',pp.NumericalImputer(variables = config.NUMERICAL_FEATURES)),
        ('Categorical Imputer', pp.CategoricalImputer(variables = config.CATEGORICAL_FEATURES)),
        ('Temporal Features', pp.TemporalVariableEstimator(variables = config.TEMPORAL_FEATURES, 
        reference_variable=config.TEMPORAL_COMPARISON)),
        ('Rare Label Encoder', pp.RareLabelCategoricalImputer(variables = config.FEATURES_TO_ENCODE)),
        ('Categorical Encoder', pp.CategoricalEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('Log Transform', pp.LogTransformation(variables = config.LOG_FEATURES)),
        ('Drop Features', pp.DropFeatures(variables_to_drop=config.DROP_FEATURES)),
        ('Scaler Transform', MinMaxScaler()),
        ('Linear Model', Lasso(alpha=0.005,random_state=42))
      ]
)

