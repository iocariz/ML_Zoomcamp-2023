
# Machine learning functions from scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.base import BaseEstimator, TransformerMixin  # Base classes for custom transformers
from sklearn.compose import ColumnTransformer, make_column_selector  # For transforming columns of dataframes
from sklearn.impute import SimpleImputer  # For handling missing data
from sklearn.pipeline import make_pipeline, Pipeline  # For creating a pipeline of transformations
from sklearn.preprocessing import StandardScaler, FunctionTransformer  # For scaling and custom transformations
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder  # For encoding categorical variables
from sklearn.model_selection import GridSearchCV,cross_val_score  # For cross-validation and hyperparameter tuning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # For calculating mean squared error of predictions
import xgboost as xgb  # Gradient boosting framework

from scipy.stats import norm
from sklearn.metrics import mean_squared_error

def monkey_patch_get_signature_names_out():
    """
    Monkey patch some classes which did not handle get_feature_names_out()
    correctly in Scikit-Learn 1.0.*.

    This function monkey patches the SimpleImputer and FunctionTransformer classes
    in Scikit-Learn to handle get_feature_names_out() correctly. It does this by
    setting the get_feature_names_out() method of SimpleImputer to the default
    get_feature_names_out() method of StandardScaler, and by monkey patching the
    FunctionTransformer class to add a feature_names_out parameter to its __init__()
    method and a get_feature_names_out() method that handles the feature_names_out
    parameter correctly.
    """
    from inspect import Signature, signature, Parameter
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    default_get_feature_names_out = StandardScaler.get_feature_names_out

    if not hasattr(SimpleImputer, "get_feature_names_out"):
      print("Monkey-patching SimpleImputer.get_feature_names_out()")
      SimpleImputer.get_feature_names_out = default_get_feature_names_out

    if not hasattr(FunctionTransformer, "get_feature_names_out"):
        print("Monkey-patching FunctionTransformer.get_feature_names_out()")
        orig_init = FunctionTransformer.__init__
        orig_sig = signature(orig_init)

        def __init__(*args, feature_names_out=None, **kwargs):
            orig_sig.bind(*args, **kwargs)
            orig_init(*args, **kwargs)
            args[0].feature_names_out = feature_names_out

        __init__.__signature__ = Signature(
            list(signature(orig_init).parameters.values()) + [
                Parameter("feature_names_out", Parameter.KEYWORD_ONLY)])

        def get_feature_names_out(self, names=None):
            if callable(self.feature_names_out):
                return self.feature_names_out(self, names)
            assert self.feature_names_out == "one-to-one"
            return default_get_feature_names_out(self, names)

        FunctionTransformer.__init__ = __init__
        FunctionTransformer.get_feature_names_out = get_feature_names_out

monkey_patch_get_signature_names_out()

# Custom transformer for interaction terms
class InteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, interaction_pairs):
        self.interaction_pairs = interaction_pairs
    
    def fit(self, X, y=None):
        # The fit method doesn't need to do anything in this case
        return self
    
    def transform(self, X):
        X = X.copy()
        for pair in self.interaction_pairs:
            feature_name = f"{pair[0]}_x_{pair[1]}"
            X[feature_name] = X[pair[0]] * X[pair[1]]
        return X

    def get_feature_names_out(self, input_features=None):
        # Generate the new feature names after the transformation
        new_feature_names = [f"{pair[0]}_x_{pair[1]}" for pair in self.interaction_pairs]
        if input_features is None:
            return new_feature_names
        else:
            # Append the new feature names to the original feature names if provided
            return input_features.tolist() + new_feature_names

def impute_missing_values(df):
 
    # Impute 'None' for categorical basement-related features
    basement_cat_cols = ['bsmtexposure', 'bsmtfintype1', 'bsmtcond', 'bsmtqual']
    df[basement_cat_cols] = df[basement_cat_cols].fillna('None')

    # Impute 'TA' for the only missing value in MSZoning
    df['mszoning'] = df['mszoning'].fillna('TA')
    return df

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Nothing to fit, return self
        return self

    def transform(self, X):
        # Apply the imputation logic
        X = impute_missing_values(X.copy())
        return X

    def get_feature_names_out(self, input_features=None):
        # If input_features is not provided, try using the columns from the input DataFrame
        # Otherwise, just return the input_features as they are
        if input_features is None:
            try:
                return X.columns
            except AttributeError:
                raise ValueError("Input features must be defined when input is not a DataFrame.")
        else:
            return input_features