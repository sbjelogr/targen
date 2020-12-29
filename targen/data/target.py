from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from functools import reduce



def get_target_and_contributions(df, *, expressions = None, imbalance = 0.1, drop_features = True):
    """Get the target and all the contribution by defining the relationships via the expression dictionary.

    Args:
        df: pd.DataFrame, input data
        expressions: dictionary, where keys are the names of the contributions, and the values are the relative
            expressions. Supported key value pairs are as follows:
            - 'linear' (value -> string containing the expression to apply)
            - 'non-linear' (value -> string containing the expression to apply)
            - 'interaction' (value -> string containing the expression to apply)
            - 'conditional' (value -> dictionary, where the keys are the conditions (in string format)
                and the values are tuples, where the first term is the expression or value to assign in case the
                 condition is true, while the second term is the expression or value to assign in case the condition is
                false)
            - 'uniform_noise' (value-> dictionary with kwargs for the function score_uniform_noise)
            - 'gaussian noise' (value-> dictionary with kwargs for the function score_gaussian_noise)
        imbalance: float, the fraction of targets that should be labelled with class=1
        drop_features: boolean, default True. If False, the output will contain the columns of the input dataframe,
        if True, it will not contain them.

    Returns: pd.DataFrame, with the same index as the input dataframe df. For every expression in the expressions
        dictionary, a column containing the contribution of those terms is evaluated. The naming convention is
        'score_<expression_name>'.
        'score_final' contains the sum of all the single expression contributions, and it's used to define the target.
        If drop_features is set to False, the output dataframe will also contain the inputs from the inout dataset.

    Example:
        from sklearn.datasets import make_classification
        import pandas as pd
        import numpy as np
        from data import target

        X, dummy = make_classification( n_samples=100,n_features=20)

        data = pd.DataFrame(X, columns = [f'col_{ix}' for ix in range(X.shape[1])])

        expressions = {
            'linear': '-0.5*col_0 + 2.*col_1 + 0.7*col_7 -0.4*col_14',
            'non_linear': '-0.7*col_1**1.5 + 0.2*sin(col_7)+ 0.9*log(col_19) -0.1*col_13**2',
            'interaction': '0.05*col_3*col_4 -0.1*(col_7/col_11)',
            'conditional': {
                'col_0>0':(
                    '-0.5*col_0 + 2.*col_1 + 0.7*col_7 -0.4*col_14', #True
                    0 #False
                ),
                'col_2>0.15':(
                    '-0.5*col_0 + 2.*col_13 + 0.2*col_17 -0.7*col_11',#True
                    '-0.5*col_1' #False
                )
            },
            'uniform_noise': {
                'weight':0.5
            },
            'gaussian_noise': {
                'weight':0.5,
                'mu_gaus': 0
            }
        }

        target.get_target_and_contributions(data, expressions=expressions. imbalance = 0.2)

    """

    df_out = df.copy()
    for type_contr in expressions.keys():
        if type_contr in ['linear', 'non_linear', 'interaction']:
            df_out[f'score_{type_contr}'] = score_from_expression(df, expr = expressions[type_contr])
        elif type_contr == 'conditional':
            df_out[f'score_{type_contr}'] = score_with_condition(df, cond_expr = expressions[type_contr])
        elif type_contr=='uniform_noise':
            df_out[f'score_{type_contr}'] = score_uniform_noise(df, **expressions[type_contr])
        elif type_contr=='gaussian_noise':
            df_out[f'score_{type_contr}'] = score_gaussian_noise(df, **expressions[type_contr])
        else:
            raise ValueError(f'Unsupported contribution {type_contr}')

    df_out['score_total'] = df_out[[col for col in df_out.columns if col.startswith('score')]].sum(axis=1)

    df_out['y'] = define_target( df_out['score_total'], imbalance_fraction=imbalance)

    if drop_features:
        return df_out.drop(df.columns, axis =1)
    else:
        return df_out



def define_target(score, *, imbalance_fraction=0.1):
    """Given the input score, define the target output, with the defined imbalance fraction.

    Args:
        score: pd.Series, score to compute the target
        imbalance_fraction: float, fraction of data to be ise

    Returns: pd.Series with 0 and 1, to be used

    """
    return pd.Series(
        [1 if ix_score else 0 for ix_score in score > np.quantile(score, (1-imbalance_fraction))],
        index = score.index
    )



def score_from_expression(df,*,expr=None, scaler='minmax', shift = 0):
    """Defines the score based on the expression.

    Args:
        df: pd.DataFrame, input data
        expr: string, contains the expression
        scaler: type of scaler to be used. Choose between 'minmax', 'standard' or None
        shift: float, uniform shift to apply to all the score expressions

    Returns: pd.Series, scores coming from this contribution

    Example:
        from sklearn.datasets import make_classification
        import pandas as pd
        import numpy as np
        from data import target

        X, dummy = make_classification( n_samples=100,n_features=20)

        data = pd.DataFrame(X, columns = [f'col_{ix}' for ix in range(X.shape[1])])

        expr = '-0.5*col_0 + 2.*col_1 + 0.7*col_7 -0.4*col_14'

        score = target.score_from_expression(data, expr=expr)

    """

    df_feats = _rescale_data(df, scaler=scaler)

    score = None
    if expr is not None:
        score = df_feats.eval(expr)
    else:
        score = pd.Series(np.zeros(shape=df.feats.shape[0],), index = df_feats.index)

    return score + shift


def score_with_condition(df, *, cond_expr=None,scaler='minmax', shift = 0):
    """

    Args:
        df: pd.DataFrame, input data
        cond_expr: dict, contains the condition and the expressions to apply, see example below.
        scaler: type of scaler to be used. Choose between 'minmax', 'standard' or None
        shift: float, uniform shift to apply to all the score expressions

    Returns: pd.Series, scores coming from this contribution

    Example:
        from sklearn.datasets import make_classification
        import pandas as pd
        import numpy as np
        from data import target

        X, dummy = make_classification( n_samples=100,n_features=20)

        data = pd.DataFrame(X, columns = [f'col_{ix}' for ix in range(X.shape[1])])

        cond_expr = {
            'col_0>0':(
                '-0.5*col_0 + 2.*col_1 + 0.7*col_7 -0.4*col_14', #If condition is True, do this
                0  #If condition is False, do this
            ),
            'col_2>0.15':(
                '-0.5*col_0 + 2.*col_13 + 0.2*col_17 -0.7*col_11', #If condition is True, do this
                '-0.5*col_1' #If condition is False, do this
            )
        }

        cond_scores = target.score_with_condition(data,cond_expr = cond_expr)
    """

    df_feats = _rescale_data(df, scaler=scaler)


    score = pd.Series(0, index = df.index)
    for k in cond_expr.keys():
        score+= _conditional_score(df, df_feats, k,cond_expr[k])

        return score + shift


def score_uniform_noise(df, *, weight=0, min =-1, max = 1):
    """Define uniform noise contribution

    Args:
        df: pd.DataFrame, input data
        weight: float, weight to assign to this value
        min: minimum value of the uniform distribution
        max: maximum value of the uniform distribution

    Returns: pd.Series, scores coming from this contribution
    """
    return pd.Series(
        weight *np.random.uniform(low= min, high = max, size=df.shape[0]),
        index = df.index
    )


def score_gaussian_noise(df, *, weight=0, mu_gaus = 0, sigma_gaus = 1):
    """Define gaussian noise contribution

    Args:
        df: pd.DataFrame, input data
        weight: float, weight to assign to this value
        mu_gaus: float, gaussian mean
        sigma_gaus: float, gaussian standard deviation

    Returns: pd.Series, scores coming from this contribution
    """

    return pd.Series(
        weight * np.random.normal(loc = mu_gaus, scale = sigma_gaus, size=df.shape[0]),
        index=df.index
    )


def _conditional_score(df, df_feats, cond, value):
    """Helper function to assign the conditional scores."""

    # get the true query
    if isinstance(value[0],str):
        true_q = df_feats.loc[df.query(cond).index].eval(value[0])
    elif isinstance(value[0],float) or isinstance(value[0],int):
        index = df.query(cond).index
        true_q = pd.Series(value[0], index = index)
    else:
        raise ValueError(f"Not possible to process the condition '{value[0]}'. See docstring example")

    if isinstance(value[1],str):
        false_q = df_feats.loc[df.query(f'not {cond}').index].eval(value[1])
    elif isinstance(value[1],float) or isinstance(value[1],int) :
        index = df.query(f'not {cond}').index
        false_q = pd.Series(value[1], index = index)
    else:
        raise ValueError(f"Not possible to process the condition '{value[1]}'. See docstring example")

    return pd.concat([true_q,false_q]).sort_index()

def _rescale_data(df,*, scaler =None, columns = None):
    """Helper function for rescaling the data.
    """

    if scaler is not None:
        if scaler=='minmax':
            scaler_trans = MinMaxScaler(feature_range=(0.00000001, 1))
        elif scaler=='standard':
            scaler_trans = StandardScaler()
        else:
            raise NotImplementedError(f"Scaler '{scaler}' not supported. Must be 'minmax', 'standard' or None")
        if columns is None:
            df_feats = pd.DataFrame(
                scaler_trans.fit_transform(df),
                columns=df.columns,
                index=df.index
            )
        else:
            df_feats = pd.DataFrame(
                scaler_trans.fit_transform(df[columns]),
                columns=columns,
                index=df.index
            )
    else: df_feats = df.copy()

    return df_feats

