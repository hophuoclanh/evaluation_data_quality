import numpy as np
from imblearn.under_sampling import TomekLinks
from cleanlab.filter import find_label_issues

def feature_correlation(dataframe):
    """
    Calculate the QoD^D_FC metric for a given dataframe.

    Args:
    - dataframe (pd.DataFrame): The input data.

    Returns:
    - float: The QoD^D_FC value.
    """
    # Drop non-numeric columns
    dataframe = dataframe.select_dtypes(include=[np.number])

    correlations = dataframe.corr().abs().unstack().sort_values(kind="quicksort", ascending=False)
    # Exclude self correlations
    correlations = correlations[correlations < 1]

    N = len(dataframe.columns)
    # Adjusted denominator based on unique pairs of features
    adjusted_denominator = N * (N - 1) / 2

    # Compute the metric
    QoD_FC = 1 - correlations.sum() / adjusted_denominator
    return QoD_FC


def feature_relevance(dataframe, label_column, alpha=0.5, beta=0.5):
    """
    Calculate the QoD^D_FR metric for a given dataframe and label column.

    Args:
    - dataframe (pd.DataFrame): The input data.
    - label_column (str): The column name of the label.
    - alpha (float): The alpha parameter.
    - beta (float): The beta parameter.

    Returns:
    - float: The QoD^D_FR value.
    """
    # For simplicity, we'll use feature importances from a decision tree
    from sklearn.tree import DecisionTreeRegressor

    X = dataframe.drop(label_column, axis=1)
    y = dataframe[label_column]

    model = DecisionTreeRegressor()
    model.fit(X, y)
    importances = model.feature_importances_

    QoD_FR = alpha * (1 - np.var(importances)) + beta * np.mean(sorted(importances)[-3:])
    return QoD_FR


def completeness(dataframe):
    """
    Calculate the QoD^D_Com metric for a given dataframe.

    Args:
    - dataframe (pd.DataFrame): The input data.

    Returns:
    - float: The QoD^D_Com value.
    """
    null_count = dataframe.isnull().sum().sum()
    total_count = np.prod(dataframe.shape)

    QoD_Com = 1 - null_count / total_count
    return QoD_Com

def label_purity(X, y, pred_probs):
    """
        Calculate the label purity metric for a given set of data points, true labels, and predicted probabilities.

        Args:
        - X (array-like): The input data points.
        - y (array-like): The true labels of the data points.
        - pred_probs (array-like): The predicted probabilities for each data point.

        Returns:
        - float: The label purity value, computed as QoD_lp.
        """
    # Using is_tomek_link from imbalanced-learn
    tl = TomekLinks()
    _, y_resampled = tl.fit_resample(X, y)

    # Get indices of Tomek links
    tomek_indices = tl.sample_indices_

    # Using find_label_issues from cleanlab
    label_issue_indices = find_label_issues(
        labels=y,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence'
    )

    intersection = len(set(tomek_indices) & set(label_issue_indices))

    # Compute the label purity as per the formula
    QoD_LP = 1 - (intersection / len(y))

    return QoD_LP