import pandas as pd
import numpy as np


def levelOfConsistency(df):
    """
    Compute level of consistency

    This function computes the level of consistency of the dataset in data frame
    format. The last column is considered as the decision, as specified in the
    requirement.

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame representation of the dataset

    Returns
    -------
    float
        level of consistency score
    """
    attribute_colnames = df.columns[:-1]
    decision_colname = df.columns[-1]
    num_data = df.shape[0]

    df_attribute_duplicate = df[df.duplicated(attribute_colnames, keep=False)]
    num_inconsistent = df_attribute_duplicate.groupby(attribute_colnames.tolist())[decision_colname].apply(
        lambda x: x.shape[0] if x.unique().shape[0] > 1 else 0).sum() if df_attribute_duplicate.shape[0] > 0 else 0

    return (num_data - num_inconsistent) / num_data


def generateDiscretizedValue(value, point_ranges, min_value, max_value):
    """
    Generate discrete symbol based on range

    This function computes the discrete symbol representation of the value based
    on the point ranges supplied. The representation will be lowerRange..upperRange.
    Note: point_ranges must be sorted already.

    Parameters
    ----------
    value: float
        The value to be transformed
    point_ranges: list
        List of ranges
    min_value: float
        The minimum value of the range
    max_value: float
        The maximum value of the range

    Returns
    -------
    str
        Discretize symbol representation of value
    """
    str_format = "{:.3f}..{:.3f}"
    # No ranges
    if len(point_ranges) == 0:
        return str_format.format(min_value, max_value)

    # Value is below all point_ranges
    if value < point_ranges[0]:
        return str_format.format(min_value, point_ranges[0])

    # value is between point_ranges
    for i in range(1, len(point_ranges)):
        if value < point_ranges[i]:
            return str_format.format(point_ranges[i - 1], point_ranges[i])

    # value is above all point_ranges
    return str_format.format(point_ranges[len(point_ranges) - 1], max_value)


def generateDiscretizedDataFrame(df, chosen_cut_points):
    """
    Generate discretized data frame based on cut_points

    This function generates discretized data frame based on chosen_cut_points.
    The decision column will always be the last column.

    Parameters
    ----------
    df: pandas.DataFrame
        The data frame to be discretized
    chosen_cut_points: dict
        Dictionary of cut points with the attribute names as the dictionary keys

    Returns
    -------
    pandas.DataFrame
        Discretize data frame
    """
    decision_colname = df.columns[-1]
    attribute_colnames = df.columns[:-1]
    numerical_attribute_colnames = df[attribute_colnames].select_dtypes(include="number").columns
    non_numerical_attribute_colnames = attribute_colnames.drop(numerical_attribute_colnames)
    df_discretized = pd.DataFrame()
    df_discretized[non_numerical_attribute_colnames] = df[non_numerical_attribute_colnames]
    for attribute in chosen_cut_points.keys():
        current_attribute = df[attribute]
        unique_current_attribute_values = current_attribute.unique()
        df_discretized[attribute] = current_attribute.apply(generateDiscretizedValue,
                                                            point_ranges=chosen_cut_points[attribute],
                                                            min_value=np.min(unique_current_attribute_values),
                                                            max_value=np.max(unique_current_attribute_values))
    df_discretized[decision_colname] = df[decision_colname]
    return df_discretized
