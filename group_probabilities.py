# Copyright (C) Microsoft Corporation.  All rights reserved.


def group_probabilities(x, output_columns, group_column,
                        probabilities_column='probabilities'):
    """Collect each group's probabilities into a list, and return a
    dataframe with one row per group and a column of its probabilities

    """
    probabilities = (
        x[probabilities_column]
        .groupby(x[group_column], sort=False)
        .apply(lambda x: x.values.tolist()))
    result = (x[output_columns]
              .drop_duplicates()
              .set_index(group_column))
    result[probabilities_column] = probabilities
    result.reset_index(inplace=True)
    result.columns = output_columns + [probabilities_column]
    return result
