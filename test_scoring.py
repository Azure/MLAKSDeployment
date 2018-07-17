# Copyright (C) Microsoft Corporation. All rights reserved.
# 23456789012345678901234567890123456789012345678901234567890123456789012345678

import pandas as pd
from duplicate_model import DuplicateModel


if __name__ == '__main__':

    model_path = 'model.pkl'
    questions_path = 'questions.tsv'
    dupes_test_path = 'dupes_test.tsv'

    model = DuplicateModel(model_path, questions_path)

    dupes_test = pd.read_csv(dupes_test_path, sep='\t', encoding='latin1')

    dupes_scores = dupes_test.Text.apply(model.score)
