# Copyright (C) Microsoft Corporation.  All rights reserved.
"""Prepare the data for pairing duplicate and original questions."""

import os
import pandas as pd
import re
import gzip
import requests
import argparse
import math
from timer import elapsed
from preprocessing import read_csv_gz, clean_text
from etl import round_sample_strat, random_merge


if __name__ == '__main__':

    # Define the arguments.
    parser = argparse.ArgumentParser(description='Prepare datasets.')
    parser.add_argument('-t', '--test_size',
                        help='the size of the test set',
                        type=float, default=0.21)
    parser.add_argument('--min_text',
                        help='the minimum length of clean text',
                        type=int, default=150)
    parser.add_argument('--min_dupes',
                        help='the minimum number of dupes per question',
                        type=int, default=12)
    parser.add_argument('--max_dupes',
                        help='the maximum number of dupes per question',
                        type=int, default=0)
    parser.add_argument('-m', '--match',
                        help='the maximum number of duplicate matches',
                        type=int, default=20)
    parser.add_argument('--outputs', help='the outputs directory',
                        default='outputs')
    args = parser.parse_args()

    # URLs to original questions, duplicate questions, and answers.
    data_url = 'https://bostondata.blob.core.windows.net/stackoverflow/{}'
    questions_url = data_url.format('orig-q.tsv.gz')
    dupes_url = data_url.format('dup-q.tsv.gz')
    answers_url = data_url.format('ans.tsv.gz')

    # Create the outputs folder.
    outputs_path = args.outputs
    os.makedirs(outputs_path, exist_ok=True)

    # Load datasets.
    questions = read_csv_gz(questions_url, names=(
        'Id', 'AnswerId', 'Text0', 'CreationDate'))
    dupes = read_csv_gz(dupes_url, names=(
        'Id', 'AnswerId', 'Text0', 'CreationDate'))
    answers = read_csv_gz(answers_url, names=('Id', 'Text0'))

    # Clean up all text, and keep only data with some clean text.
    for df in (questions, dupes, answers):
        df['Text'] = df.Text0.apply(clean_text).str.lower()
    questions = questions[questions.Text.str.len() > 0]
    answers = answers[answers.Text.str.len() > 0]
    dupes = dupes[dupes.Text.str.len() > 0]

    # First, remove dupes that are questions, then remove duplicated
    # questions and dupes.
    dupes = dupes[~dupes.index.isin(questions.index)]
    questions = questions[~questions.index.duplicated(keep='first')]
    dupes = dupes[~dupes.index.duplicated(keep='first')]

    # Keep only questions with answers and dupes, answers to
    # questions, and dupes of questions.
    questions = questions[questions.AnswerId.isin(answers.index)
                          & questions.AnswerId.isin(dupes.AnswerId)]
    answers = answers[answers.index.isin(questions.AnswerId)]
    dupes = dupes[dupes.AnswerId.isin(questions.AnswerId)]

    # Verify data integrity.
    assert questions.AnswerId.isin(answers.index).all()
    assert answers.index.isin(questions.AnswerId).all()
    assert questions.AnswerId.isin(dupes.AnswerId).all()
    assert dupes.AnswerId.isin(questions.AnswerId).all()

    # Report on the data.
    print('Text statistics:')
    print(pd.DataFrame([questions.Text.str.len().describe()
                        .rename('questions'),
                        answers.Text.str.len().describe()
                        .rename('answers'),
                        dupes.Text.str.len().describe()
                        .rename('dupes')]))
    print('\nDuplication statistics:')
    print(pd.DataFrame([dupes.AnswerId.value_counts().describe()
                        .rename('duplications')]))
    print('\nLargest class: {:.2%}'.format(
        dupes.AnswerId.value_counts().max()
        / dupes.shape[0]))

    # Reset each dataframe's index.
    questions.reset_index(inplace=True)
    answers.reset_index(inplace=True)
    dupes.reset_index(inplace=True)

    # Apply the minimum text length to questions and dupes.
    questions = questions[questions.Text.str.len() >= args.min_text]
    dupes = dupes[dupes.Text.str.len() >= args.min_text]

    # Keep only questions with dupes, and dupes of questions.
    label_column = 'AnswerId'
    questions = questions[questions[label_column].isin(dupes[label_column])]
    dupes = dupes[dupes[label_column].isin(questions[label_column])]

    # Restrict the questions to those with a minimum number of dupes.
    answerid_count = dupes.groupby(label_column)[label_column].count()
    answerid_min = answerid_count.index[answerid_count >= args.min_dupes]
    questions = questions[questions[label_column].isin(answerid_min)]
    dupes = dupes[dupes[label_column].isin(answerid_min)]

    # Limit the dupes to a maximum number per label choosing at random
    # what to keep.
    if args.max_dupes >= args.min_dupes:
        dupes = dupes.groupby(label_column).apply(max_sample,
                                                  max=args.max_dupes)

    # Verify data integrity.
    assert questions[label_column].isin(dupes[label_column]).all()
    assert dupes[label_column].isin(questions[label_column]).all()

    # Report on the data.
    print('Restrictions: min_text={}, min_dupes={}'.format(
        args.min_text, args.min_dupes))
    print('Restricted text statistics:')
    print(pd.DataFrame([questions.Text.str.len().describe()
                        .rename('questions'),
                        dupes.Text.str.len().describe()
                        .rename('dupes')]))
    print('\nRestricted duplication statistics:')
    print(pd.DataFrame([dupes[label_column].value_counts().describe()
                        .rename('duplications')]))
    print('\nRestricted largest class: {:.2%}'.format(
        dupes[label_column].value_counts().max()
        / dupes.shape[0]))

    # Split dupes into train and test ensuring at least one of each
    # label class is in test.
    dupes_test = round_sample_strat(dupes, dupes[label_column],
                                    frac=args.test_size)
    dupes_train = dupes[~dupes.Id.isin(dupes_test.Id)]

    assert (dupes_test[label_column].unique().shape[0]
            == dupes[label_column].unique().shape[0])

    # Create questions and dupes training and test datasets.
    # Questions only go in the training data.
    balanced_columns = ['Id', 'AnswerId', 'Text']
    balanced_train = pd.concat([questions, dupes_train])[balanced_columns]
    balanced_test = dupes_test[balanced_columns]
    print('balanced_train {:,}. Largest class: {:.2%}'.format(
        balanced_train.shape[0],
        balanced_train[label_column].value_counts().max()
        / balanced_train.shape[0]))
    print('balanced_test {:,}. Largest class: {:.2%}'.format(
        balanced_test.shape[0],
        balanced_test[label_column].value_counts().max()
        / balanced_test.shape[0]))

    balanced_train_path = os.path.join(outputs_path, 'balanced_train.tsv')
    print('Writing {:,} to {}'.format(balanced_train.shape[0],
                                      balanced_train_path))
    balanced_train.to_csv(balanced_train_path, sep='\t', header=True,
                          index=False)

    balanced_test_path = os.path.join(outputs_path, 'balanced_test.tsv')
    print('Writing {:,} to {}'.format(balanced_test.shape[0],
                                      balanced_test_path))
    balanced_test.to_csv(balanced_test_path, sep='\t', header=True,
                         index=False)

    # The relevant columns for text pairs data.
    balanced_pairs_columns = ['Id_x', 'AnswerId_x', 'Text_x', 'Id_y', 'Text_y',
                              'AnswerId_y', 'Label', 'n']

    # Use AnswerId to pair each training dupe with its matching
    # question and also with N-1 questions not its match.
    balanced_pairs_train = elapsed(random_merge)(dupes_train, questions,
                                                 N=args.match)

    # Label records by matching AnswerIds.
    balanced_pairs_train['Label'] = (
        balanced_pairs_train.AnswerId_x == balanced_pairs_train.AnswerId_y
    ).astype(int)

    # Keep only the relevant data.
    balanced_pairs_train = balanced_pairs_train[balanced_pairs_columns]

    # Sort the data by dupe ID and Label.
    balanced_pairs_train.sort_values(by=['Id_x', 'Label'],
                                     ascending=[True, False],
                                     inplace=True)

    # Use AnswerId to pair each training dupe with all questions.
    balanced_pairs_test = elapsed(random_merge)(dupes_test, questions,
                                                N=questions.shape[0])

    # Label records by matching AnswerIds.
    balanced_pairs_test['Label'] = (
        balanced_pairs_test.AnswerId_x == balanced_pairs_test.AnswerId_y
    ).astype(int)

    # Keep only the relevant data.
    balanced_pairs_test = balanced_pairs_test[balanced_pairs_columns]

    # Sort the data by dupe ID and Label.
    balanced_pairs_test.sort_values(by=['Id_x', 'Label'],
                                    ascending=[True, False],
                                    inplace=True)

    # Report on the datasets.
    print('balanced_pairs_train: {:,} rows with {:.2%} matches'.format(
        balanced_pairs_train.shape[0], balanced_pairs_train.Label.mean()))

    print('balanced_pairs_test: {:,} rows with {:.2%} matches'.format(
        balanced_pairs_test.shape[0], balanced_pairs_test.Label.mean()))

    # Save the data.
    balanced_pairs_train_path = os.path.join(outputs_path,
                                             'balanced_pairs_train.tsv')
    print('Writing {:,} to {}'.format(balanced_pairs_train.shape[0],
                                      balanced_pairs_train_path))
    balanced_pairs_train.to_csv(balanced_pairs_train_path, sep='\t',
                                header=True, index=False)

    balanced_pairs_test_path = os.path.join(outputs_path,
                                            'balanced_pairs_test.tsv')
    print('Writing {:,} to {}'.format(balanced_pairs_test.shape[0],
                                      balanced_pairs_test_path))
    balanced_pairs_test.to_csv(balanced_pairs_test_path, sep='\t',
                               header=True, index=False)
