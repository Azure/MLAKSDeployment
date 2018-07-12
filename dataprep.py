# Copyright (C) Microsoft Corporation.  All rights reserved.
"""Prepare the data for pairing duplicate and original questions."""

import os
import pandas as pd
import argparse
from timer import elapsed
from preprocessing import read_csv_gz, clean_text
from etl import round_sample_strat, random_merge


if __name__ == '__main__':

    # Define the arguments.
    parser = argparse.ArgumentParser(description='Prepare datasets.')
    parser.add_argument('-t', '--test_size',
                        help='the size of the test set',
                        type=float, default=0.2)
    parser.add_argument('-m', '--match',
                        help='the maximum number of duplicate matches',
                        type=int, default=40)
    parser.add_argument('--min_text',
                        help='the minimum length of clean text',
                        type=int, default=150)
    parser.add_argument('--min_dupes',
                        help='the minimum number of dupes per question',
                        type=int, default=1)
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
    dupes = dupes[~dupes.Id.isin(questions.Id)]
    questions = questions[~questions.Id.duplicated(keep='first')]
    dupes = dupes[~dupes.Id.duplicated(keep='first')]

    # Keep only questions with answers and dupes, answers to
    # questions, and dupes of questions.
    questions = questions[questions.AnswerId.isin(answers.Id)
                          & questions.AnswerId.isin(dupes.AnswerId)]
    answers = answers[answers.Id.isin(questions.AnswerId)]
    dupes = dupes[dupes.AnswerId.isin(questions.AnswerId)]

    # Verify data integrity.
    assert questions.AnswerId.isin(answers.Id).all()
    assert answers.Id.isin(questions.AnswerId).all()
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

    # Output questions, answers, and dupes.
    questions_path = os.path.join(outputs_path, 'questions.tsv')
    print('Writing {:,} to {}'.format(questions.shape[0], questions_path))
    questions.to_csv(questions_path, sep='\t', header=True, index=False)

    answers_path = os.path.join(outputs_path, 'answers.tsv')
    print('Writing {:,} to {}'.format(answers.shape[0], answers_path))
    answers.to_csv(answers_path, sep='\t', header=True, index=False)

    dupes_path = os.path.join(outputs_path, 'dupes.tsv')
    print('Writing {:,} to {}'.format(dupes.shape[0], dupes_path))
    dupes.to_csv(dupes_path, sep='\t', header=True, index=False)

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

    questions_train_path = os.path.join(outputs_path, 'questions_train.tsv')
    print('Writing {:,} to {}'.format(questions.shape[0],
                                      questions_train_path))
    questions.to_csv(questions_train_path, sep='\t', header=True, index=False)

    dupes_train_path = os.path.join(outputs_path, 'dupes_train.tsv')
    print('Writing {:,} to {}'.format(dupes_train.shape[0], dupes_train_path))
    dupes_train.to_csv(dupes_train_path, sep='\t', header=True, index=False)

    dupes_test_path = os.path.join(outputs_path, 'dupes_test.tsv')
    print('Writing {:,} to {}'.format(dupes_test.shape[0], dupes_test_path))
    dupes_test.to_csv(dupes_test_path, sep='\t', header=True, index=False)

    # Create questions and dupes training and test datasets.
    # Questions only go in the training data.
    QnD_columns = ['Id', 'AnswerId', 'Text']
    QnD_train = pd.concat([questions, dupes_train])[QnD_columns]
    QnD_test = dupes_test[QnD_columns]
    print('QnD_train {:,}. Largest class: {:.2%}'.format(
        QnD_train.shape[0],
        QnD_train[label_column].value_counts().max()
        / QnD_train.shape[0]))
    print('QnD_test {:,}. Largest class: {:.2%}'.format(
        QnD_test.shape[0],
        QnD_test[label_column].value_counts().max()
        / QnD_test.shape[0]))

    QnD_train_path = os.path.join(outputs_path, 'QnD_train.tsv')
    print('Writing {:,} to {}'.format(QnD_train.shape[0], QnD_train_path))
    QnD_train.to_csv(QnD_train_path, sep='\t', header=True, index=False)

    QnD_test_path = os.path.join(outputs_path, 'QnD_test.tsv')
    print('Writing {:,} to {}'.format(QnD_test.shape[0], QnD_test_path))
    QnD_test.to_csv(QnD_test_path, sep='\t', header=True, index=False)

    # The relevant columns for text pairs data.
    pairs_columns = ['Id_x', 'AnswerId_x', 'Text_x', 'Id_y', 'Text_y',
                     'AnswerId_y', 'Label', 'n']

    # Use AnswerId to pair each training dupe with its matching
    # question and also with N-1 questions not its match.
    pairs_train = elapsed(random_merge)(dupes_train, questions, N=args.match)

    # Label records by matching AnswerIds.
    pairs_train['Label'] = (pairs_train.AnswerId_x == pairs_train.AnswerId_y
                            ).astype(int)

    # Keep only the relevant data.
    pairs_train = pairs_train[pairs_columns]

    # Sort the data by dupe ID and Label.
    pairs_train.sort_values(by=['Id_x', 'Label'], ascending=[True, False],
                            inplace=True)

    # Use AnswerId to pair each training dupe with all questions.
    pairs_test = elapsed(random_merge)(dupes_test, questions,
                                       N=questions.shape[0])

    # Label records by matching AnswerIds.
    pairs_test['Label'] = (pairs_test.AnswerId_x == pairs_test.AnswerId_y
                           ).astype(int)

    # Keep only the relevant data.
    pairs_test = pairs_test[pairs_columns]

    # Sort the data by dupe ID and Label.
    pairs_test.sort_values(by=['Id_x', 'Label'], ascending=[True, False],
                           inplace=True)

    # Report on the datasets.
    print('pairs_train: {:,} rows with {:.2%} matches'.format(
        pairs_train.shape[0], pairs_train.Label.mean()))

    print('pairs_test: {:,} rows with {:.2%} matches'.format(
        pairs_test.shape[0], pairs_test.Label.mean()))

    # Save the data.
    pairs_train_path = os.path.join(outputs_path, 'pairs_train.tsv')
    print('Writing {:,} to {}'.format(pairs_train.shape[0], pairs_train_path))
    pairs_train.to_csv(pairs_train_path, sep='\t', header=True, index=False)

    pairs_test_path = os.path.join(outputs_path, 'pairs_test.tsv')
    print('Writing {:,} to {}'.format(pairs_test.shape[0], pairs_test_path))
    pairs_test.to_csv(pairs_test_path, sep='\t', header=True, index=False)
