# Copyright (C) Microsoft Corporation. All rights reserved.
# 23456789012345678901234567890123456789012345678901234567890123456789012345678

from __future__ import print_function
import os
import warnings
import argparse
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.externals import joblib
from ItemSelector import ItemSelector
from timer import elapsed
from label_rank import label_rank

warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='lightgbm')

if __name__ == '__main__':

    # Define the arguments.
    parser = argparse.ArgumentParser(description='Fit and evaluate a model'
                                     ' based on train-test datasets.')
    parser.add_argument('-d', '--data', help='the training dataset name',
                        default='balanced_pairs_train.tsv')
    parser.add_argument('-t', '--test', help='the test dataset name',
                        default='balanced_pairs_test.tsv')
    parser.add_argument('-i', '--estimators',
                        help='the number of learner estimators',
                        type=int, default=4000)
    parser.add_argument('-n', '--ngrams',
                        help='the maximum size of word ngrams',
                        type=int, default=1)
    parser.add_argument('-g', '--google',
                        help='use Google News Word2Vec embedding',
                        action='store_true')
    parser.add_argument('-f', '--fasttext',
                        help='use Wikipedia FastText embedding',
                        action='store_true')
    parser.add_argument('--topics',
                        help='the number of topics in topic vectors',
                        type=int, default=0)
    parser.add_argument('-u', '--unweighted',
                        help='do not use instance weights',
                        action='store_true')
    parser.add_argument('--min_child_samples',
                        help='the minimum number of samples in a child(leaf)',
                        type=int, default=20)
    parser.add_argument('-m', '--match',
                        help='the maximum number of duplicate matches',
                        type=int, default=20)
    parser.add_argument('--outputs', help='the outputs directory',
                        default='.')
    parser.add_argument('--inputs', help='the inputs directory',
                        default='.')
    parser.add_argument('-s', '--save', help='save the model',
                        action='store_true')
    parser.add_argument('--model', help='the model file', default='model.pkl')
    parser.add_argument('--instances', help='the instances file',
                        default='inst.txt')
    parser.add_argument('--labels', help='the labels file',
                        default='labels.txt')
    parser.add_argument('-r', '--rank',
                        help='the maximum rank of correct answers',
                        type=int, default=3)
    parser.add_argument('-b', '--batch_size',
                        help='the number of dupes in each batch',
                        type=int, default=100)
    parser.add_argument('-v', '--verbose',
                        help='the verbosity of the estimator',
                        type=int, default=-1)
    args = parser.parse_args()

    # The input data.
    inputs_path = args.inputs
    data_path = os.path.join(inputs_path, args.data)
    test_path = os.path.join(inputs_path, args.test)

    # The output data.
    outputs_path = args.outputs
    model_path = os.path.join(outputs_path, args.model)
    instances_path = os.path.join(outputs_path, args.instances)
    labels_path = os.path.join(outputs_path, args.labels)

    # Create the outputs folder.
    os.makedirs(outputs_path, exist_ok=True)

    # Load the data.
    print('Reading {}'.format(data_path))
    train = pd.read_csv(data_path, sep='\t', encoding='latin1')

    # Limit the number of training duplicate matches.
    train = train[train.n < args.match]

    # The input data columns.
    feature_columns = ['Text_x', 'Text_y']
    label_column = 'Label'
    group_column = 'Id_x'
    answerid_column = 'AnswerId_y'
    name_columns = ['Id_x', 'Id_y']

    # Report on the dataset.
    print('train: {:,} rows with {:.2%} matches'.format(
        train.shape[0], train[label_column].mean()))

    # Compute instance weights.
    weight_column = 'Weight'
    if args.unweighted:
        weight = pd.Series([1.0], train[label_column].unique())
    else:
        label_counts = train[label_column].value_counts()
        weight = train.shape[0]/(label_counts.shape[0]*label_counts)
    train[weight_column] = train[label_column].apply(lambda x: weight[x])

    # Collect the ordered AnswerId.
    labels = sorted(train[answerid_column].unique())
    label_order = pd.DataFrame({'label': labels})

    # Select and format the training data.
    train_X = train[feature_columns]
    train_y = train[label_column]
    sample_weight = train[weight_column]
    groups = train[group_column]
    names = train[name_columns]

    # Select the training hyperparameters.
    n_estimators = args.estimators
    min_child_samples = args.min_child_samples
    estimator = lgb.LGBMClassifier(n_estimators=n_estimators,
                                   min_child_samples=min_child_samples,
                                   verbose=args.verbose)
    if args.ngrams > 0:
        ngram_range = (1, args.ngrams)
    else:
        ngram_range = None
    assert ngram_range is not None

    # The featurization pipeline(s) for each text column.
    featurization = [
        (column,
         make_pipeline(ItemSelector(column),
                       text.TfidfVectorizer(ngram_range=ngram_range)))
        for column in feature_columns]
    features = FeatureUnion(featurization)

    # The model pipeline.
    model = Pipeline([
        ('features', features),
        ('model', lgb.LGBMClassifier(n_estimators=n_estimators))
    ])

    # Fit the model.
    elapsed(model.fit)(train_X, train_y, model__sample_weight=sample_weight)

    # write the model to file.
    if args.save:
        joblib.dump(model, model_path)
        print('{}: {.0f} MB'.format(
            model_path, os.path.getsize(model_path)/(2**20)))

    # Read the test data.
    print('Reading {}'.format(test_path))
    test = pd.read_csv(test_path, sep='\t', encoding='latin1')
    print('test {:,} rows with {:.2%} matches'.format(
        test.shape[0], test[label_column].mean()))

    # Collect the model predictions.
    test_X = test[feature_columns]
    test['probabilities'] = elapsed(model.predict_proba)(test_X)[:, 1]

    # Order the testing data by dupe Id and question AnswerId.
    test.sort_values([group_column, answerid_column], inplace=True)

    # Extract the ordered probabilities.
    probabilities = (
        test.probabilities
        .groupby(test[group_column], sort=False)
        .apply(lambda x: tuple(x.values)))

    # Get the individual records.
    output_columns_x = ['Id_x', 'AnswerId_x', 'Text_x']
    test_score = (test[output_columns_x]
                  .drop_duplicates()
                  .set_index(group_column))
    test_score['probabilities'] = probabilities
    test_score.reset_index(inplace=True)
    test_score.columns = ['Id', 'AnswerId', 'Text', 'probabilities']

    # Rank the correct answers.
    test_score['Ranks'] = test_score.apply(lambda x:
                                           label_rank(x.AnswerId,
                                                      x.probabilities,
                                                      label_order.label),
                                           axis=1)

    # Compute the number of correctly ranked answers
    for i in range(1, args.rank+1):
        print('Accuracy @{} = {:.2%}'.format(
            i, (test_score['Ranks'] <= i).mean()))
    mean_rank = test_score['Ranks'].mean()
    print('Mean Rank {:.4f}'.format(mean_rank))

    # Write the scored instances.
    test_score.to_csv(instances_path, sep='\t', index=False,
                      encoding='latin1')
    label_order.to_csv(labels_path, sep='\t', index=False)
