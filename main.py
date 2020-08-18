import numpy as np
import pandas as pd
import keras
import pickle
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import backend as K
from string import ascii_uppercase as letters
import random
from random import randint
from scipy.stats import ortho_group
from pathlib import Path
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##################################### Alpha Encodings #####################################


def get_distr_j_encoding(k=26, j=3):
    array_dict = {}
    str_dict = {}
    for i, letter in enumerate(letters):
        indexes = np.random.choice(a=k, size=j, replace=False)
        encoding = np.array([1 if i in indexes else 0 for i in range(k)])
        encoding_str = ''.join(str(b) for b in encoding)
        while encoding_str in str_dict.values():
            indexes = np.random.choice(a=k, size=j, replace=False)
            encoding = np.array([1 if i in indexes else 0 for i in range(k)])
            encoding_str = ''.join(str(b) for b in encoding)
        array_dict[letter] = encoding
        str_dict[letter] = encoding_str
    return array_dict, str_dict


def get_normal_encoding(k=26):
    array_dict = {}
    str_dict = {}
    matrix = np.random.normal(loc=0, size=(k, k)) / (k**0.5)
    for i, row in enumerate(matrix):
        encoding_str = ''.join(str(b) for b in row)
        array_dict[letters[i]] = row
        str_dict[letters[i]] = encoding_str
    return array_dict, str_dict

# Define one hot encoding


def get_one_hot_encoding(k=26):
    array_dict = {}
    str_dict = {}
    for i, letter in enumerate(letters):
        encoding = np.zeros(k)
        encoding[i] = 1
        encoding_str = ''.join(str(int(b)) for b in encoding)

        array_dict[letter] = encoding
        str_dict[letter] = encoding_str
    return array_dict, str_dict


def get_distr_encoding(k=26):
    array_dict = {}
    str_dict = {}
    for i, letter in enumerate(letters):
        encoding = np.random.randint(0, 2, k)
        encoding_str = ''.join(str(b) for b in encoding)
        while encoding_str in str_dict.values():
            encoding = np.random.randint(0, 2, k)
            encoding_str = ''.join(str(b) for b in encoding)
        array_dict[letter] = encoding
        str_dict[letter] = encoding_str
    return array_dict, str_dict


def get_orthogonal_encoding(k=26):
    matrix = ortho_group.rvs(dim=k)
    array_dict = {}
    str_dict = {}
    for i, row in enumerate(matrix):
        array_dict[letters[i]] = row
        str_dict[letters[i]] = ''.join(str(c) for c in row)
    return array_dict, str_dict

##################################### Alpha Data #####################################


def create_encoded_words(encoding, words):
    x_test_list = [np.append(encoding[w[0]], encoding[w[1]]) for w in words]
    return np.array(x_test_list)


def create_test_words():
    words = ['AA', 'AB']
    for x in letters[24:26]:
        for y in letters[24:26]:
            words.append(x + y)

    words.append(letters[randint(0, 23)] + 'Y')
    words.append(letters[randint(0, 23)] + 'Z')
    return words


def create_train_words():
    good_words = np.array([i + i for i in letters[:24]])
    bad_words = np.array([x + y for x in letters[:24] for y in letters[:24] if x != y])

    x_train = good_words
    x_train = np.append(x_train, np.random.choice(bad_words, 48, replace=False))
    y_train = np.append(np.ones(24), np.zeros(48))

    return x_train, y_train


def get_alpha_datasets(encoding_f, args={}, seed=1, model_type='FFWD'):
    assert model_type in ['FFWD', 'LSTM']
    random.seed(seed)

    test_words = create_test_words()
    train_words, y_train = create_train_words()

    encoding, str_encoding = encoding_f(**args)

    x_test = create_encoded_words(encoding, test_words)
    x_train = create_encoded_words(encoding, train_words)

    # Add bad word to test set (depending on x_train)
    x_test[1] = x_train[np.where(y_train == 0)[0][0]]
    y_test = np.array([1, 0, 1, 0, 0, 1, 0, 0])

    if model_type == 'LSTM':
        x_train = x_train.reshape(len(x_train), 2, x_train.shape[1] // 2)
        x_test = x_test.reshape(len(x_test), 2, x_test.shape[1] // 2)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)

    return (x_train, y_train), (x_test, y_test), (train_words, test_words)

##################################### MNIST Data #####################################


def create_datasets(df, n_examples=10):
    train_set_list = []
    for n in range(8):
        train_set_list.append(df[df['label'] == n].sample(n_examples, replace=False))

    train_set = pd.concat(train_set_list)
    return train_set, df[~df.index.isin(train_set.index)]


def create_train_nums(train_df):
    x_train = []
    y_train = []
    i = 0
    for row1 in train_df.iterrows():
        for row2 in train_df.iterrows():
            encoding1 = row1[1][[str(i) for i in range(10)]].values
            encoding2 = row2[1][[str(i) for i in range(10)]].values

            if row1[1]['label'] == row2[1]['label']:
                x_train.append(np.concatenate([encoding1, encoding2]))
                y_train.append(1)
            else:
                if (i % 7 == 0) or (i % 7 == 1):
                    x_train.append(np.concatenate([encoding1, encoding2]))
                    y_train.append(0)
            i += 1

    return np.array(x_train), np.array(y_train)


def create_test_nums(train_df, test_df):
    x_test = []
    y_test = []
    labels = []
    numbers = [str(i) for i in range(10)]

    # Train data encodings
    random_num = random.randint(0, 7)
    df = train_df[train_df['label'] == random_num].sample(2, replace=False)
    x_test.append(np.concatenate([df.iloc[0][numbers].values, df.iloc[1][numbers].values]))
    y_test.append(1)
    labels.append("XX")

    random_num = random.randint(0, 7)
    encoding1_train = train_df[train_df['label'] == random_num].sample(1, replace=False).iloc[0][numbers].values
    encoding2_train = train_df[train_df['label'] != random_num].sample(1, replace=False).iloc[0][numbers].values
    x_test.append(np.concatenate([encoding1_train, encoding2_train]))
    y_test.append(0)
    labels.append("XY")

    # Test data encodings
    random_num = random.randint(0, 7)
    df = test_df[test_df['label'] == random_num].sample(2, replace=False)
    x_test.append(np.concatenate([df.iloc[0][numbers].values, df.iloc[1][numbers].values]))
    y_test.append(1)
    labels.append("X'X'")

    random_num = random.randint(0, 7)
    encoding1_valid = test_df[test_df['label'] == random_num].sample(1, replace=False).iloc[0][numbers].values
    encoding2_valid = test_df[test_df['label'] != random_num].sample(1, replace=False).iloc[0][numbers].values
    x_test.append(np.concatenate([encoding1_valid, encoding2_valid]))
    y_test.append(0)
    labels.append("X'Y'")

    eight_nine_df = pd.concat([
        test_df[test_df['label'] == 8].sample(1, replace=False),
        test_df[test_df['label'] == 9].sample(1, replace=False)
    ])

    for row1 in eight_nine_df.iterrows():
        for row2 in eight_nine_df.iterrows():
            encoding8 = row1[1][numbers].values
            encoding9 = row2[1][numbers].values
            x_test.append(np.concatenate([encoding8, encoding9]))

            if row1[1]['label'] == row2[1]['label']:
                y_test.append(1)
            else:
                y_test.append(0)

            labels.append(str(int(row1[1]['label'])) + str(int(row2[1]['label'])))

    x_test.append(np.concatenate([encoding1_valid, encoding8]))
    y_test.append(0)
    labels.append("X'8")

    x_test.append(np.concatenate([encoding1_valid, encoding9]))
    y_test.append(0)
    labels.append("X'9")

    return np.array(x_test), np.array(y_test), labels


def get_mnist_datasets(cv_pred_df, model_type='FFWD', seed=1):
    assert model_type in ['FFWD', 'LSTM']
    random.seed(seed)

    train_df, test_df = create_datasets(cv_pred_df)
    x_test, y_test, labels = create_test_nums(train_df, test_df)
    x_train, y_train = create_train_nums(train_df)

    if model_type == 'LSTM':
        x_train = x_train.reshape(len(x_train), 2, x_train.shape[1] // 2)
        x_test = x_test.reshape(len(x_test), 2, x_test.shape[1] // 2)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)

    return (x_train, y_train), (x_test, y_test), labels

# This function recompiles model for for each training instance


def create_model(n=0, units=2**5, initializer=keras.initializers.RandomNormal(mean=0),
                 optimizer=keras.optimizers.Adam(), dropout=0.75, model_type='FFWD', activation='relu', input_dim=26):
    assert model_type in ['FFWD', 'LSTM']
    model = Sequential()

    if model_type == 'FFWD':
        model.add(Dense(units=units,
                        activation=activation,
                        kernel_initializer=initializer,
                        bias_initializer=initializer,
                        input_dim=input_dim * 2))
        for j in range(n):
            model.add(Dense(units=units,
                            activation=activation,
                            kernel_initializer=initializer,
                            bias_initializer=initializer))

        model.add(Dense(units=1,
                        activation='sigmoid',
                        kernel_initializer=initializer,
                        bias_initializer=initializer))

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

    else:
        if n > 0:
            for j in range(n):
                model.add(LSTM(units,
                               input_shape=(2, input_dim),
                               return_sequences=True,
                               kernel_initializer=initializer,
                               bias_initializer=initializer))
                model.add(Dropout(dropout))

        model.add(LSTM(units,
                       input_shape=(2, input_dim),
                       kernel_initializer=initializer,
                       bias_initializer=initializer))
        model.add(Dropout(dropout))

        model.add(Dense(1, activation='sigmoid',
                        kernel_initializer=initializer,
                        bias_initializer=initializer))

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# This function runs the experiments and saves both the training histories and test set results

# If ALPHA experiment, use the following arguments
# data_type='ALPHA'
# encoding_f: Function object for creating encodings
# encoding_args: Keyword arguments to pass into encoding_f

# If MNIST experiment, use the following arguments:
# data_type='MNIST'
# cv_pred_df: Pandas dataframe of cv model outputs
def run_experiment(out_folder, epochs_list, units_list, encoding_f=None, encoding_args={}, n_experiments=40, seed=1, model_type='FFWD',
                   data_type='ALPHA', cv_pred_df=None, learning_rate=0.001, logfile='log.pkl', optimizer_type='ADAM', verbose=False):

    assert model_type in ['FFWD', 'LSTM']
    assert data_type in ['ALPHA', 'MNIST']
    assert optimizer_type in ['ADAM', 'SGD']

    # Fit models and create dataframes
    df_list = []

    if data_type == 'ALPHA':
        assert encoding_f is not None
        _, _, (_, test_words) = get_alpha_datasets(encoding_f, encoding_args, model_type=model_type)
        test_labels = test_words.copy()
        test_labels[1] = 'xy'
    else:
        assert cv_pred_df is not None
        _, _, test_nums = get_mnist_datasets(cv_pred_df, model_type=model_type)
        test_labels = test_nums.copy()

    # Start main loop
    for i in range(3):
        random.seed(seed)
        history_list = []
        df = pd.DataFrame(columns=test_labels)
        for j in range(n_experiments):
            if verbose:
                print(f'Training: {i+1} Layers - Experiment {j+1}')

            K.clear_session()

            # Create x_train words
            if data_type == 'ALPHA':
                (x_train, y_train), (x_test, y_test), _ = get_alpha_datasets(encoding_f,
                                                                             encoding_args,
                                                                             model_type=model_type)
            else:
                (x_train, y_train), (x_test, y_test), _ = get_mnist_datasets(cv_pred_df,
                                                                             model_type=model_type)

            if optimizer_type == 'ADAM':
                optimizer = keras.optimizers.Adam(learning_rate)
            else:
                optimizer = keras.optimizers.SGD(learning_rate)
            initializer = keras.initializers.RandomNormal(mean=0)

            input_dim = 26 if data_type == 'ALPHA' else 10
            model = create_model(i, units=units_list[i],
                                 initializer=initializer, optimizer=optimizer,
                                 model_type=model_type, input_dim=input_dim)
            history = model.fit(x_train, y_train,
                                epochs=epochs_list[i], batch_size=len(x_train), verbose=0, shuffle=True,
                                validation_data=(x_test, y_test))
            history_list.append(history.history)
            df = df.append(pd.DataFrame(model.predict(x_test).T, columns=test_labels), ignore_index=True)

        save_dict = {}
        save_dict['df_list'] = df
        save_dict['history_list'] = history_list

        try:
            os.makedirs(out_folder)
        except FileExistsError:
            pass

        with open(os.path.join(out_folder, f'{epochs_list[i]}it_{units_list[i]}units_{i+1}layers.pkl'), 'wb') as f:
            pickle.dump(save_dict, f)
            f.close()

    if data_type == 'ALPHA':
        dump_dict = dict(epochs=epochs_list, units=units_list,
                         encoding=encoding_f.__name__,
                         n_experiments=n_experiments,
                         data_type=data_type,
                         model_type=model_type,
                         optimizer_type=optimizer_type,
                         optimizer=optimizer.get_config())
    else:
        dump_dict = dict(epochs=epochs_list, units=units_list,
                         n_experiments=n_experiments,
                         data_type=data_type,
                         model_type=model_type,
                         optimizer_type=optimizer_type,
                         optimizer=optimizer.get_config())

    print(f'Dumping logfile to {os.path.join(out_folder, logfile)}...')
    with open(os.path.join(out_folder, logfile), 'wb') as f:
        pickle.dump(dump_dict, f)
        f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Experiment Options")
    parser.add_argument('model', choices=['FFWD', 'LSTM'], help='Type of IE model (FFWD or LSTM)')
    parser.add_argument('data', choices=['ALPHA', 'MNIST'], help='Data type of the experiment (ALPHA or MNIST)')
    parser.add_argument('-i', '--iterations', type=int, help='Number of epochs for each run', required=True)
    parser.add_argument('-u', '--units', type=int, help='Number of hidden units for each model', required=True)
    parser.add_argument('-e', '--encoding', choices=['one-hot', 'distributed', 'haar'],
                        help='Type of encoding for ALPHA experiments. Only set if data is ALPHA')
    parser.add_argument('-c', '--cv-output', dest='cv_output',
                        help='Filepath for CV model output file.')
    parser.add_argument('-r', '--runs', default=40,
                        type=int, help='Number of runs per experiment')
    parser.add_argument('-l', '--learning-rate', dest='learning_rate', default=0.001,
                        type=float, help='Learning rate')
    parser.add_argument('-o', '--optimizer', choices=['ADAM', 'SGD'], default='ADAM',
                        help='Optimizer for the learning algorithm')
    parser.add_argument('-f', '--out-folder', dest='out_folder', default='out_folder',
                        help='Name of output folder')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Output verbosity')

    args = parser.parse_args()

    if args.data == 'ALPHA':
        if args.cv_output:
            parser.error('--cv-output can only be set when data=MNIST')
        elif not args.encoding:
            parser.error('--encodings needs to be set when data=ALPHA')

        encodings_dict = {
            'one-hot': get_one_hot_encoding,
            'distributed': get_distr_j_encoding,
            'haar': get_orthogonal_encoding
        }
        run_experiment(out_folder=args.out_folder, epochs_list=[args.iterations] * 3, units_list=[args.units] * 3,
                       encoding_f=encodings_dict[args.encoding], n_experiments=args.runs, model_type=args.model,
                       data_type=args.data, learning_rate=args.learning_rate, optimizer_type=args.optimizer, verbose=args.verbose)

    elif args.data == args.data == 'MNIST':
        if args.encoding:
            parser.error('--encoding can only be set when data=ALPHA')
        elif not args.cv_output:
            parser.error('--cv-output needs to be set when data=MNIST')

        df = pd.read_csv(args.cv_output)
        run_experiment(out_folder=args.out_folder, epochs_list=[args.iterations] * 3, units_list=[args.units] * 3,
                       cv_pred_df=df, n_experiments=args.runs, model_type=args.model,
                       data_type=args.data, learning_rate=args.learning_rate, optimizer_type=args.optimizer, verbose=args.verbose)
