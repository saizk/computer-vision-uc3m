# -*- coding: utf-8 -*-
"""
LAB TEMPLATE

Audio features

"""

# Generic imports
import time
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
# Import ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# Import audio feature library
import librosa as lbs
# Import our custom lib for audio feature extraction (makes use of librosa)
import audio_features as af

from sklearn.metrics import confusion_matrix
from transformer import ExtractFeatures, CombineFeatures

# from trans2 import ExtractFeatures, CombineFeatures

##############################################################################
# Data read (and prepare)
##############################################################################
def load_data(cat_folder, dog_folder):
    # Get file names
    cat_files = [f'{cat_folder}/{f}' for f in listdir(cat_folder)]
    dog_files = [f'{dog_folder}/{f}' for f in listdir(dog_folder)]

    # Unify data and code labels
    X = deepcopy(cat_files)
    X.extend(dog_files)
    # Label 0 --> CAT
    # Label 1 --> DOG (Arbitrary positive class)
    y = list(np.concatenate((np.zeros(len(cat_files)),
                             np.ones(len(dog_files))), axis=0).astype(int))
    return load_audio(X), y


def load_audio(X):
    num_data = len(X)
    audios = [lbs.load(X[i], sr=lbs.get_samplerate(X[i]))[0]
              for i in range(num_data)]
    return audios


def plot_roc(y_test, y_scores, auc_svm):
    # Obtain ROC curve values (FPR, TPR)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_scores)
    # Plot ROC curve (displaying AUC)
    plt.subplots(1, figsize=(10, 10))
    plt.title(f'Receiver Operating Characteristic Curve - AUC = {auc_svm:.3f}')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


##############################################################################
# Feature extraction function
##############################################################################
def compute_features(audio, func_list, features_list):
    results = []
    for idx, (audio_func, kwargs) in enumerate(func_list):
        kwargs = parse_kwargs(kwargs, audio_sr=audio[1])
        features = audio_func(audio[0], **kwargs)

        for f_idx, feat_func in enumerate(features_list):
            results.append(feat_func(features))

    return results


def parse_kwargs(kwargs, audio_sr):
    if kwargs.get('sr'):
        kwargs['sr'] = audio_sr
    return kwargs


def extract_features(X, func_list: list, features_list: list, verbose: bool = True):
    """
    >> Function to be completed by the student
    Extracts a feature matrix for the input data X
    ARGUMENTS:
        :param X: Audio data file paths to analyze
        :param verbose:
    """

    num_data = len(X)  # Number of samples to process
    n_feat = len(features_list)  # Specify the number of features to extract

    M = np.zeros((num_data,
                  n_feat * len(func_list)))  # Generate empty feature matrix

    for i in range(num_data):
        if verbose:
            print(f'{i + 1}/{num_data}... ', end='')

        audio_data = X[i]

        for idx, (audio_func, kwargs) in enumerate(func_list):
            kwargs = parse_kwargs(kwargs, audio_sr=16000)
            features = audio_func(audio_data, **kwargs)

            for f_idx, feat_func in enumerate(features_list):
                M[i, idx * n_feat + f_idx] = feat_func(features)

        if verbose:
            print('Done')
    return M


##############################################################################
# Training Process
##############################################################################
def train(M_train, y_train, clf):
    return clf


##############################################################################
# Evaluation Process
##############################################################################
def evaluate(M_test_n, clf):
    # Obtain predicted labels and scores (probabilities) according to model
    y_pred = clf.predict(M_test_n)
    y_scores = clf.predict_proba(M_test_n)[:, 1]
    return y_pred, y_scores


def combination_feature_functions(X_train, y_train, features_list,
                                  scoring='roc_auc', cv=5):
    comb_grid = {
        # 'cf__preprocess_audio':       [True, False],
        'cf__get_energy': [True, False],
        'cf__get_energy_entropy': [True, False],
        'cf__get_spectral_entropy': [True, False],
        'cf__get_spectral_flux': [True, False],
        'cf__get_spectral_centroid': [True, False],
        'cf__get_spectral_spread': [True, False],
        'cf__get_spectral_contrast': [True, False],
        'cf__get_spectral_flatness': [True, False],
        'cf__get_spectral_rolloff': [True, False],
        'cf__get_zero_crossing_rate': [True, False],
        'cf__get_harmonic_ratio': [True, False],
        'cf__get_mfccs': [True, False],
        'cf__get_rms': [True, False],
        'cf__get_poly_features': [True, False]
    }
    pipeline = Pipeline([
        ("cf", CombineFeatures(features_list, verbose=True)),
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True))]
    )
    grid = GridSearchCV(
        pipeline, param_grid=comb_grid,
        scoring=scoring, cv=cv,
        n_jobs=-1, verbose=True
    ).fit(X_train, y_train)

    return grid


def parameter_tuning_feature_functions(
        X_train, y_train,
        features_list, func_config: dict,
        scoring='roc_auc', cv=5
):
    param_grid = {
        'af__thr': [10, 20, 30],
        'af__flen': [512, 1024, 2048],
        'af__hop': [512, 1024, 2048],
        'af__nsub': [8, 10, 12],
        'af__porder': [1, 2, 3],
        # 'af__perc':   [0.85, 0.9, 0.95],
        # 'af__n_mfcc': [10, 20, 30, 40]
    }
    pipeline = Pipeline([
        ("af", ExtractFeatures(
            features_list, **func_config,
            verbose=True
        )),
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True))]
    )
    grid = GridSearchCV(
        pipeline, param_grid=param_grid,
        scoring=scoring, cv=cv,
        n_jobs=-1, verbose=True
    ).fit(X_train, y_train)

    return grid


def print_grid(grid):
    print(grid.best_estimator_)
    print(grid.best_params_)
    print(grid.best_score_)
    if input('grid? ') == 'y':
        print(grid)
        print(grid.cv_results_)


def main():
    root_dir = 'data'
    X, y = load_data(f'{root_dir}/cat', f'{root_dir}/dog')
    test_size = 0.3  # Fix size of test set
    ran_seed = 999  # Manual seed (for replicability)
    X_train, X_test, y_train, y_test = train_test_split(X, y,  # Perform train-test division
                                                        test_size=test_size,
                                                        random_state=ran_seed)

    features_list = [np.nanmax, np.nanmin, np.nanmean, np.nansum]

    # CV BEST COMBINATION OF FILTERS
    st = time.time()
    comb_grid = combination_feature_functions(
        X_train, y_train, features_list,
        scoring='roc_auc'
    )
    print_grid(comb_grid)
    print(f'Elapsed time: {time.time() - st}')

    # CV BEST HYPERPARAMETERS FOR FILTERS
    st = time.time()
    func_config = {  # Best combination of filters
        'preprocess_audio': True,
        'get_energy': True,
        'get_energy_entropy': True,
        'get_zero_crossing_rate': True,
        'get_poly_features': True
    }
    param_grid = parameter_tuning_feature_functions(
        X_train, y_train,
        features_list, func_config, scoring='roc_auc'
    )
    print_grid(param_grid)
    print(f'Elapsed time: {time.time() - st}')

    # FINAL PIPELINE
    pipeline = Pipeline([
        ("af", ExtractFeatures(
            features_list, **func_config,
            thr=10, flen=2048, hop=512, nsub=12, porder=2,
            verbose=True
        )),
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True))]
    )
    # X_train = extract_features(X_train, func_list, features_list)
    # X_test = extract_features(X_test, func_list, features_list)
    pipeline.fit(X_train, y_train)
    y_pred, y_scores = evaluate(X_test, pipeline)

    auc_svm = roc_auc_score(y_test, y_scores)  # Get Area Under the Curve
    plot_roc(y_test, y_scores, auc_svm)
    confusion_matrix(y_scores, y_pred)


if __name__ == '__main__':
    main()
