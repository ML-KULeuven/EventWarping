import ast
import os
import pickle
import sys
from time import sleep

sys.path += [r"C:\Users\dries\Python projects\EventWarping\src"]
sys.path += [r"C:\Users\dries\Python projects\AItoolkit"]

from pathlib import Path
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import confusion_matrix as CM, precision_score, \
    recall_score
from copy import copy
from sklearn.tree import plot_tree

from eventwarping.formats import setlistfile2setlistsfile, smooth_series
from eventwarping.eventseries import EventSeries
from eventwarping.constraints import NoMergeTooDistantSymbolSetConstraint, MaxMergeSymbolConstraint, NoXorMergeSymbolSetConstraint
from eventwarping.window import LinearScalingWindow
import matplotlib.pyplot as plt
import numpy as np

def rf_clf(x_train, x_test, y_train, y_test):
    clf = RFC(min_samples_leaf=30, max_features=100, n_estimators=100)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cm = CM(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=None)
    rec = recall_score(y_test, y_pred, average=None)
    acc = np.mean(y_pred == y_test)
    print(cm, prec, rec, acc)

    plt.figure()
    plt.imshow(clf.feature_importances_.reshape((25,-1)).T)
    return clf, y_pred

def decision_tree_plot(items_folder, x, y, is_class):
    # decision tree plot
    with (items_folder / 'alphabet.pkl').open('rb') as file:
        alphabet = pickle.load(file)
    alphabet[0] = ""
    # select some instances
    x7 = copy(x).reshape((-1, 25, 50))
    for i in range(len(x7)):
        x7[i, :, 45] = (sum(x7[i, :, 45]) > 5) * 1
    x7 = x7.reshape((-1, 25 * 50))
    sel = np.array([True] * len(x7))
    # learn and plot tree (2 classes)
    dt = DTC(min_samples_leaf=30)
    dt.fit(x7[sel][::2], is_class[sel][::2])
    plt.figure()
    plot_tree(dt,
              feature_names=[(i // 50, alphabet[i % 50]) for i in range(1250)],
              fontsize=5, filled=True, impurity=False)
    # learn and plot tree (3 classes)
    dt = DTC(min_samples_leaf=30)
    dt.fit(x7[sel][::2], y[::2])
    plt.figure()
    plot_tree(dt,
              feature_names=[(i // 50, alphabet[i % 50]) for i in range(1250)],
              fontsize=5, filled=True, impurity=False)

def plot_warping_100_apnea(model_name, x_nowarp, x, y, model_class):
    fig, axs = plt.subplots(10, 10)
    fig.suptitle(f'{model_name} data after warping with {model_name} model')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    j = 0
    for i in range(len(x)):
        if y[i] == model_class:  # hypopnea
            axs[j // 10, j % 10].imshow(x[i].reshape((25, -1))[0:10, 5:26].T,
                                        aspect='auto')
            axs[j // 10, j % 10].set_xticklabels([])
            axs[j // 10, j % 10].set_yticklabels([])
            j += 1
        if j == 100:
            break
    fig, axs = plt.subplots(10, 10)
    fig.suptitle(f'{model_name} data before warping')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    j = 0
    for i in range(len(x_nowarp)):
        if y[i] == model_class:
            axs[j // 10, j % 10].imshow(x_nowarp[i].reshape((25, -1))[0:10, 5:26].T,
                                        aspect='auto')
            axs[j // 10, j % 10].set_xticklabels([])
            axs[j // 10, j % 10].set_yticklabels([])
            j += 1
        if j == 100:
            break


def read_test_data(folder):
    test_data = []
    with (folder / 'test_data.txt').open('r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            test_data += [eval(line)]
    return test_data


def acc_in_function_of_apnea_postion(results_folder, is_wake, is_apnea, is_hypopnea, y_pred, y):
    test_data = read_test_data(results_folder)[~is_wake][1::2]
    apnea_index_per_test_series = [np.argwhere([len(i.intersection({1, 2, 3, 4})) > 0 for i in j]) for j in test_data]
    apnea_indices = np.array([apnea_index_per_test_series[i][0] for i in range(len(test_data)) if is_apnea[~is_wake][1::2][i]])[:,0]
    hypopnea_indices = np.array([apnea_index_per_test_series[i][0] for i in range(len(test_data)) if is_hypopnea[~is_wake][1::2][i]])[:, 0]
    for i in range(10):
        a = (y_pred == y[1::2])[[~is_wake][1::2]][apnea_indices == i]
        print(i, len(a), np.mean(a))
    for i in range(10):
        a = (y_pred == y[1::2])[is_hypopnea[~is_wake][1::2]][hypopnea_indices == i]
        print(i, len(a), np.mean(a))


def likelihood_calculation(x, y, labels, model, laplace_smoothing=1):
    # ll_p0 = np.divide(np.add(np.sign(model._warped_series[labels == 0]).sum(axis=0), laplace_smoothing),
    #                   sum(labels == 1) + 2 * laplace_smoothing).flatten()
    # ll_p1 = np.divide(np.add(np.sign(model._warped_series[labels == 1]).sum(axis=0), laplace_smoothing),
    #                   sum(labels == 1) + 2 * laplace_smoothing).flatten()
    # ll_p2 = np.divide(np.add(np.sign(model._warped_series[labels == 2]).sum(axis=0), laplace_smoothing),
    #                   sum(labels == 1) + 2 * laplace_smoothing).flatten()
    # ll0 = np.sort(x * ll_p0, axis=1)[:, -50:].sum(axis=1)
    # ll1 = np.sort(x * ll_p1, axis=1)[:, -50:].sum(axis=1)
    # ll2 = np.sort(x * ll_p2, axis=1)[:, -50:].sum(axis=1)
    # plt.figure()
    # for i in range(3):
    #     plt.hist(np.sort(x * ll0, axis=1)[:, -30:].sum(axis=1)[y == i], 100, density=True)
    # plt.figure()
    # for i in range(3):
    #     plt.hist(np.sort(x * ll1, axis=1)[:, -30:].sum(axis=1)[y == i], 100, density=True)
    # # distinction between categories visible for apnea and no apnea masks, not for hypopnea mask
    # # likelihood per event but events not independent => use correlation

    # correlation of most prominent items
    plt.figure()
    category = 1
    warped = np.sign(model._warped_series[labels == category])
    freq = np.sum(warped, axis=0).flatten()
    ord_freq = np.argsort(freq)
    warped = warped.reshape(-1, 25 * 50)
    c = np.corrcoef(warped.T)
    plt.imshow(np.sign(model._warped_series[labels == category]).sum(axis=0).T)
    x, y, u, v = [], [], [], []
    n = 0
    for i in ord_freq[::-1]:
        if i % 50 > 44:
            continue
        c[i, i] = 0
        ord_c = np.argsort(c[i])
        n_nan = sum(np.isnan(c[i]))
        for j in ord_c[-n_nan - 5:-n_nan]:
            print(np.round(c[i, j], 2))
            x += [i // 50]
            y += [i % 50]
            u += [j // 50 - x[-1]]
            v += [j % 50 - y[-1]]
        n += 1
        if n == 30:
            break
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)


def bayesian_network_analysis(x, y, labels, model, laplace_smoothing=1):
    import bnlearn as bn
    import pandas as pd
    from time import time
    category = 1
    n_var = 50
    n_time = 15
    warped = np.sign(model._warped_series[:, :n_time, :n_var])
    df = pd.DataFrame(warped.reshape((-1, n_var*n_time)))
    # feat_sel = (np.sum(warped[labels==1], axis=0)>2000).reshape((45*20))
    # df = df.iloc[:,feat_sel]
    df['class'] = labels
    bn_model = bn.structure_learning.fit(df, methodtype='cl', root_node='class')
    bn_model2 = bn.parameter_learning.fit(bn_model, df, n_jobs=1)
    # query = bn.inference.fit(bn_model2, variables=['class'])
    pred = bn.predict(bn_model2, df[1::100], variables=['class'], method=None, verbose=0)
    pred_prob = np.array([[i[j] for j in range(3)] for i in pred['p']])
    pred_class = np.argmax(pred_prob, axis=1)
    acc = np.mean(pred_class == labels[1::100])

    # plot network
    plt.figure()
    s, t, u, v = [], [], [], []
    for i, j in bn_model['model_edges']:
        if i != 'class' and j != 'class':
            r = np.random.random(1)*0.2-0.1
            s += [int(i) // n_var + r]
            t += [int(i) % n_var]
            u += [int(j) // n_var - s[-1] + r]
            v += [int(j) % n_var - t[-1]]
    plt.imshow(np.sign(model._warped_series[labels == category]).sum(axis=0).T)
    plt.quiver(s, t, u, v, angles='xy', scale_units='xy', scale=1)
    plt.show()

def do_evaluations(labels, model, model_name, model_class, items_folder, results_folder):
    is_apnea = labels == 1
    is_hypopnea = labels == 2
    is_random = labels == 0
    is_wake = np.sum(model.series[:,:,48],1) > 5
    labels_sleep = labels[~is_wake]

    if model_class == 1:
        is_class = is_apnea[~is_wake]
        class_name = 'apnea'
    if model_class == 2:
        is_class = is_hypopnea[~is_wake]
        class_name = 'hypopnea'

    # plot symbols of model
    model.plot_symbols(filter_series=is_apnea, title=f'warped by {model_name} model')
    model.plot_symbols(filter_series=is_hypopnea, title=f'warped by {model_name} model')
    model.plot_symbols(filter_series=is_random, title=f'warped by {model_name} model')

    # learn classifiers on warped series
    x = np.sign(model.warped_series[~is_wake])
    # x = np.concatenate((x, model.diff_series[~is_wake]), axis=2)  #############
    x = x.reshape((sum(~is_wake), -1))
    x_nowarp = np.sign(model.series[~is_wake].reshape(-1, 25 * 50))
    y = labels_sleep

    bayesian_network_analysis(x, y, labels, model, laplace_smoothing=1)

    likelihood_calculation(x, y, labels, model, laplace_smoothing=1)

    print('before warping')
    clf, _ = rf_clf(x_nowarp[::2], x_nowarp[1::2], y[0::2], y[1::2])

    print('after warping')
    clf, y_pred = rf_clf(x[::2], x[1::2], y[::2], y[1::2])

    print('after warping train set')
    clf, _ = rf_clf(x[::2], x[::2], y[::2], y[::2])

    print(f'after warping: 2 classes ({model_name} or not)')
    clf, _ = rf_clf(x[::2], x[1::2], is_class[::2], is_class[1::2])

    # learn and plot decision tree
    decision_tree_plot(items_folder, x, y, is_class)

    # plot 100 apnea/hypopnea
    plot_warping_100_apnea(model_name, x_nowarp, x, y, model_class)

    # influence of start position apnea/hypopnea on accuracy
    acc_in_function_of_apnea_postion(results_folder, is_wake, is_apnea, is_hypopnea, y_pred, y)

    plt.show()
    print('evaluation ended')
