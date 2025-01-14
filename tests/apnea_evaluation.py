import ast
import os
import pickle
import sys
import pandas as pd
from time import sleep

import scipy.linalg

from src.hmm import NewCategoricalHMM
from tests.rsrc.cnn_classification import cnn_evaluation

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

def rf_clf(x_train, x_test, y_train, y_test, min_samples_leaf=10, max_depth=5):
    clf = RFC(min_samples_leaf=min_samples_leaf, n_estimators=100, max_depth=max_depth)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cm = CM(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=None)
    rec = recall_score(y_test, y_pred, average=None)
    acc = np.mean(y_pred == y_test)
    print(cm, prec, rec, acc)

    # plt.figure()
    # plt.imshow(clf.feature_importances_.reshape((25,-1)).T)
    return clf, y_pred, acc

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
        if y[i] == model_class:
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

def train_hmm(train_data, n_comp=40, own_init=False, n_iter=100):
    from scipy.linalg import toeplitz
    import random

    # set parameters
    n_inst, n_time, n_feat = train_data.shape
    random.seed(123669)

    # define initial parameters
    trans = (toeplitz(np.arange(n_comp+1, 1, -1), np.arange(n_comp+1, 1, -1)))**3
    trans = (trans.T / trans.sum(axis=1)).T
    emissions = np.mean(train_data, axis=0)
    if n_comp < len(emissions):
        emissions = emissions[:n_comp]
    else:
        d = n_comp - len(emissions)
        emissions = np.vstack((emissions, 0.1 + 0.9*train_data.reshape(-1, n_feat)[np.random.randint(0, len(train_data), d)]))
    starts = [1 / n_comp] * n_comp

    # plt.figure()
    # plt.imshow(emissions.T)
    # plt.figure()
    # plt.imshow(trans.T)

    # initiate and fit model
    if own_init:
        model = NewCategoricalHMM(n_comp, n_iter=n_iter, init_params='')
        model.transmat_ = trans
        model.startprob_ = starts
        model.emissionprob_ = emissions
    else:
        model = NewCategoricalHMM(n_comp, n_iter=n_iter, init_params='est')
    data = train_data.reshape((-1, n_feat))
    model.fit(data, lengths=[n_time]*(len(data)//n_time))

    # plot components
    plt.figure()
    plt.imshow(model.emissionprob_.T)
    plt.figure()
    plt.imshow(model.transmat_.T)

    return model

def plot_hmm(model, x, y, sort_ind=0):
    n_comp = model.n_components
    n_inst, n_time, n_feat = x.shape

    # order components
    order = np.argsort(model.emissionprob_[:, sort_ind])
    model.emissionprob_ = model.emissionprob_[order]
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[order]
    model.transmat_ = model.transmat_.T[order].T

    fig, axs = plt.subplots(3)
    for j in range(3):
        data = np.sign(x)[y == j].reshape((-1, n_feat))
        prob = model.predict_proba(data)
        for i in range(n_comp):
            pattern = np.mean(prob[:, i].reshape((-1, n_time)), axis=0)
            axs[j].plot(pattern)
            if i == 1:
                apnea_pattern = pattern

    plt.figure()
    plt.imshow(model.emissionprob_.T)
    plt.figure()
    plt.imshow(model.transmat_.T)

    plt.show()

def do_evaluations(labels, model, model_name, model_class, items_folder, results_folder):
    is_apnea = labels == 1
    is_hypopnea = labels == 2
    is_other = labels == 0
    is_wake = np.sum(model.series[:,:,48],1) > 5
    labels_sleep = labels[~is_wake]

    if model_class == 1:
        is_class = is_apnea[~is_wake]
        class_name = 'apnea'
    if model_class == 2:
        is_class = is_hypopnea[~is_wake]
        class_name = 'hypopnea'

    #####################################################

    def remove_smoothing(x, labels):
        """If multiple values of one signal are present, only keep the average."""
        # fig, axs = plt.subplots(1,3)
        # for i in range(3):
        #     axs[i].imshow(x[labels == i].sum(axis=0).T)

        for i in range(4):
            z = x[:, :, (1+i*10):(11+i*10)]
            n_inst, n_time, n_feat = z.shape
            sums = np.sum(z, axis=2).flatten()
            mean_ind = np.floor(np.array(pd.DataFrame(np.argwhere(z)).groupby([0, 1]).mean()).flatten()).astype(int)
            without_smooth = np.zeros(z.shape)
            without_smooth[np.repeat(np.arange(n_inst), n_time)[sums != 0],
                            np.tile(np.arange(n_time), n_inst)[sums != 0],
                            mean_ind] = 1
            x[:, :, (1+i*10):(11+i*10)] = without_smooth

        # fig, axs = plt.subplots(1,3)
        # for i in range(3):
        #     axs[i].imshow(x[labels == i].sum(axis=0).T)
        return x

    ################## HMM model ########################
    # def calculate_bics(data_label, label):
    #     n_inst, n_time, n_feat = data_label.shape
    #     length = [n_time]*n_inst
    #     bics = [np.inf, np.inf, np.inf]
    #     i = 10
    #     while True:
    #         directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    #         filename = str(directory / f'bic_{i}_components_class_{label}.pkl')
    #
    #         try:
    #             with open(filename, 'rb') as handle:
    #                 hmm = pickle.load(handle)
    #             bic = hmm.bic(data_label.reshape((-1, hmm.n_features)), length)
    #             bics += [bic]
    #         except:
    #             hmm = train_hmm(data_label, i, False)
    #             bic = hmm.bic(data_label.reshape((-1, hmm.n_features)), length)
    #             bics += [bic]
    #             with open(filename, 'wb') as handle:
    #                 pickle.dump(hmm, handle)
    #
    #         print(i, bic)
    #         if (bic > bics[-2]) & (bic > bics[-3]) & (bic > bics[-4]):
    #             break
    #         else:
    #             i += 10
    #     print(1)
    #
    # def compare_bics(data_label, label):
    #     n_inst, n_time, n_feat = data_label.shape
    #     length = [n_time]*n_inst
    #     comps = []
    #     bics = []
    #     files = [j for j in [i for i in os.walk(".")][0][2] if ('bic' in j) and (f'class_{label}' in j)]
    #     for file in files:
    #         with open('.\\' + file, 'rb') as handle:
    #             model = pickle.load(handle)
    #         comps += [int(file.split('_')[1])]
    #         bics += [model.bic(data_label.reshape((-1, model.n_features)), length)]
    #     plt.figure()
    #     plt.plot(comps, bics)
    #     plt.show()
    #
    # data = np.sign(np.concatenate([model.warped_series[:, :15, 5:25],model.warped_series[:, :15, 45:]], axis=2))
    # data = remove_smoothing(data[~is_wake], labels_sleep)
    #
    # # for label in [0,1,2]:
    # #     data_label = data[labels == label]
    # #     calculate_bics(data_label, label)
    # #
    # # for label in [0,1,2]:
    # #     data_label = data[labels == label]
    # #     compare_bics(data_label, label)
    #
    # n_iter = 100
    # hmm0 = train_hmm(data[labels_sleep == 0], n_iter=n_iter)
    # hmm1 = train_hmm(data[labels_sleep == 1], n_iter=n_iter)
    # hmm2 = train_hmm(data[labels_sleep == 2], n_iter=n_iter)
    #
    # ll = []
    # ll += [[hmm0.score(k) for k in data]]
    # ll += [[hmm1.score(k) for k in data]]
    # ll += [[hmm2.score(k) for k in data]]
    #
    # plot_hmm(hmm1, data, labels_sleep)
    # hmm_pred = np.argmax(ll, axis=0)
    # acc = np.mean(labels_sleep == hmm_pred)
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(labels_sleep, hmm_pred)
    # #
    # # for j in range(3):
    # #     plt.figure()
    # #     for i in ll:
    # #         plt.plot((np.array(i) - ll[j])[labels == j])

    ####################################################
    x = np.sign(model.warped_series)
    mask = np.sign(model.warped_series[is_apnea]).mean(axis=0)
    mask = np.clip(mask, 0.001, np.inf)
    a = x * mask
    l = np.ones((len(x), x.shape[1]))
    for i in range(4):
        l *= np.max(a[:, :, (5 + 10 * i):(15 + 10 * i)], axis=2)
    ll = np.prod(l[:, 3:10], axis=1)
    ll *= np.quantile(x[:,:,45], 0.8, axis=1)  # dip
    ll = np.clip(ll, 10**(-20), np.inf)
    for i in range(3):
        plt.hist(np.log(ll)[labels == i], 100, alpha=0.5, density=True, cumulative=True)
    plt.show()
    ####################################################

    # # plot symbols of model
    # model.plot_symbols(filter_series=is_apnea)
    # model.plot_symbols(filter_series=is_hypopnea)
    # model.plot_symbols(filter_series=is_other)

    # learn classifiers on warped series
    x = np.sign(model.warped_series)
    x_nowarp = np.sign(model.series)
    # x = np.concatenate((x, model.diff_series), axis=2)  #############
    # x = remove_smoothing(x, None)  # remove the intermediate smoothing
    # x_nowarp = remove_smoothing(x_nowarp, None)  # remove the intermediate smoothing
    x = x[~is_wake].reshape((sum(~is_wake), -1))
    x_nowarp = x_nowarp[~is_wake].reshape((sum(~is_wake), -1))
    y = labels_sleep


    # bayesian_network_analysis(x, y, labels, model, laplace_smoothing=1)

    likelihood_calculation(x, y, labels, model, laplace_smoothing=1)

    cnn_evaluation(x_nowarp, x, y)

    print('before warping')
    clf, _ = rf_clf(x_nowarp[::2], x_nowarp[1::2], y[0::2], y[1::2])

    print('after warping')
    clf, y_pred = rf_clf(x[::2], x[1::2], y[::2], y[1::2])

    print('before warping, normal or not')
    clf, _ = rf_clf(x_nowarp[::2], x_nowarp[1::2], y[0::2]>0, y[1::2]>0)

    print('after warping, normal or not')
    clf, y_pred = rf_clf(x[::2], x[1::2], y[::2]>0, y[1::2]>0)

    print('after warping train set')
    clf, _ = rf_clf(x[::2], x[::2], y[::2], y[::2])

    print(f'after warping: 2 classes ({model_name} or not)')
    clf, _ = rf_clf(x[::2], x[1::2], is_class[::2], is_class[1::2])

    # # learn and plot decision tree
    # decision_tree_plot(items_folder, x, y, is_class)
    #
    # # plot 100 apnea/hypopnea
    # plot_warping_100_apnea(model_name, x_nowarp, x, y, model_class)
    #
    # # influence of start position apnea/hypopnea on accuracy
    # # acc_in_function_of_apnea_postion(results_folder, is_wake, is_apnea, is_hypopnea, y_pred, y)

    plt.show()
    print('evaluation ended')
