import ast
import os
import pickle
import sys
import numpy as np

sys.path += [r"C:\Users\dries\Python projects\EventWarping\src"]
sys.path += [r"C:\Users\dries\Python projects\AItoolkit"]

from respiratory_analysis.make_plots import plot_examples_of_sequences, plot_warping_test_data, plot_density, \
    plot_warping_single_series, \
    plot_densities_toy_example, plot_costs, plot_examples_kernels
from respiratory_analysis.misc import dist_respiratory_itemsets, make_test_set
from respiratory_analysis.warping import train_alignment_apnea_and_hypopnea, warp_test_data_with_apnea_hypopnea_model, \
    train_alignment_none
from eventwarping.constraints import NoMergeTooDistantSymbolSetConstraint, MaxMergeSymbolConstraint, \
    NoXorMergeSymbolSetConstraint
from eventwarping.formats import setlistfile2setlistsfile
from eventwarping.window import LinearScalingWindow

from pathlib import Path


if __name__ == '__main__':
    # define data and parameters
    symbol_ordenings = [[5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                        [25, 26, 27, 28, 29, 30, 31, 32, 33, 34], [35, 36, 37, 38, 39, 40, 41, 42, 43, 44]]

    items_folder = Path(r"C:\Users\dries\python projects\itemsets_final")  # training and files test set
    items_smooth_folder = Path(r"C:\Users\dries\python projects\smoothed_itemsets_final")
    models_folder = Path(r"C:\Users\dries\python projects\warping_models")
    names = [i for i in os.listdir(items_folder) if (i.endswith('itemsets.txt')) and not (i.endswith('parsed_itemsets.txt'))]
    names_train = names[::2]
    names_test = names[1::2]

    max_dist = 10
    constraints = [
        NoMergeTooDistantSymbolSetConstraint(dist_respiratory_itemsets, max_dist),
        MaxMergeSymbolConstraint(5),
        NoXorMergeSymbolSetConstraint([1, 2, 3, 4])
    ]
    param_dict = {"window": LinearScalingWindow(5), "intonly": True, "constraints": constraints, "max_series_length": 25, "n_iter":20}
    n_iter = param_dict['n_iter']

    ##############################################

    # # correct itemsets for unexpected short drops
    # for k, name in enumerate(names):
    #     fn = items_folder / name
    #     print(f'correct short drops {k}/{len(names)}')
    #     with fn.open("r") as fp:
    #         data = fp.read()
    #     data = ast.literal_eval(data)
    #     for lows, highs in [[{5, 6, 7}, {11, 12, 13, 14}], [{15, 16, 17}, {21, 22, 23, 24}]]:
    #         high_peak = [len(i.intersection(highs)) > 0 for i in data]
    #         low_peak = [len(i.intersection(lows)) > 0 for i in data]
    #         short_drop = (np.convolve(np.array(high_peak)-0.5, [1,-1, 1], 'same') == 1.5) & low_peak
    #         short_drop_ind = np.where(short_drop)[0]
    #         for i in short_drop_ind:
    #             data[i] = data[i].difference(lows).union(data[i-1].intersection(highs))
    #     with fn.open("w") as fp:
    #         fp.write(repr(data))

    #############################################

    # # save smoothed itemsets
    # for k, name in enumerate(names):
    #     print(f'Smoothing {k}/{len(names)}')
    #     setlistfile2setlistsfile(items_folder / name, items_smooth_folder / name, start={1,3}, stop={2,4}, margin=5, symbol_ordenings=symbol_ordenings)

    ##############################################

    # # train event alignment
    # train_alignment_apnea_and_hypopnea([items_smooth_folder / i for i in names_train], param_dict, models_folder)

    ##############################################

    # # warp test data with model
    # with open(models_folder / "es_apnea", 'rb') as file:
    #     es_apnea = pickle.load(file)
    # with open(models_folder / "es_hypopnea", 'rb') as file:
    #     es_hypopnea = pickle.load(file)

    # make_test_set([items_folder / i for i in names_test], items_smooth_folder, symbol_ordenings=symbol_ordenings)
    # files_test_data = [items_smooth_folder / 'test_data.txt']
    # warp_test_data_with_apnea_hypopnea_model(files_test_data, param_dict, es_apnea, es_hypopnea, models_folder)

    ##############################################

    # # train warping on sequences without apnea or hypopnea
    # files_test_data = [items_smooth_folder / 'test_data.txt']
    # with (items_smooth_folder / 'test_data_labels.txt').open('r') as file:
    #     labels_test = np.array(eval(file.read()))
    # train_alignment_none(files_test_data, labels_test, param_dict, models_folder)

    #############################################

    # Make plots paper

    with open(models_folder / 'es_apnea.pkl', 'rb') as file:
        es_apnea = pickle.load(file)
    with open(models_folder / 'es_hypopnea.pkl', 'rb') as file:
        es_hypopnea = pickle.load(file)
    with open(models_folder / 'es_none.pkl', 'rb') as file:
        es_none = pickle.load(file)
    with open(models_folder / "test_warped_by_apnea.pkl", 'rb') as file:
        es_test_apnea = pickle.load(file)
    with open(models_folder / "test_warped_by_hypopnea.pkl", 'rb') as file:
        es_test_hypopnea = pickle.load(file)
    with (items_smooth_folder / 'test_data_labels.txt').open('r') as file:
        labels_test = np.array(eval(file.read()))

    plot_examples_of_sequences(es_apnea, es_hypopnea, 'series_examples', 12, 31)
    plot_warping_test_data(es_test_apnea, labels_test, 1, name='apnea_warped_by_apnea')
    plot_warping_test_data(es_test_hypopnea, labels_test, 2, name='hypopnea_warped_by_hypopnea')
    plot_density([es_apnea, es_hypopnea,es_none], 'all_densities')
    plot_warping_single_series(es_test_apnea, 'single_warping_apnea', 53)
    plot_warping_single_series(es_test_hypopnea, 'single_warping_hypopnea', 53)
    plot_densities_toy_example('densities_toy_example')
    plot_costs([es_apnea, es_hypopnea])
    plot_examples_kernels()




    ##############################################

    # # evaluation on test data set
    # with open(models_folder / file_name_apnea_model, 'rb') as file:
    #     es_apnea_model = pickle.load(file)
    # with open(models_folder / file_name_hypopnea_model, 'rb') as file:
    #     es_hypopnea_model = pickle.load(file)
    # # diff_series = series_to_diff_series(es_apnea_model.warped_series, symbol_ordenings)
    # # es_apnea_model.diff_series = diff_series
    #
    # with (items_smooth_folder / 'test_data_labels.txt').open('r') as file:
    #     labels_test = np.array(eval(file.read()))
    # print("----------apnea model-------------")
    # do_evaluations(labels_test, es_apnea_model, 'apnea', 1, items_folder, items_smooth_folder)
    # plt.show()

    ##############################################

    # # evaluation on individual file
    # ## train clf on test set data
    # with open(models_folder / file_name_apnea_model, 'rb') as file:
    #     model = pickle.load(file)
    # x = np.sign(model.warped_series)
    # x = x.reshape((len(x), -1))
    # with (items_smooth_folder / 'test_data_labels.txt').open('r') as file:
    #     y = np.array(eval(file.read()))
    # clf = RFC(min_samples_leaf=30, max_features=100, n_estimators=100)
    # clf.fit(x, y)
    #
    # # # evaluate on file
    # # folder_test_files = Path(r"C:\Users\dries\python projects\itemsets_multilabeled")
    # # files = [folder_test_files / i for i in os.listdir(folder_test_files) if (i.endswith('itemsets.txt'))]
    # # for i in files:
    # #     brt_file = [j for j in uza_multilabeled_single_files if str(i.parts[-1]).replace('_itemsets.txt','') in j][0]
    # #     pred = predict_apnea_per_file(i, clf, model, uza_multilabeled_single_files[brt_file])
    #
    # path = Path(r"C:\Users\dries\python projects\itemsets_new\DZ_00001_0000414_itemsets.txt")
    # pred = predict_apnea_per_file(path, clf, model)

    print(1)
