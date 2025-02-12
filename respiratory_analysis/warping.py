import numpy as np
import pickle

from eventwarping.eventseries import EventSeries


################################################
# Train alignment
################################################

def train_alignment(fns, param_dict, selected, excluded_events=None):
    """
    Initialize an eventseries object and align the selected series in files fn
    """

    # initialize eventseries
    es = EventSeries().from_setlistfiles(selected=selected, fns=fns, **param_dict)
    for symbol in [1, 2, 3, 4]:
        es.rescale_weights[symbol] = 5

    # remove other apnea/hypopnea such that only 1 remains per series
    es.series[:, :, 2][np.cumsum(es.series[:, :, 1], 1) == 0] = 0
    es.series[:, :, 4][np.cumsum(es.series[:, :, 3], 1) == 0] = 0
    es.series[:, ::-1, 1][np.cumsum(es.series[:, ::-1, 2], 1) == 0] = 0
    es.series[:, ::-1, 3][np.cumsum(es.series[:, ::-1, 4], 1) == 0] = 0

    # set excluded events to 0 in series
    if excluded_events is not None:
        for event in excluded_events:
            es.series[:, :, event] = 0

    # do alignment
    for i, ws in enumerate(es.warp_yield(iterations=param_dict['n_iter'], restart=True)):
        print(f"=== {i + 1:>2} ===")

    return es

def train_alignment_apnea_and_hypopnea(fns, param_dict, folder):
    # initialize eventseries
    es_all = EventSeries().from_setlistfiles(fns=fns, **param_dict)
    is_apnea = es_all.series[:,:,1].sum(axis=1).astype(bool)
    is_hypopnea = es_all.series[:,:,3].sum(axis=1).astype(bool)

    es_apnea = train_alignment(fns, param_dict, is_apnea, excluded_events=None)
    es_hypopnea = train_alignment(fns, param_dict, is_hypopnea, excluded_events=None)

    with open(folder / "es_apnea", 'wb') as file:
        pickle.dump(es_apnea, file)
    with open(folder / "es_hypopnea", 'wb') as file:
        pickle.dump(es_hypopnea, file)

    return es_apnea, es_hypopnea


def train_alignment_none(test_data_files, labels_test_data, param_dict, folder, excluded_events=None):
    """
    Train warping on 10000 sequences without apnea or hypopnea.
    This is done on the test set since fn contains only sequences that include an apnea or hypopnea
    """
    # initialize eventseries
    sel = labels_test_data == 0
    sel *= np.cumsum(sel) <= 10000
    es_none = EventSeries().from_setlistfiles(selected=labels_test_data==0, fns=test_data_files, **param_dict)

    # set excluded events to 0 in series
    if excluded_events is not None:
        for event in excluded_events:
            es_none.series[:, :, event] = 0

    # do alignment
    for i, ws in enumerate(es_none.warp_yield(iterations=param_dict['n_iter'], restart=True)):
        print(f"=== {i + 1:>2} ===")

    with open(folder / "es_none", 'wb') as file:
        pickle.dump(es_none, file)

    return es_none

################################################
# Warp test data with model
################################################

def warp_with_model(files_test_data, param_dict, model, excluded_events=None):
    es = EventSeries().from_setlistfiles(files_test_data, **param_dict)
    es.series[:, :, :5] = 0

    # set excluded events to 0 in series
    if excluded_events is not None:
        for event in excluded_events:
            es.series[:, :, event] = 0

    es.warp_with_model(model=model, iterations=param_dict['n_iter'])

    return es

def warp_test_data_with_apnea_hypopnea_model(files_test_data, param_dict, apnea_model, hypopnea_model, models_folder, excluded_events=None):
    es_apnea_model = warp_with_model(files_test_data, param_dict, apnea_model, excluded_events)
    es_hypopnea_model = warp_with_model(files_test_data, param_dict, hypopnea_model, excluded_events)

    with open(models_folder / "test_warped_by_apnea", 'wb') as file:
        pickle.dump(es_apnea_model, file)
    with open(models_folder / "test_warped_by_hypopnea", 'wb') as file:
        pickle.dump(es_hypopnea_model, file)

    return es_apnea_model, es_hypopnea_model