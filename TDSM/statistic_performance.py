import os
import csv
import pickle
import argparse
from copy import deepcopy
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, \
    precision_score, recall_score, roc_curve, auc, matthews_corrcoef, \
    make_scorer
import pandas as pd

from datasets import get_dataset, DATASETS_CONFIG

np.seterr(divide='ignore', invalid='ignore')


def load_map(path):
    """ Load score map or label map.
    Args:
    - path (str): path to .pkl file containing map
    Returns:
    - data (np.array): loaded map
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def specificity(ground_truth, predictions):
    tp, tn, fn, fp = 0.0,0.0,0.0,0.0
    for l,m in enumerate(ground_truth):
        if m==predictions[l] and m==1:
            tp+=1
        if m==predictions[l] and m==0:
            tn+=1
        if m!=predictions[l] and m==1:
            fn+=1
        if m!=predictions[l] and m==0:
            fp+=1
    return tn/(tn+fp)


def load_score_maps(acq_name):
    """ Load score map.
    Args:
    - acq_name (list[str]): list containing acquisition name
    Returns:
    - score_map_list (list[np.array]): list containing score maps
    """
    sc_mp_tmpl = CONF_LOADING['score_map_template']
    sc_mp_pth = CONF_LOADING['score_map_path']

    score_map_list = []
    for acq in acq_name:
        full_path = os.path.join(sc_mp_pth, sc_mp_tmpl.format(acq))
        score_map_list.append(load_map(full_path))

    return score_map_list


def thresh_score_generator(gt_map_list, score_map_list, criterion, step=0.01):
    """ Compute optimal decision threshold based on the specified
    criterion.
    Args:
    - gt_map_list (list[np.array]): list containing ground-truth maps
    - score_map_list (list[np.array]): list containing score maps
    - criterion (func): criterion to be used to compute scores
    - step (float): float in [0, 1] to define the spacing between threshold
    values
    Returns:
    - thresholds (np.array): array containing threshold values
    - scores (np.array): array containing score values for each
    acquisition
    """
    # Generate threshold list
    thresholds = np.arange(0, 1+step, step)

    # Generate metric scores for each acquisition
    scores = np.nan*np.ones((len(gt_map_list), len(thresholds)))
    for x, (gt_map, score_map) in enumerate(zip(gt_map_list, score_map_list)):
        if len(list(np.unique(gt_map))) != 1:
            bin_gt_map = binary_index(gt_map, np.unique(gt_map)[-1])
            for y, thresh in enumerate(thresholds):
                bin_score_map = binary_thresh(score_map, thresh)
                scores[x, y] = criterion(bin_gt_map, bin_score_map)

    return thresholds, scores


def csv_write(csv_path, data, index=None, columns=None, skip_line=False):
    """
    Generate specified data into a csv file
    Args:
    - csv_path (str): path to csv file
    - data (np.array): data to be writen
    - index (list[]): index (line name) of the DataFrame
    - columns (list[]): columns name of the DataFrame
    - skip_line (bool): skip a line into the csv file before writing
    """
    with open(csv_path, 'a+') as csvfile:
        writer = csv.writer(csvfile)
        if skip_line:
            writer.writerow([''])
        if index is None and columns is None:
            writer.writerow(data)
    if index or columns:
        df = pd.DataFrame(data, columns=columns, index=index)
        print('\n', df)
        df.to_csv(csv_path, mode='a+')


def plot_curves(thresholds, scores, criterion, acq_name, class_name='',
                y_lim=[0, 1]):
    """ Display criterion scores relatively to the decision threshold
    Args:
    - thresholds (list[float]): list containing threshold values
    - scores (list[float]): list containing score values
    - criterion (str): criterion used
    - class_name (str): class name
    - acq_name (list[str]): acquisition names
    - y_lim (list[int]): y axis limits
    """
    # Compute optimal decision threshold
    avg_metric_scores = np.nanmean(scores, axis=0)
    optimal_threshold = thresholds[np.argmax(avg_metric_scores)]

    # Plot score curves
    plt.figure(figsize=(20, 12), dpi=94)
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim(y_lim)
    plt.xlabel('Decision threshold (a.u.)')
    plt.ylabel('{} score (a.u.)'.format(criterion))
    for x, name in enumerate(acq_name):
        plt.plot(thresholds, scores[x, :], label=name)
    plt.plot([optimal_threshold]*2, [0, 1], 'r:', label='optimal threshold')
    plt.plot(thresholds, avg_metric_scores, 'b--', label='average')
    plt.title('{} score, {}'.format(criterion, class_name))
    plt.legend(loc='lower right')
    bbox_props = dict(boxstyle="round", fc="white", ec="k",  lw=1)
    axes.text(optimal_threshold+0.02, 0.05,
              'threshold: {:.2f}\nmean {} score: {:.2f}'.format(
                  optimal_threshold, criterion, avg_metric_scores.max()),
              bbox=bbox_props)


def binary_thresh(score_map, thresh):
    """ Convert a score map to binary map respecting a decision
    threshold.
    Args:
    - score_map (np.array): score map array
    - thresh (float): threshold to apply on decision score
    Returns:
    - bin_map (np.array): binary map
    """
    return (score_map >= thresh).astype('int')


def binary_index(gt_map, class_index):
    """ Convert a gt map to binary map respecting a class index
    Args:
    - gt_map (np.array): gt map array
    - class_index (int): class index to be retained
    Returns:
    - bin_map (np.array): binary map
    """
    return (gt_map == class_index).astype('int')


def custom_auc(gt_map, pred_map, pos_label=None):
    """ Generate AUC value from prediction and ground truth map
    Args:
    - gt_map (np.array): ground truth map
    - pred_map (nparray): prediction map
    Returns:
    - auc_score (float): auc score
    """
    fpr, tpr, thresholds = roc_curve(gt_map, pred_map, pos_label=pos_label)
    auc_score = auc(fpr, tpr)
    return auc_score


def main():
    """ Main.
    """
    # Load score map and ground truth map
    _, gt_map_global, label_values, ignored_labels, new_conf = get_dataset(
        DATASET, CONF_DATASET['loading'])
    DATASETS_CONFIG.update(new_conf)

    os.makedirs(CONF_LOADING['out_path'], exist_ok=True)

    data_info = DATASETS_CONFIG[DATASET]['info_data']

    acq_name = data_info['img_name']
    nb_img = len(acq_name)
    nb_class = len(label_values)

    gt_map_list = list(np.split(gt_map_global, nb_img, 1))

    score_map_list = load_score_maps(acq_name)
    score_map_global = np.concatenate(tuple(score_map_list), axis=1)

    avg_mode = EVAL_CONF['avg_mode']

    # Remove ignored labels pixels
    for ind in ignored_labels:
        gt_map_global[gt_map_global == ind] = 0
    mask = gt_map_global != 0
    gt_map_global = gt_map_global[mask]
    score_map_global = score_map_global[mask]
    for acq_ind, (gt_map, score_map) in enumerate(
            zip(gt_map_list, score_map_list)):
        mask = gt_map != 0
        gt_map_list[acq_ind] = gt_map_list[acq_ind][mask]
        score_map_list[acq_ind] = score_map_list[acq_ind][mask]
    retained_class = np.unique(gt_map_global)

    optimal_threshold = {}

    # Generate optimal threshold for each class
    if EVAL_CONF['decision_threshold_optimization']:
        for crit, func in CRITERIONS.items():
            optimal_threshold[crit] = {}
            # Overwrite existing csv file
            csv_path = os.path.join(CONF_LOADING['out_path'],
                                    '{}_stat_report.csv'.format(crit))
            with open(csv_path, 'w'):
                pass
            csv_write(csv_path, [crit])

        # Compute thresholds/scores for each class
        for class_ind in range(nb_class):
            print('\n', '='*100)
            print('class:', class_ind, label_values[class_ind])

            if class_ind in retained_class:
                class_name = label_values[class_ind]

                if avg_mode == 'micro':
                    process_sc_mp = [score_map_global[:, class_ind]]
                    copy_gt_map = deepcopy(gt_map_global)
                    copy_gt_map[copy_gt_map != class_ind] = 0
                    process_gt_mp = [copy_gt_map]

                elif avg_mode == 'macro':
                    process_sc_mp = [x[:, class_ind] for x in score_map_list]
                    process_gt_mp = deepcopy(gt_map_list)
                    for ind_mp, gt_mp in enumerate(gt_map_list):
                        process_gt_mp[ind_mp][gt_mp != class_ind] = 0

                for crit, func in CRITERIONS.items():
                    optimal_threshold[crit][class_ind] = {}
                    csv_path = os.path.join(CONF_LOADING['out_path'],
                                            '{}_stat_report.csv'.format(crit))

                    thresholds, scores = thresh_score_generator(
                        process_gt_mp, process_sc_mp, func)

                    # Plot criterion scores
                    plot_curves(thresholds, scores, crit, acq_name,
                                y_lim=[0, 1], class_name=class_name)
                    plt.savefig(
                        os.path.join(CONF_LOADING['out_path'],
                                     '{}_{}.png'.format(crit, class_name)))

                    for ind_mp, acq_nm in enumerate(acq_name):
                        optimal_threshold[crit][class_ind][acq_nm] = \
                            thresholds[np.argmax(scores[ind_mp, :])]

                    # Compute optimal decision threshold
                    avg_metric_scores = np.nanmean(scores, axis=0)
                    optimal_threshold[crit][class_ind]['Global'] = thresholds[
                        np.argmax(avg_metric_scores)]

                    # Generate csv file
                    csv_write(csv_path, [class_name], skip_line=True)
                    csv_write(csv_path, scores, index=acq_name,
                              columns=thresholds)
                    disp_thresh = np.array(list(
                        optimal_threshold[crit][class_ind].values()))
                    csv_write(csv_path, disp_thresh,
                              columns=['optimal thresholds'],
                              index=acq_name+['GLOBAL'], skip_line=True)

    # Generate evaluation
    column_name = list(MODALITIES.keys())

    # Compute metrics with threshold applied
    for crit, optm_thrsh in optimal_threshold.items():
        csv_path = os.path.join(CONF_LOADING['out_path'],
                                '{}_stat_report.csv'.format(crit))
        csv_write(csv_path, ['-'*20]*10, skip_line=True)
        for class_ind in range(nb_class):
            # Only compute metrics on not ignored classes
            if class_ind in retained_class:
                csv_write(csv_path, [label_values[class_ind]], skip_line=True)
                scores_spe = np.nan*np.ones(
                    (len(gt_map_list), len(column_name)))
                scores_glob = np.nan*np.ones(
                    (len(gt_map_list), len(column_name)))

                for ind_mp, (gt_mp, sc_mp, acq_nm) in enumerate(
                        zip(gt_map_list, score_map_list, acq_name)):
                    # Avoid bias when the class is not represented
                    if class_ind in np.unique(gt_mp):
                        # Apply decision thresholds
                        process_gt_mp = np.zeros(gt_mp.shape+(nb_class,))
                        process_sc_mp_spe = np.zeros(sc_mp.shape)
                        process_sc_mp_glob = np.zeros(sc_mp.shape)

                        process_sc_mp_spe[:, class_ind] = binary_thresh(
                            sc_mp[:, class_ind], optm_thrsh[class_ind][acq_nm])
                        process_sc_mp_glob[:, class_ind] = binary_thresh(
                            sc_mp[:, class_ind],
                            optm_thrsh[class_ind]['Global'])

                        process_gt_mp[:, class_ind][gt_mp == class_ind] = 1

                        # Compute metrics
                        for ind_mod, (mod, func) in enumerate(
                                MODALITIES.items()):
                            if  mod != 'TNR':
                                scores_spe[ind_mp, ind_mod] = func(
                                    process_gt_mp[:, class_ind],
                                    process_sc_mp_spe[:, class_ind])
                                scores_glob[ind_mp, ind_mod] = func(
                                    process_gt_mp[:, class_ind],
                                    process_sc_mp_glob[:, class_ind])
                            else:
                               
                                scores_spe[ind_mp, ind_mod] = func(
                                    gt_mp, sc_mp[:, class_ind],
                                    pos_label=class_ind)
                                scores_glob[ind_mp, ind_mod] = scores_spe[
                                    ind_mp, ind_mod]

                # Write scores
                csv_write(csv_path, ['Acquisition specific scores'])
                csv_write(csv_path, scores_spe, columns=column_name,
                          index=acq_name)
                # Compute metric averages
                avg = list(np.nanmean(scores_spe, axis=0))
                std = list(np.nanstd(scores_spe, axis=0))
                txt = ['Average'] + ['{:.2f} +/- {:.4f}'.format(
                    x, y) for x, y in zip(avg, std)]
                csv_write(csv_path, txt)

                csv_write(csv_path, ['Global scores'], skip_line=True)
                csv_write(csv_path, scores_glob, columns=column_name,
                          index=acq_name)
                # Compute metric averages
                avg = list(np.nanmean(scores_glob, axis=0))
                std = list(np.nanstd(scores_glob, axis=0))
                txt = ['Average'] + ['{:.2f} +/- {:.4f}'.format(
                    x, y) for x, y in zip(avg, std)]
                csv_write(csv_path, txt)

    # Compute metrics on raw prediction
    csv_path = os.path.join(CONF_LOADING['out_path'], 'stat_report.csv')
    # Overwrite existing csv file
    with open(csv_path, 'w'):
        pass

    for class_ind in range(nb_class):
        # Only compute metrics on not ignored classes
        if class_ind in retained_class:
            csv_write(csv_path, [label_values[class_ind]], skip_line=True)
            scores = np.nan*np.ones((len(gt_map_list), len(column_name)))

            for ind_mp, (gt_mp, sc_mp, acq_nm) in enumerate(
                    zip(gt_map_list, score_map_list, acq_name)):
                # Avoid bias when the class is not represented
                if class_ind in np.unique(gt_mp):
                    # Apply decision thresholds
                    process_gt_mp = np.zeros(gt_mp.shape+(nb_class,))
                    process_sc_mp = np.zeros(sc_mp.shape)

                    # Remove ignored class confidence
                    for ind in ignored_labels:
                        sc_mp[:, ind] = -1
                    pred = np.argmax(sc_mp, axis=-1)

                    process_sc_mp[:, class_ind][pred == class_ind] = 1
                    process_gt_mp[:, class_ind][gt_mp == class_ind] = 1

                    # Compute metrics
                    for ind_mod, (mod, func) in enumerate(MODALITIES.items()):
                        if mod != 'TNR':
                            scores[ind_mp, ind_mod] = func(
                                process_gt_mp[:, class_ind],
                                process_sc_mp[:, class_ind])
                        else:
                            scores[ind_mp, ind_mod] = func(
                                gt_mp, sc_mp[:, class_ind],
                                pos_label=class_ind)

            # Write scores
            csv_write(csv_path, scores, columns=column_name,
                      index=acq_name)

            # Compute metric averages
            avg = list(np.nanmean(scores, axis=0))
            std = list(np.nanstd(scores, axis=0))
            txt = ['Average'] + [
                '{:.2f} +/- {:.4f}'.format(x, y) for x, y in zip(avg, std)]
            csv_write(csv_path, txt)

    if CONF_LOADING['verbose']:
        plt.show()


CRITERIONS = {
    'F1': f1_score,
    'MCC': matthews_corrcoef}

MODALITIES = {
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score,
        'F1 score': f1_score,
        'MCC': matthews_corrcoef,
        'AUC': custom_auc,
        'Specificity': specificity}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="Specify configuration path")
    args = parser.parse_args()
    PARSED_YML = yaml.load(open(args.config_file),
                           Loader=yaml.FullLoader)
    CONF_LOADING = PARSED_YML['loading']

    # Load the experiment dataset configuration
    LOADING_YML = yaml.load(open(CONF_LOADING['training_config_path']),
                            Loader=yaml.FullLoader)
    CONF_DATASET = LOADING_YML['dataset']
    CONF_DATASET['loading']['folder'] = CONF_LOADING['data_path']
    CONF_DATASET['loading']['class_index'] = CONF_LOADING['class_index']

    # Dataset name
    DATASET = CONF_DATASET['dataset_name']

    # Evaluation configuration
    EVAL_CONF = PARSED_YML['evaluation']

    main()
