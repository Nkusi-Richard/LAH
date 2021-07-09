# -*- coding: utf-8 -*-
import random
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, \
    balanced_accuracy_score, f1_score, cohen_kappa_score, \
    precision_score, recall_score
import sklearn.model_selection
from scipy import ndimage
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import itertools
import spectral
# import visdom
import matplotlib.pyplot as plt
from mpltools import special
from scipy import io
import imageio
import os
import copy
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import cv2
from plot_matconf import plot_confusion_matrix
from PIL import Image
import scipy.ndimage

np.random.seed()


def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("\nComputation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("\nComputation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available!"
              "Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))


def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.

    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)

    Returns:
        arr_2d: int 2D array of labels

    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        print(c)
        m = np.all(arr_3d == np.array(c))
        arr_2d[m] = i

    return arr_2d


def display_predictions(pred, vis, info_data, gt=None, caption=""):
    img_name = info_data['img_name']
    [h, w] = info_data['img_shape']
    nb_img = len(img_name)
    if pred.shape[1] > w + 1:
        white = np.ones((h, 2, 3)) * 255
        for x in range(1, nb_img):
            pred[:, x * w:x * w + 2, :] = white
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                   opts={'caption': caption})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                   nrow=2,
                   opts={'caption': caption})


def display_dataset(img, vis, info_data):
    """Display the specified hypercube.
    Args:
    - img: 3D hyperspectral image
    - vis (Visdom.visdom): visdom display
    - info_data (dict): dictionary containing dataset info
    """
    rgb_list, _ = rgb_generator(info_data, img=img, greyscale=False)
    rgb = np.concatenate(tuple(rgb_list), axis=1)

    # send to visdom server
    vis.images([np.transpose(rgb, (2, 0, 1))],
               opts={'caption': 'RGB representation'})


def explore_spectrums(img, complete_gt, class_names, vis,
                      info_data, ignored_labels=None, plot_name='',
                      exp_name='', plot_scale=[0, 1]):
    """Plot and save sampled spectrums with mean & std for each class.
      Args:
    - img (np.array): 3D hyperspectral image
    - complete_gt (np.array): 2D array of labels
    - class_names (list[str]): list of class names
    - ignored_labels (list[int]): list of labels to ignore
    - vis (visdom.Visdom) : Visdom display
    - info_data (dict): dictionary containing preprocess information
    - plot_name (str): name of the plot to be saved
    - exp_name (str): experiment name
    - plot_scale (list[str]): y axis limits
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    info_spectrums = {}

    # Configure x_scale:
    start = 0
    stop = img.shape[-1]
    if 'band_downsampling' in info_data.keys():
        [start, stop] = info_data['band_downsampling']
    bd_range = [500 + start * 5, 1000 - (100 - stop) * 5]
    x_scale = np.arange(bd_range[0], bd_range[1], 5)

    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)

        # Prepare figures
        plt.figure()
        ax = plt.gca()
        ax.set_ylim(plot_scale[0], plot_scale[1])
        plt.title('{}: {}'.format(plot_name, class_names[int(c)]))
        ax.set_ylim(plot_scale[0], plot_scale[1])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Relative Reflectance (a.u.)')

        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(x_scale, spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)

        # Plot the mean spectrum with thickness based on std
        special.errorfill(x_scale, mean_spectrum, std_spectrum,
                          color="#3F5D7D", alpha_fill=0.6)

        if 'overall' in plot_name:
            vis.matplot(plt)
        info_spectrums[class_names[int(c)]] = [mean_spectrum, std_spectrum]

        # Save figure:
        save_figure(exp_name, class_names[int(c)], folder=plot_name,
                    modality='Curves')

    return info_spectrums


def plot_spectrums(spectrums, vis, info_data, label_values, exp_name,
                   palette, retained_class=[],
                   title="", plot_scale=[0, 1], plot_name=''):
    """Plot the specified dictionary of spectrums.
    Args:
    - spectrums (dict): dictionary containing mean and std spectrums
    - vis (Visdom.visdom): Visdom display
    - info_data (dictionary): contains dataset information
    - label_values (list[str]): list containing labels
    - exp_name (str): experiment name
    - palette (dict): dictionary containing color map
    - retained_class (list[str]): classes of interest to display their std
    - title (str): plot title
    - plot_scale (list[int]): y axis limits
    - plot_name (str): name of the plot to be saved
    """
    win = None
    fig = {}

    # Configure x_scale
    try:
        [start, stop] = info_data['band_downsampling']
    except KeyError:
        start = 0
        stop = len(next(iter(spectrums.values()))[0])
    bd_range = [500 + start * 5, 1000 - (100 - stop) * 5]
    x_scale = np.arange(bd_range[0], bd_range[1], 5)

    for x in range(2):
        fig[x] = plt.figure()
        ax = plt.gca()
        ax.set_ylim(plot_scale[0], plot_scale[1])
        plt.title('{}: mean spectral curves'.format(title))

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Relative Reflectance (a.u.)')

    for label, [mean_spec, std_spec] in spectrums.items():
        color = [x / 255.0 for x in palette[label_values.index(label)]]

        # Sample and plot spectrums from the selected class
        plt.figure(fig[0].number)
        plt.plot(x_scale, mean_spec, color=color, label=label)

        plt.figure(fig[1].number)
        if not(label in retained_class):
            std_spec = 0

        special.errorfill(x_scale, mean_spec, std_spec, label=label,
                          color=color, alpha_fill=0.5)

        # Plot on visdom instance
        n_bands = len(mean_spec)
        update = None if win is None else 'append'
        win = vis.line(X=np.arange(n_bands), Y=mean_spec, name=label,
                       win=win, update=update,
                       opts={'title': title})

    for x, preff in enumerate(['', '_std']):
        plt.figure(fig[x].number)
        plt.legend(loc="lower right")

        # save curve figures
        out_nm = 'All_Classes{}'.format(preff)
        save_figure(exp_name, out_nm, folder=plot_name, modality='Curves')
        if not(retained_class):
            break


def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples

    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)


def get_random_pos(img, window_shape):
    """ Return the corners of a random window in the input image

    Args:
        img: 2D (or more) image, e.g. RGB or grayscale image
        window_shape: (width, height) tuple of the window

    Returns:
        xmin, xmax, ymin, ymax: tuple of the corners of the window

    """
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, exp_name, info_data, ignored_labels=[],
            cross_dict=None, n_classes=None, labels=None,
            patient_specific=False):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    n_classes = np.max(target) + 1 if n_classes is None else n_classes
    info_spe = copy.deepcopy(info_data)
    _, name_list = rgb_generator(info_spe, cross_dict=cross_dict)
    if cross_dict:
        cv_step = cross_dict['cv_step']
        cv_coord = cross_dict['cv_coord'][cv_step]
        prediction = prediction[:, cv_coord[0]:cv_coord[1]]
        target = target[:, cv_coord[0]:cv_coord[1]]
        folder = 'Kfold_specific/'
    else:
        folder = '{}'

    if patient_specific:
        info_spe['img_name'] = name_list
        folder = 'patient_specific/{}'
    else:
        info_spe = {'img_name': ['']}
        if cross_dict:
            name_list = ['k{}'.format(cv_step + 1)]
        else:
            name_list = ['overall']

    pred_list = map_generator(prediction, info_spe, cross_dict=None)
    target_list = map_generator(target, info_spe, cross_dict=None)

    results = {}
    for name, target, prediction in zip(name_list, target_list,
                                        pred_list):
        ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
        for l in ignored_labels:
            ignored_mask[target == l] = True
        ignored_mask = ~ignored_mask
        target = target[ignored_mask]
        prediction = prediction[ignored_mask]

        cm = confusion_matrix(
            target,
            prediction,
            labels=range(n_classes))

        # remove NaN lines and plot cm
        cm_disp = np.delete(cm, ignored_labels, axis=0)
        cm_disp = np.delete(cm_disp, ignored_labels, axis=1)
        labels_disp = np.delete(np.array(labels), ignored_labels)
        plot_confusion_matrix(cm_disp, labels_disp, normalize=True,
                              title='Confusion Matrix: ' + name)

        # Save figure and pickle file
        out_nm = 'CM_{}'.format(os.path.basename(name))
        save_figure(exp_name, out_nm,
                    folder=folder.format(os.path.dirname(name)),
                    save_obj=cm,
                    modality='pkl_files')
        save_figure(exp_name, out_nm,
                    folder=folder.format(os.path.dirname(name)),
                    modality='CM')

        # Compute global accuracy
        accuracy = 100*accuracy_score(target, prediction)
        balanced_acc = 100*balanced_accuracy_score(target, prediction)

        # Compute F1 score, precision & recall
        f1_scores = np.nan*np.ones(n_classes)
        precision = np.nan*np.ones(n_classes)
        recall = np.nan*np.ones(n_classes)

        # TODO: check if one class is missing in prediction or target
        # Some rates might be NaN
        f1_raw = 100*f1_score(target, prediction, average=None)
        precision_raw = 100*precision_score(target, prediction, average=None)
        recall_raw = 100*recall_score(target, prediction, average=None)
        for i, class_index in enumerate(
                np.unique(np.concatenate((prediction, target)))):
            f1_scores[int(class_index)] = f1_raw[i]
            precision[int(class_index)] = precision_raw[i]
            recall[int(class_index)] = recall_raw[i]

        # Compute micro/macro/weighted average metrics
        average = {"micro": {}, "macro": {}, "weighted": {}}
        for mode, score in average.items():
            f1_avg = 100*f1_score(target, prediction, average=mode)
            prec_avg = 100*precision_score(target, prediction, average=mode)
            rec_avg = 100*recall_score(target, prediction, average=mode)
            score.update({"Precision": prec_avg,
                          "Recall": rec_avg,
                          "F1 score": f1_avg})

        # Compute kappa score
        kappa = cohen_kappa_score(target, prediction)

        results[name] = {"Confusion matrix": cm,
                         "Accuracy": accuracy,
                         "Balanced accuracy": balanced_acc,
                         "Precision": precision,
                         "Recall": recall,
                         "F1 score": f1_scores,
                         "Average": average,
                         "Kappa": kappa}

    return results


def show_results(results, vis, exp_name, ignored_labels, label_values=None,
                 agregated=False, subset=''):
    text = "\n Results {}:\n".format(subset)

    check = not(subset)

    if agregated:
        cm_sum = []
        zero = np.zeros((len(label_values), len(results)))
        class_scores = {"Precision": np.copy(zero),
                        "Recall": np.copy(zero),
                        "F1 score": np.copy(zero)}

        tmpl_scores = {"Precision": [],
                       "Recall": [],
                       "F1 score": []}
        avg_scores = {"micro": copy.deepcopy(tmpl_scores),
                      "macro": copy.deepcopy(tmpl_scores),
                      "weighted": copy.deepcopy(tmpl_scores)}
        global_scores = {"Accuracy": [],
                         "Balanced accuracy": [],
                         "Kappa": []}

        for x, (name, r) in enumerate(results.items()):
            # Retain only acquisitions from the specified subset
            # if subset is not specified, all acquisitions are retained
            if (subset in name) or check:
                for metric, scores in class_scores.items():
                    scores[:, x] = r[metric]
                    for mode, avg_metrics in avg_scores.items():
                        avg_metrics[metric].append(r["Average"][mode][metric])

                for metric, scores in global_scores.items():
                    scores.append(r[metric])
                cm_sum.append(r["Confusion matrix"])

        # Class-based scores averaged over acquisitions
        classes_avg = {}
        for metric, scores in class_scores.items():
            classes_avg[metric] = [np.nanmean(scores, axis=1),
                                   np.nanstd(scores, axis=1)]

        # Global scores averaged over acquisitions
        global_avg = {}

        for metric, scores in global_scores.items():
            global_avg[metric] = [np.mean(scores), np.std(scores)]

        for mode, avg_metrics in avg_scores.items():
            for metric, scores in avg_metrics.items():
                score_template = "{} ({})".format(metric, mode)
                global_avg[score_template] = [np.mean(scores),
                                              np.std(scores)]

        # remove NaN lines and plot cm
        cm_sum = np.sum(cm_sum, axis=0)
        cm_disp = np.delete(cm_sum, ignored_labels, axis=0)
        cm_disp = np.delete(cm_disp, ignored_labels, axis=1)
        labels_disp = np.delete(np.array(label_values), ignored_labels)
        plot_confusion_matrix(
            cm_disp, labels_disp, normalize=True,
            title='Confusion Matrix: {} overall'.format(subset))
        cm = cm_sum

        # Save confusion matrix figure
        out_jpg = 'overall_{}'.format(subset)
        out_pkl = 'CM_CrossVal_{}'.format(subset)
        save_figure(exp_name, out_jpg, modality='CM')
        save_figure(exp_name, out_pkl, save_obj=cm, modality='pkl_files')

        text += "Agregated results :\n"
        for i, (metric, score) in enumerate(global_avg.items()):
            if i % 3 == 0 and i > 0:
                text += "\n"
            text += ("{0:20}: {1:.03f} +- {2:.03f}\n".format(
                metric, score[0], score[1]))

        for metric, scores in classes_avg.items():
            text += "{} :\n".format(metric)
            for label, mean, std in zip(label_values, scores[0], scores[1]):
                if not np.isnan(mean):
                    text += "\t{:12}: {:.03f} +- {:.03f}\n".format(
                        label, mean, std)
            text += "---\n"
    else:
        text += "{} fold:\n".format(next(iter(results.keys())))
        results = next(iter(results.values()))

        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        balanced_acc = results["Balanced accuracy"]
        kappa = results["Kappa"]

        text += "{0:20}: {1:.03f}%\n".format('Accuracy', accuracy)
        text += "{0:20}: {1:.03f}%\n".format('Balanced accuracy', balanced_acc)
        text += "{0:20}: {1:.03f}\n".format('Kappa', kappa)
        text += "---\n"

        metrics = ["F1 score", "Precision", "Recall"]
        for metric in metrics:
            text += "{} :\n".format(metric)
            for label, score in zip(label_values, results[metric]):
                if not np.isnan(score):
                    text += "\t{}: {:.03f}\n".format(label, score)
            text += "---\n"

    # vis.text(text.replace('\n', '<br/>'))
    print(text)
    return text


def sample_gt(gt, train_size, mode='random', cross_dict=None):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
    - gt: a 2D array of int labels
    - train_size: [0, 1] float
    - cross_dict: dict (contains cross_val infos)
    Kwargs:
    - mode: str (sampling mode), choose among:
    {'random', 'fixed', 'disjoint', 'cross_val'}
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)

    if mode == 'random':
        train_indices, test_indices = sklearn.model_selection.train_test_split(
            X, train_size=train_size, stratify=y)
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
        test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
    elif mode == 'fixed':
        print("Sampling {} with train size = {}".format(mode, train_size))
        train_indices, test_indices = [], []
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features

            train, test = sklearn.model_selection.train_test_split(
                X, train_size=train_size)
            train_indices += train
            test_indices += test
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[train_indices] = gt[train_indices]
        test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            total = np.count_nonzero(mask)
            for x in range(gt.shape[1]):
                first_half_count = np.count_nonzero(mask[:x, :])
                try:
                    # ratioCb train/test:
                    # second_half_count = np.count_nonzero(mask[x:, :])
                    # ratio = first_half_count / second_half_count

                    # train proportion (ratio train/total):
                    ratio = first_half_count / total
                    if ratio > 0.98 * train_size and ratio < 1.2 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0
        test_gt[train_gt > 0] = 0

    elif mode == 'cross_val':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        cv_coord = cross_dict['cv_coord'][cross_dict['cv_step']]
        mask = gt != 0
        mask[:, cv_coord[0]:cv_coord[1]] = 0
        test_gt[mask] = 0
        train_gt[test_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    """ Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.

    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    Args:
        ground_truth: the annotations array
        n_classes: number of classes (optional, defaults to max(ground_truth))
        ignored_classes: id of classes to ignore (optional)
    Returns:
        numpy array with the IMF coefficients
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights


def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def data_stat(train_gt, gt, img, label_values):
    """ Compute and display basic data statistics
    Args:
    - train_gt (np.array): training ground-truth map
    - gt (np.array): ground-truth map
    - img (np.array): HSI hypercube
    - label_values (list[str]): list containing label names
    """
    samples = np.count_nonzero(train_gt)
    total = np.count_nonzero(gt)
    print(np.unique(gt))
    print("Image has dimensions {}x{} and {} channels".format(
        *img.shape))
    print("{} samples selected for training over {} ({}%)".format(
        samples, total, round(samples * 100 / total, 2)))
    print('\n{0} classes, {1}'.format(len(label_values), label_values))
    stat = np.zeros((len(label_values,)))
    for c in np.unique(gt):
        stat[int(c)] = np.count_nonzero(train_gt == c)
    summ = np.sum(stat[1:])

    # TODO: check ZeroDivisionError
    print('Classes occurences (training set):')
    for lbl, val in zip(label_values[1:], stat[1:]):
        print("{} {} {}%".format(
            lbl, int(val), round(val / summ * 100, 2)))


def display_legend(classes, vis, palette, exp_name, ignored_labels=[],
                   verbose=False):
    """ Generate and save a color map legend
    Args:
    - classes (list[str]): list containing label names
    - vis (Visdom.visdom): visdom display
    - palette (dict): color map
    - ignored_labels (list[int]): list containing label index to be ignored
    - final_shape (tuple): shape used to resize output image
    """
    plt.figure()
    subplot_size = len(classes) - len(ignored_labels)

    count = 0
    for i, label in enumerate(classes):
        if i not in ignored_labels:
            square = np.ones((5, 5)) * (i)
            color_square = convert_to_color_(square, palette=palette)
            ax = plt.subplot2grid((subplot_size, 1), (count, 0))
            ax.yaxis.set_label_position("right")
            plt.imshow(color_square)
            y_lbl = plt.ylabel(label, labelpad=30)
            y_lbl.set_rotation(0)
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

            ax.set_yticklabels([])
            ax.set_xticklabels([])
            count += 1

    # Save figure
    save_figure(exp_name, 'legend', modality='Visualization')

    # Display legend
    vis.matplot(plt)
    if verbose:
        plt.show()


def display_error(img, pred, err, gt_mask, pred_mask, vis, info_data,
                  classes, palette, gt, exp_name='',
                  cross_dict=None):
    """ Display and save error visualization.
    Args:
    - img (np.array): hypercube
    - pred (np.array): colored prediciton map
    - err (np.array): colored error map
    - vis (Visdom.visdom): visdom display
    - info_data (dict): dictionary containing dataset info
    - classes (list[str]): list containing labels
    - palette (dict): dictionary containing color map
    - gt (np.array): colored ground-truth map
    - exp_name (str): experiment name
    - cross_dict (dict): dictionary containing cross val info
    """
    rgb_list, img_name = rgb_generator(info_data, img=img,
                                       cross_dict=cross_dict,
                                       greyscale=False)
    grey_list, _ = rgb_generator(info_data, img=img,
                                 cross_dict=cross_dict)
    pred_mask_list = map_generator(pred_mask, info_data, cross_dict=cross_dict)
    pred_list = map_generator(pred, info_data, cross_dict=cross_dict)
    gt_list = map_generator(gt, info_data, cross_dict=cross_dict)
    err_list = map_generator(err, info_data, cross_dict=cross_dict)
    gt_mask_list = map_generator(gt_mask, info_data, cross_dict=cross_dict)

    for i, (name, grey_d, rgb_d, pred_d, err_d, gt_d, pred_mas, gt_mask_d) in enumerate(zip(
            img_name, grey_list, rgb_list, pred_list, err_list, gt_list, pred_mask_list, gt_mask_list)):
        """
        white = np.ones((err_d.shape[0], 3)) * 255
        pred_d[:, -1, :] = white
        gt_d[:, -1, :] = white
        rgb_d[:, -1, :] = white
        err_d[:, -1, :] = white

        # display on visdom
        vis.images([np.transpose(rgb_d, (2, 0, 1)),
                    np.transpose(gt_d, (2, 0, 1)),
                    np.transpose(pred_d, (2, 0, 1)),
                    np.transpose(err_d, (2, 0, 1))],
                   # np.transpose(legend, (2, 0, 1))],
                   nrow=4,
                   opts={'caption': name})"""

        # save image
        tg = np.copy(gt_d)
        mask = gt_d == 0
        print("MASK SHAPE")
        print(mask.shape)
        ma = np.nonzero(gt_d)
        print(ma)
        xcropp = ma[0]
        ycropp = ma[1]
        ymin, ymax = int(ycropp.min()), int(np.ceil(ycropp.max()))
        xmin, xmax = int(xcropp.min()), int(np.ceil(xcropp.max()))

        """
        rgb_dd = np.dot(rgb_d[...,:3], [0.2989, 0.5870, 0.1140])
        print(rgb_dd.shape)
        pred_dd = convert_from_color_(pred_d, palette=palette)
        print(pred_dd.shape)
        mask_rgb = rgb_d >= 0
        
        print(np.unique(pred_d))
        pred_d[mask_rgb] = 1
        print("OKKKKKKKKKKKKKKKKKKK")
        """
        pred_mask = np.copy(pred_d)
        pred_mask[tg == 0] = 0
        #pred_mas =  pred_mask != 0
        #pred_mask[:, -1, :] = white

        alpha = 1
        disp_img_ = np.dot(rgb_d[...,:3], [0.2989, 0.5870, 0.1140])
        mask_disp = disp_img_ >= 180
        disp_img_[~mask_disp] = 0
        disp_img_ =  np.stack((disp_img_,)*3,axis=-1).astype(np.uint8)
        #pred_img = overlayed_img(grey_d, pred_d, pred_mas, alpha=alpha)
        #gt_img = overlayed_img(grey_d, gt_d,~mask, alpha=alpha)
        #out_img = np.concatenate((rgb_d, gt_img, pred_img, err_d, pred_d, disp_img_),
        #                         axis=1)
        pred_img = overlayed_img(rgb_d, pred_d, pred_mas, alpha=alpha)
        gt_img = overlayed_img(rgb_d, gt_d,gt_mask_d, alpha=alpha)
        rgb_d = rgb_d[xmin-5:xmax+5, ymin-5:ymax+5]
        gt_img = gt_img[xmin-5:xmax+5, ymin-5:ymax+5]
        pred_img = pred_img[xmin-5:xmax+5, ymin-5:ymax+5]

        zoom_factor = 10
        out_img = np.concatenate((ndimage.zoom(rgb_d, (zoom_factor,) * 2 + (1,) * (rgb_d.ndim - 2)), ndimage.zoom(gt_img, (zoom_factor,) * 2 + (1,) * (gt_img.ndim - 2)),ndimage.zoom(pred_img, (zoom_factor,) * 2 + (1,) * (pred_img.ndim - 2))),
                                 axis=1)
        #out_img = np.concatenate((rgb_d, gt_img, pred_img),
        #                         axis=1)

        print("RGB_D")
        print(rgb_d.shape)
        print(rgb_d.shape)
        #out_img = rgb_d
        #out_img = np.concatenate((ndimage.zoom(rgb_d, (10,10,10))),
        #                         axis=1)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        save_figure(exp_name, name, save_obj=out_img, folder='Err',
                    modality='Visualization')


def display_histo(img, log=True):
    '''Display the histogram of the whole dataset
    Args:
    - img (np.array): hypercube
    - log (bool): to plot the y axis with a log scale
    '''
    # histogramme:
    histo = np.copy(img).flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(histo, bins='auto')
    # ax.set_xlim(-2, 42)
    ax.grid(True)
    if log:
        plt.yscale('log')


def concat_rgb(info_data):
    '''Concatenate rgb image for medical dataset
    Args:
    - info_data (dict): dictionary containing data information
    Returns:
    - rgb (np.array): concatenated rgb images
    '''
    rgb_tmpl = '{0}_RGB-Image.png'
    width = info_data['img_shape'][1]

    # form rgb image
    rgb = np.zeros(tuple(info_data['cube_shape']) + (3,))
    for i, acquisition in enumerate(info_data['img_name']):
        img_name = os.path.basename(acquisition).split(' ')[0]
        path = os.path.join(info_data['folder'], acquisition,
                            rgb_tmpl.format(img_name))
        img = Image.open(path)
        # rotate
        rgb[:, i * width:(i + 1) * width, :] = scipy.ndimage.rotate(img, 270)
    return rgb


def normalization(img, thresh=None):
    '''Rescale data values between 0 and 1. If thresh is specified,
    the values are truncated pre-normalization.
    Args:
    - img (np.array): hypercube
    - thresh (float): threshold to truncate the values
    Returns:
    - img (np.array): normalized hypercube
    '''
    # display_histo(img)
    if not thresh:
        thresh = np.amax(img)
    img[img > thresh] = thresh  # troncate higher values
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # display_histo(img)
    return img


def standardization(img, thresh):
    '''Transform data to have a mean of 0 and a std of 1. If thresh is specified,
    the values are truncated post-standardization
    Args:
    - img (np.array): hypercube
    - thresh (float): threshold to truncate the values
    Returns:
    - img (np.array): standardized hypercube
    '''
    # display_histo(img)
    img = (img - np.mean(img)) / np.std(img)
    if not thresh:
        thresh = np.amax(img)
    img[img > thresh] = thresh
    # display_histo(img)
    return img


def channel_wised_normalization(img):
    '''Realize a normalization respecting the band max/min
    '''
    for i in range(img.shape[-1]):
        max_bd = np.amax(img[:, :, i])
        min_bd = np.amin(img[:, :, i])
        img[:, :, i] = (img[:, :, i] - min_bd) / (max_bd - min_bd)
    return img


def HSI_ml(img, gt, ignored_labels, label_values, palette, mode='LDA',
           n_components=2):
    """ Compute a Principal Components Analysis on the HSI hypercube
    """
    print("Applying {} to HSI cube...".format(mode))
    for x in ignored_labels:
        gt[gt == x] = 0

    img = img[gt > 0]
    gt = gt[gt > 0]

    if mode == 'LDA':
        model = LinearDiscriminantAnalysis(n_components=n_components)
        # X_transformed = model.fit_transform(img, gt)
    elif mode == 'PCA':
        model = IncrementalPCA(n_components)
        # X_transformed = model.fit_transform(img)
    exp_var = model.explained_variance_ratio_

    print("Explained variance: {} (total:{}%)".format(
        exp_var, 100 * np.sum(exp_var)))

    # if n_components == 2:
    #     lbl = np.unique(gt).astype('int')
    #     label_values = [label_values[x] for x in lbl]
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #     ax.set_xlabel('Comp. 1')
    #     ax.set_ylabel('Comp. 2')
    #     ax.set_title('2 component {}\n'
    #                  'explained variance: {}%'.format(mode, round(
    #                      100*np.sum(exp_var))))

    #     for ind in lbl:
    #         indicesToKeep = X_transformed[gt == ind]
    #         ax.scatter(indicesToKeep[:, 0], indicesToKeep[:, 1],
    #                    cmap=palette,
    #                    s=10)
    #     ax.legend(label_values)
    #     ax.grid()


def class_fusion(fused_class, gt, ignored_labels, label_values, class_name):
    """ Function to fuse classes for hierarchy classification.
    Please pay attention that if on class inside fused_class is ignored, all
    other fused classes will be ignored also. Moreover, to keep a pleasant
    display of the prediction, please do not includ unclassified class in
    the fused classes.
    Args:
    - fused_class (list[str]): list containing the name of the class  to fuse
    - gt (np.array): ground truth mask
    - ignored_labels (list[int]): list containing label index to ignore
    - label_values (list[str]): list containing class labels
    - class_name (str): name of the class created from fusion
    Returns:
    - gt (np.array): updated ground truth mask
    - ignored_labels (list[int]): list containing updated label index to ignore
    - label_values (list[str]): list containing updated class labels
    """
    fused_index = [label_values.index(x) for x in fused_class]
    fused_ignored = [i for i in ignored_labels if i in fused_index]
    fused_index.sort(reverse=True)
    if fused_ignored:
        for x in fused_ignored:
            ignored_labels.pop(ignored_labels.index(x))
    ignored_labels = np.array(ignored_labels)
    for x in fused_index:
        gt[gt == x] = -1
        gt[gt > x] -= 1
        ignored_labels[ignored_labels > x] -= 1
        del label_values[x]
    label_values.append(class_name)
    gt[gt == -1] = len(label_values) - 1

    ignored_labels = list(ignored_labels)

    if fused_ignored:
        ignored_labels.append(len(label_values) - 1)
    return gt, list(ignored_labels), label_values


class FocalLoss(nn.CrossEntropyLoss):
    r"""This criterion Focal Loss was introduced by Lin et al., from Facebook,
     in this paper.
     They claim to improve one-stage object detectors using Focal Loss to train
     a detector they name RetinaNet.
     Focal loss is a Cross-Entropy Loss that weighs the contribution of each
    sample to the loss based in the classification error.
     The idea is that, if a sample is already classified correctly by the CNN,
     its contribution to the loss decreases. With this strategy, they claim to
    solve the problem of class imbalance by making the loss implicitly focus
    in those problematic classes.
     Moreover, they also weight the contribution of each class to the lose in
    a more explicit class balancing.

    Args:
        - weight (Tensor, optional): a manual rescaling weight given to each
    class. If given, has to be a Tensor of size `C`
        - size_average (bool, optional): Deprecated (see :attr:`reduction`).
    By default, the losses are averaged over each loss element in the batch.
    Note that for some losses, there are multiple elements per sample.
    If the field :attr:`size_average` is set to ``False``, the losses are
    instead summed for each minibatch. Ignored when reduce is ``False``.
    Default: ``True``
        - ignore_index (int, optional): Specifies a target value that is
    ignored and does not contribute to the input gradient.
    When :attr:`size_average` is ``True``, the loss is averaged over
    non-ignored targets.
        - reduce (bool, optional): Deprecated (see :attr:`reduction`).
    By default, the losses are averaged or summed over observations for each
    minibatch depending on :attr:`size_average`. When :attr:`reduce` is
    ``False``, returns a loss per batch element instead and ignores
    :attr:`size_average`. Default: ``True``
        - reduction (string, optional): Specifies the reduction to apply to the
    output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be
    applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    Note: :attr:`size_average` and :attr:`reduce` are in the process of being
    deprecated, and in the meantime, specifying either of those two args will
    override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq
    \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.
    """
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction)
        self.reduction = reduction
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        CE_loss = F.cross_entropy(input, target, reduction='none',
                                  weight=self.weight)
        # CE_loss = nn.CrossEntropyLoss(reduction='none')(input, target)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * CE_loss

        if (self.reduction != 'none'):
            return torch.mean(F_loss)
        else:
            return F_loss


def ROC_curve(probabilities, gt, ignored_labels, label_values, exp_name,
              palette, vis, info_data, cross_dict=None,
              patient_specific=False):
    """ Compute and possibly plot the ROC curve for each class
    Args:
    - probabilities (np.array): class confidence scores of each pixel
    - gt (np.array): ground truth mask
    - ignored_labels (list(int)): list containing label indexes to be ignored
    - label_values (list(str)): list containing class names
    - exp_name (str): name of the experiment
    - palette (dict): dicitonary of colors
    - vis (Visdom.visdom): visdom display
    - cross_dict (dict): dictionary containing cross validation info
    - patient_specific (bool): activate patient-specific report mode, if False
    k-fold specific mode is activated by default
    """
    info_spe = copy.deepcopy(info_data)
    _, name_list = rgb_generator(info_spe, cross_dict=cross_dict)
    if cross_dict:
        cv_step = cross_dict['cv_step']
        cv_coord = cross_dict['cv_coord'][cv_step]
        probabilities = probabilities[:, cv_coord[0]:cv_coord[1], :]
        gt = gt[:, cv_coord[0]:cv_coord[1]]
        folder = 'Kfold_specific/{}'
    else:
        folder = '{}'

    if patient_specific:
        info_spe['img_name'] = name_list
        folder = 'patient_specific/{}'
    else:
        info_spe = {'img_name': ['']}
        if cross_dict:
            name_list = ['k{}'.format(cv_step + 1)]
        else:
            name_list = ['overall']

    proba_list = map_generator(probabilities, info_spe, cross_dict=None)
    gt_list = map_generator(gt, info_spe, cross_dict=None)

    roc_dict = {}
    for name, proba_d, gt_d in zip(name_list, proba_list, gt_list):
        ignored_mask = np.zeros(gt_d.shape[:2], dtype=np.bool)
        for l in ignored_labels:
            ignored_mask[gt_d == l] = True
        ignored_mask = ~ignored_mask
        y_true = gt_d[ignored_mask]
        y_score = proba_d[ignored_mask]

        fpr = dict()
        tpr = dict()
        thresh = dict()
        class_ind = dict()
        roc_auc = dict()
        for i, class_id in enumerate(np.unique(y_true)):
            class_id = int(class_id)
            class_ind[i] = label_values[class_id]
            fpr[i], tpr[i], thresh[i] = roc_curve(y_true, y_score[:, class_id],
                                                  pos_label=class_id)
            roc_auc[i] = auc(fpr[i], tpr[i])

            color = [x / 255.0 for x in palette[class_id]]

            lw = 2
            plt.figure()
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve (area = {0:0.2f})'.format(roc_auc[i]))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve: {}, {}'.format(name,
                                                 label_values[class_id]))
            plt.legend(loc="lower right")

            # Save figure
            out_nm = '{}_{}'.format(os.path.basename(name),
                                    label_values[class_id])
            save_figure(exp_name, out_nm, folder=folder.format(name),
                        modality='ROC')
            plt.close()

        lw = 2
        plt.figure()
        for i, class_id in enumerate(np.unique(y_true)):
            color = [x / 255.0 for x in palette[class_id]]
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(label_values[int(class_id)], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve: ' + name)
        plt.legend(loc="lower right")

        # Display and save figure
        vis.matplot(plt)
        out_nm = '{}_classes'.format(os.path.basename(name))
        save_figure(exp_name, out_nm, folder=folder.format(name),
                    modality='ROC')

        # Fill directory
        roc_dict[name] = [fpr, tpr, thresh, class_ind]
    return roc_dict


def ROC_combined(ROC_info, exp_name, patient_specific=False, subset=''):
    """ Plot the Kfold-specific or patient-specific ROC curves on the same
    plot for each class
    Args:
    - ROC_info (dict): dictionary containing the patient-specific fpr and tpr
    - exp_name (str): name of the experiment
    - subset (str): specify the subset name to be combined
    """
    if patient_specific:
        folder = 'patient_specific/{}'
    else:
        folder = 'Kfold_specific/'

    lw = 2
    unique_class = []
    for index, [_, _, _, class_ind] in ROC_info.items():
        unique_class += [x for _, x in class_ind.items()]
    unique_class = set(unique_class)

    fig = dict()
    for label in unique_class:
        fig[label] = plt.figure(figsize=[12.8, 9.6])
        plt.title('{} patient-specific ROC curves: {}'.format(
            subset, label))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # ax = plt.gca()
        # ax.set_aspect('equal')

    # Activate the subset checking
    check = not(subset)

    for index, [fpr, tpr, _, class_ind] in ROC_info.items():
        if (subset in index) or check:
            for ind, label in class_ind.items():
                roc_auc = auc(fpr[ind], tpr[ind])
                plt.figure(fig[label].number)
                plt.plot(fpr[ind], tpr[ind], lw=lw,
                         label='{0} ROC curve (area = {1:0.2f})'.format(
                             index, roc_auc))
                if len(ROC_info) > 5:
                    plt.legend(bbox_to_anchor=(1.0, 0.5), loc='center left')
                    plt.tight_layout()
                else:
                    plt.legend(loc="lower right")

    # Save dictionary and figures
    save_figure(exp_name, 'ROC_combined', folder=folder.format(subset),
                save_obj=ROC_info, modality='pkl_files')
    for label in unique_class:
        plt.figure(fig[label].number)
        out_nm = '{}_indiv'.format(label)
        save_figure(exp_name, out_nm, folder=folder.format(subset),
                    modality='ROC')


def overlayed_img(rgb_img, color_pred, mask, alpha=0.4):
    """ Generate a binary image of the positive prediction of the
    specified class overlayed on the RGB image.
    Args:
    - rgb_img (np.array): RGB image
    - color_pred (np.array): associated colored prediction maskxsy
    - retained_class (int): index of the class to consider for the
    - palette (dict): dictionary containing palette color
    - alpha (float): alpha coefficient for transparency overlay
    Returns:
    - overlay_img (np.array): overlayed image
    """
    # Generate overlay
    overlay = np.copy(rgb_img)
    overlay[mask] = color_pred[mask]

    # Aplly overlay:
    overlay_img = cv2.addWeighted(overlay, alpha, rgb_img, 1 - alpha, 0)

    return overlay_img


def display_overlay(img, pred, gt, gt_mask, pred_mask, vis,
                    info_data, palette, cross_dict=None, alpha=0.4,
                    exp_name=''):
    """ Display the prediction pass through a mask overlayed on the
    rgb image of the HSI hypercube.
    Args:
    - rgb (np.array): HSI hypercube
    - pred (np.array): colored prediction matrix
    - gt (np.array): colored ground_truth matrix
    - gt_mask (np.array): mask used to display the ground-truth of interest,
    if True, the corresponing pixel is displayed.
    - pred_mask (np.array): mask use to display the predictions of interest,
    if True, the corresponing pixel is displayed.
    - vis (visdom.Visdom): visdom visualizer
    - info_data (dict): dictionary containing data information
    - palette (dict): dictionary of colors
    - cross_dict (dict): dictionary containing cross validation information
    - alpha (float): alpha coefficient for transparency overlay
    """
    rgb_list, img_name = rgb_generator(info_data, img=img,
                                       cross_dict=cross_dict)
    pred_list = map_generator(pred, info_data, cross_dict=cross_dict)
    gt_list = map_generator(gt, info_data, cross_dict=cross_dict)
    pred_mask_list = map_generator(pred_mask, info_data, cross_dict=cross_dict)
    gt_mask_list = map_generator(gt_mask, info_data, cross_dict=cross_dict)

    for i, (name, rgb_d, pred_d, gt_d, pred_mask_d, gt_mask_d) in enumerate(
            zip(img_name, rgb_list, pred_list, gt_list,
                pred_mask_list, gt_mask_list)):
        pred_img = overlayed_img(rgb_d, pred_d, pred_mask_d, alpha=alpha)
        gt_img = overlayed_img(rgb_d, gt_d, gt_mask_d, alpha=alpha)

        vis.images([np.transpose(pred_img, (2, 0, 1)),
                    np.transpose(gt_img, (2, 0, 1))],
                   nrow=2,
                   opts={'caption': 'overlay ' + name})

        # save image
        out_img = np.concatenate((pred_img, gt_img), axis=1)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        save_figure(exp_name, name, save_obj=out_img,
                    modality='Visualization', folder='overlay')


def pixel_downsampling(train_gt, test_gt, pix_ds_dict, label_values):
    """ Downsample a desired set, relatively to a class. The
    samples not retained can be transferred to enhance another
    specified set.
    Args:
    - train_gt (np.array): ground truth mask of training set
    - test_gt (np.array): ground truth mask of testing set
    - pix_ds_dict (dict): dictionary containing pixel downsampling infos
    - label_values (list[str]): list containing labels
    Returns:
    - train_gt (np.array): downsampled set
    - test_gt (np;array): downsampled set
    """

    def sampling(set_gt, set_tf):
        if pix_ds_dict['mode'] != 'patch':

            # Generate mask of the class of interest:
            if pix_ds_dict['class']:
                mask = set_gt == label_values.index(pix_ds_dict['class'])
            else:
                mask = set_gt != 0

            # Apply the downsampling and transfer pixels:
            process_gt = np.copy(set_gt)
            process_gt[~mask] = 0

            process_gt, temp_gt = sample_gt(process_gt,
                                                pix_ds_dict['rate'],
                                                mode=pix_ds_dict['mode'])
            set_gt[mask] = process_gt[mask]
            if pix_ds_dict['transfer']:
                set_tf[mask] = temp_gt[mask]
        else:
            patch_ind = label_values.index('FT_patch')
            class_ind = label_values.index(pix_ds_dict['class'])
            mask_patch = set_gt == patch_ind

            set_gt[mask_patch] = 0
            set_tf[mask_patch] = class_ind
            set_tf[set_tf == patch_ind] = class_ind
        return set_gt, set_tf

    if pix_ds_dict['set']:
        # Setup downsampling sets:
        if pix_ds_dict['set'] == 'train':
            train_gt, test_gt = sampling(train_gt, test_gt)
        elif pix_ds_dict['set'] == 'test':
            test_gt, train_gt = sampling(test_gt, train_gt)
    else:
        pix_ds_dict.update({'transfer': False})
        train_gt, _ = sampling(train_gt, test_gt)
        test_gt, _ = sampling(test_gt, train_gt)
    return train_gt, test_gt


def band_downsampling(img, bd_range):
    """ Remove portion of the spectrum range from the HSI cube.
    Args:
    - img (np.array): HSI hypercube
    - bd_range (list): spectrum range to keep
    Returns:
    - down_img (np.array): downsampled HSI hypercube
    """
    if bd_range[1] > img.shape[-1]:
        bd_range[1] = img.shape[-1]
    down_img = img[:, :, bd_range[0]:bd_range[1]]
    return down_img


def class_normalization(img, gt, picked_class, label_values,
                        mode='standardization'):
    """ Realize a band-wised classification computing the mean and std of selected
    pixels of a specified class.
    Args:
    - img (np.array): hypercube
    - gt (np.array): ground truth mask
    - picked_class (str): class chosen to compute mean and std
    - label_values (list[str]): list containing class names
    Kwargs:
    - mode (str): to chose between normalization ([0; 1] valutes) or
    standardization (mean=0 & std=1). Chose among:
    {'standardization', 'normalization', 'diff', 'rel_diff'}
    """
    norm_img = np.copy(img)
    class_ind = label_values.index(picked_class)
    mask = gt == class_ind
    retained_pix = img[mask]

    for band in range(retained_pix.shape[-1]):
        if mode == 'standardization':
            mean_bd = np.mean(retained_pix[:, band])
            std_bd = np.std(retained_pix[:, band])
            norm_img[:, :, band] = (img[:, :, band]-mean_bd)/std_bd
        if mode == 'normalization':
            max_bd = np.amax(retained_pix[:, band])
            min_bd = np.amin(retained_pix[:, band])
            norm_img[:, :, band] = (
                norm_img[:, :, band]-min_bd)/(max_bd-min_bd)
        if mode == 'diff':
            mean_bd = np.mean(retained_pix[:, band])
            norm_img[:, :, band] = img[:, :, band]-mean_bd
        if mode == 'rel_diff':
            mean_bd = np.mean(retained_pix[:, band])
            norm_img[:, :, band] = (img[:, :, band])/mean_bd
    return norm_img


def save_figure(exp_name, out_nm, save_obj=None, folder='', modality=''):
    """ Save the specified image, dictionary, array or the current plt.figure().
    Args:
    - exp_name (str): experiment name
    - folder (str): specific folder to build complete path
    - out_nm (str): output name
    - save_obj (np.array): object to be saved, if None, the current figure is
    saved instead
    Kwargs:
    - modality (str): plot modality,
    choose among: {'CM', 'ROC', 'Curves', 'Visualization', 'pkl_files',
    'score_map'}
    """
    out_dir = os.path.join(exp_name, 'report', modality, folder)
    os.makedirs(os.path.join(out_dir, os.path.dirname(out_nm)), exist_ok=True)

    # Save pkl file
    if ((modality == 'pkl_files' or modality == 'score_map') and
            save_obj is not None):
        out_nm = '{}.pkl'.format(out_nm)
        with open(os.path.join(out_dir, out_nm), 'wb') as pkl_out:
            os.chmod(os.path.join(out_dir, out_nm), 0o666)
            pickle.dump(save_obj, pkl_out)

    # Save image
    elif modality == 'Visualization' and save_obj is not None:
        out_nm = '{}.jpg'.format(out_nm)
        cv2.imwrite(os.path.join(out_dir, out_nm), save_obj)
        pass

    elif modality == 'scores' and save_obj is not None:
        out_nm = '{}.pkl'.format(out_nm)

    # Save plot
    else:
        out_nm = '{}.jpg'.format(out_nm)
        plt.savefig(os.path.join(out_dir, out_nm))


def load_cv_ckpt(model_name, folder, cross_dict, full_resume=False):
    """ Generate the path of the corresponding checkpoint of the current
    cross validation fold, this can be also used to resume training.
    Args:
    - model_name (str): model name
    - folder (str): folder containing ckpt
    - cross_dict (dictionary): contains cross_val infos
    - resume (bool): to load last checkpoint, if False load the best one
    Returns:
    - model_path (torch.nn): path to the ckpt to load
    """
    cv_step = cross_dict['cv_step']
    ckpt_dir = os.path.join(folder, 'k{}'.format(cv_step + 1))
    if 'SVM' in model_name:
        extension = 'pkl'
    else:
        extension = 'pth'

    if full_resume:
        ckpt_name = 'LAST_{}.{}'.format(model_name, extension)
        if not os.path.exists(os.path.join(ckpt_dir, ckpt_name)):
            ckpt_name = 'BEST_{}.{}'.format(model_name, extension)
            if not os.path.exists(os.path.join(ckpt_dir, ckpt_name)):
                return ''
    else:
        ckpt_name = 'BEST_{}.{}'.format(model_name, extension)

    return os.path.join(ckpt_dir, ckpt_name)


def rgb_generator(info_data, img=None, cross_dict=None, greyscale=True):
    """ Generate rgb images to be displayed.
    Args:
    - info_data (dict): dictionary containing data information
    - img (np.array): HSI hypercube
    - cross_dict (dict): dictionary containing cross validation information
    Returns:
    - rgb_list (list[np.array]): list containing rgb images to be displayed
    - img_name (list[str]): list containing image names to be displayed
    """
    img_name = info_data['img_name']
    rgb_list = []
    try:
        rgb_tmpl = '{0}_RGB-Image.png'
        for fold in img_name:
            name = os.path.basename(fold).split(' ')[0]
            path = os.path.join(info_data['folder'], fold,
                                rgb_tmpl.format(name))
            # rotate
            rgb_list.append(scipy.ndimage.rotate(Image.open(path), 270))
    except FileNotFoundError:
        if img is not None:
            rgb = calc_rgb(img)
            rgb_list = np.split(rgb, len(info_data['img_name']), 1)

    if cross_dict:
        img_shape = info_data['img_shape']
        cv_step = cross_dict['cv_step']
        cv_coord = cross_dict['cv_coord'][cv_step]
        fold_size = cross_dict['fold_size']
        if rgb_list:
            rgb_list_disp = []
            it_step = cv_coord[0] // img_shape[1]
            for x in range(fold_size):
                rgb_list_disp.append(rgb_list[it_step + x])
            rgb_list = rgb_list_disp
        if (cv_step + 1) * fold_size < len(img_name):
            img_name = img_name[cross_dict['cv_step'] * fold_size:
                                (1 + cross_dict['cv_step']) * fold_size]
        else:
            img_name = img_name[-fold_size:]

    if greyscale:
        for i, img in enumerate(rgb_list):
            # convert to greyscale
            rgb_list[i] = np.stack((np.dot(img, [0.2989, 0.5870, 0.1140]),)*3,
                                   axis=-1).astype(np.uint8)

    return rgb_list, img_name


def map_generator(gt, info_data, cross_dict=None):
    """
    Args:
    - gt (np.array): 2D or 3D prediction map
    - info_data (dict): dictionary containing data information
    - cross_dict (dict): dictionary containing cross validation information
    """
    def crop(coord_img):
        if gt.ndim == 2:
            return gt[:, coord_img[0]:coord_img[1]]
        return gt[:, coord_img[0]:coord_img[1], :]

    if len(info_data['img_name']) > 1:
        img_shape = info_data['img_shape']
        nb_img = len(info_data['img_name'])
        if cross_dict:
            cv_coord = cross_dict['cv_coord'][cross_dict['cv_step']]
            fold_size = cross_dict['fold_size']
            gt_list = []
            for x in range(fold_size):
                coord_img = [cv_coord[0] + img_shape[1] * x]
                coord_img.append(coord_img[-1] + img_shape[1])
                gt_list.append(crop(coord_img))
        else:
            gt_list = np.split(gt, nb_img, 1)
    else:
        gt_list = [gt]
    return gt_list


def calc_rgb(cube, minWL=500, WLsteps=5):
    """ Extract RGB data for display from HSI cube.
    Args:
    - cube (np.array): HSI cube
    - minWL (int): minimum wave length of the spectrum
    - WLsteps (int): step of the spectrum sampling
    Returns:
    - RGB_image (np.array): image to be displayed
    """
    blue_range_start = int((530 - minWL) / WLsteps)
    blue_range_end = int((560 - minWL) / WLsteps)
    green_range_start = int((540 - minWL) / WLsteps)
    green_range_end = int((590 - minWL) / WLsteps)
    red_range_start = int((585 - minWL) / WLsteps)
    red_range_end = int((725 - minWL) / WLsteps)
    factor = 1.02 * 255 * 1.5  # like in RGB-Image 1.5.vi

    RGB_image = np.zeros((cube.shape[0], cube.shape[1], 3), dtype=np.float)
    # for blue pixel take 530-560nm
    RGB_image[:, :, 2] = cube[:, :, blue_range_start:blue_range_end].mean(
        axis=2)
    # for the green pixel take 540-590nm
    RGB_image[:, :, 1] = cube[:, :, green_range_start:green_range_end].mean(
        axis=2)
    # for the red pixel take 585-725nm
    RGB_image[:, :, 0] = cube[:, :, red_range_start:red_range_end].mean(axis=2)
    # scale to 255
    RGB_image = np.clip((RGB_image * factor), 0, 255).astype(np.uint8)

    # apply gamma correction
    LUT_gamma = np.empty((1, 256), np.uint8)
    for i in range(256):
        LUT_gamma[0, i] = np.clip(pow(i / 255.0, 0.5) * 255.0, 0, 255)
    cv2.LUT(RGB_image, LUT_gamma, dst=RGB_image)
    # RGB_image = np.rot90(RGB_image, k=1, axes=(0, 1))   # rotate image

    return RGB_image


def name_mapping(info_data, exp_name, cross_dict):
    """ Generate a .txt file with the cross validation name mapping
    Args:
    - info_data (dict): ditionary containing dataset infos
    - exp_name (str): experiment name
    """
    img_name = info_data['img_name']
    fold_size = cross_dict['fold_size']
    nb_img = len(img_name)

    nmap_path = os.path.join(exp_name, 'report', 'name_mapping.txt')
    with open(nmap_path, 'w+') as f:
        for x in range(cross_dict['kfold']):
            if (x + 1) * fold_size < nb_img:
                f.write('k{} {}\n'.format(x + 1, '; '.join(
                    img_name[x * fold_size:(x + 1) * fold_size])))
            else:
                f.write('k{} {}\n'.format(
                    x + 1, '; '.join(img_name[-fold_size:])))


def save_score(gt_map, info_data, label_values, exp_name=None):
    """ Save score map in a dedicated folder
    Args:
    - gt_map (np.array): prediction map
    - info_data (dict): dictionary containing dataset information
    - exp_name (str): name of the experiment
    - label_values (list): list containing labels
    """
    img_name = info_data['img_name']
    out_tmpl = '{}_score_map'

    _, img_name = rgb_generator(info_data, cross_dict=None)
    proba_list = map_generator(gt_map, info_data, cross_dict=None)

    for i, (score_map, map_nm) in enumerate(zip(proba_list, img_name)):
        out_nm = out_tmpl.format(map_nm)
        save_figure(exp_name, out_nm, save_obj=score_map, modality='score_map')

    label_index = {i: x for i, x in enumerate(label_values)}
    save_figure(exp_name, 'label_index', save_obj=label_index,
                modality='score_map')


def snv_norm(data):
    """ Apply SNV normalization. For more information, please refer to this
    review:
        http://wiki.eigenvector.com/index.php?title=Advanced_Preprocessing:_Sample_Normalization
    Args:
    - data (np.array): HSI cube
    Returns:
    - data_norm (np.array): normalized HSI cube
    """
    number_variables = data.shape[2]
    data_reshaped = np.reshape(data, (-1, number_variables))
    mean = np.mean(data_reshaped, axis=1)
    std = np.std(data_reshaped, axis=1)
    data_norm = (data_reshaped-np.vstack(mean))/np.vstack(std)
    data_norm = data_norm.reshape(data.shape)
    return data_norm


def read_cube(path):
    """ Open the ".dat" file and load the HSI cube
    Args:
    -path (str): path to the cube.

    Returns:
    - HSI_cube (np.array): loaded hsi cube.
    """
    dt1 = np.dtype([('height', np.dtype('>i4')),
                   ('width', np.dtype('>i4')),
                   ('bands', np.dtype('>i4'))])
    meta_data = np.fromfile(path, dtype=dt1, count=1)

    shape = (int(meta_data['height']),
             int(meta_data['width']),
             int(meta_data['bands']))

    dt2 = np.dtype([('img', np.dtype('>f4'))])
    raw_data = np.fromfile(path, dtype=dt2, offset=12)
    HSI_cube = np.reshape(raw_data['img'], shape)
    return HSI_cube


def read_mask(mask_path, label_info):
    """ Read png mask.
    Args:
    - mask_path (str): path to the png image
    - label_info (list[int, str, list[str]]): list containing labels
    index, name and color
    Returns:
    - mask (np.array): ground truth mask
    """
    img1 = np.asarray(Image.open(mask_path))
    img = np.copy(img1)
    mask_1 = np.zeros(img.shape, dtype='bool')
    mask_2 =  np.zeros(img.shape, dtype='bool')
    mask_3 =  np.zeros(img.shape, dtype='bool')
    print(np.unique(img))
    mask_1[img == 0] = True
    mask_2[img != 0] = True
    #img[img == 255] =2 
    ##img[img < 100] = 0
    img[img ==0] = 1
    img[img ==100]= 3
    img[img ==255]= 2 
    print(np.unique(img))

    """
    mask = np.zeros(tuple(img.shape[:2]))

    # Background mask for bounding box:
    bg_filter = np.zeros(tuple(img.shape[:2]), dtype=int)
    bg_idx = 0

    for [index, label, color] in label_info:
        if 'background' in label.lower():
            bg_filter = ~((img == color).all(-1))
            bg_idx = index
        else:
            mask[(img == color).all(-1)] = int(index)

    bg_filter[mask != 0] = False
    mask[bg_filter] = bg_idx
    """

    return img


def saving_results_analysis(prediction,gt,info_data,cross_dict,exp_name,img):
    img_name = info_data['img_name']
    img_shape = info_data['img_shape']
    saving_dir = './report/{}/results_analysis/issue_50/'.format(exp_name)
    os.makedirs(saving_dir, exist_ok=True)
    extention_gt = '_gt.pkl'
    extention_pred = '_pred.pkl'
    extention_gt_png = '_gt.png'
    extention_pred_png = '_pred.png'

    data_list = []
    rgb_list, img_name = rgb_generator(info_data, img=img,
                                       cross_dict=cross_dict,
                                       greyscale=False)

    pred_list = map_generator(prediction, info_data, cross_dict=cross_dict)
    #proba_list = map_generator(proba, info_data, cross_dict=cross_dict)
    gt_list = map_generator(gt, info_data, cross_dict=cross_dict)
    #gt1_list = map_generator(gt1, info_data, cross_dict=cross_dict)
    """
    if cross_dict:
        cv_coord = cross_dict['cv_coord'][cross_dict['cv_step']]
        fold_size = cross_dict['fold_size']
        pred_list = []
        gt_list = []
        for x in range(fold_size):
            crop = lambda x, y: x[:, y[0]:y[1], :]  # noqa: E731
            coord_img = [cv_coord[0]+img_shape[1]*x]
            coord_img.append(coord_img[-1]+img_shape[1])
            pred_list.append(crop(prediction, coord_img))
            gt_list.append(crop(gt, coord_img))
        img_name = img_name[
                cross_dict['cv_step']*fold_size:(
                    1+cross_dict['cv_step'])*fold_size]"""
    for i, (name, pred_d, gt_d  ) in enumerate(zip(
            img_name, pred_list,gt_list)):

        print(name)
        name = name.split("/")
        #title = ["folder_class","folder_name","predicted_class","pred_average_score"]
        gt_d_label,classes,average,data_list,summ,data,class_pred,summ_ = [],[],[],[],[],[],[],[]
        total = 0
        #pred_d =  convert_from_color_(pred_d,palette)
        #gt_d = convert_from_color_(gt_d,palette)
        #gt1_d = convert_from_color_(gt1_d,palette)
        """pred_NL = np.copy(pred_d)
        mask = np.zeros(gt_d.shape, dtype='bool')
        mask[gt_d == 0] = True
        pred_d[mask] = 0"""
        """mask[pred_d == 1] = True
        pred_d[mask] =9"""
        #pred_d_ = pred_d
        """mask[gt1_d == 2] = True
        pred_d[mask] = 0
        mask[gt_d == 7] = True
        pred_d_[mask] = 0"""
        """
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        mask_for_phase1_NL = np.zeros(gt_d.shape, dtype='bool')
        mask_for_phase1_NL_1 = np.zeros(gt_d.shape, dtype='bool')
        mask_for_phase1_NL_8 = np.zeros(gt_d.shape, dtype='bool')
        proba_max = np.max(proba, axis=-1)
        print(proba_max.shape)
        mask_for_phase1_NL[gt1_d == 2] = True
        proba_max[mask_for_phase1_NL] = 0
        proba_max_region_interest = np.copy(proba_max)
        proba_max_region_interest_ = proba_max_region_interest[np.nonzero(proba_max_region_interest)]
        mean_score_segmented_region = np.mean(proba_max_region_interest_ )
        median_score_segmented_region = np.median(proba_max_region_interest_)
        proba_max_for_L = np.copy(proba_max)
        pred_NL[mask_for_phase1_NL] = 0
        pred_L = np.copy(pred_NL)
        mask_for_phase1_NL_1[pred_NL != 1]= True
        mask_for_phase1_NL_8[pred_L != 8]= True
        proba_max[mask_for_phase1_NL_1] = 0
        proba_max_for_L[mask_for_phase1_NL_8] = 0
        print(np.count_nonzero(pred_NL))
        pred_with_scores_1_control = proba_max[np.nonzero(proba_max)]
        pred_with_scores_8_ischemic = proba_max_for_L[np.nonzero(proba_max_for_L)]
        mean_score_for_control = np.mean(pred_with_scores_1_control)
        median_score_for_control = np.median(pred_with_scores_1_control)
        mean_score_for_ischemic = np.mean(pred_with_scores_8_ischemic)
        median_score_for_ischemic = np.median(pred_with_scores_8_ischemic)
        print(mean_score_for_control)
        print(median_score_for_control)
        print(mean_score_for_ischemic)
        print(median_score_for_ischemic)
        print(np.count_nonzero(pred_with_scores_1_control))
        print(np.count_nonzero(pred_with_scores_8_ischemic))
        print(np.unique(pred_d))
        classes = np.unique(pred_d)
        classes_ = np.unique(pred_d_)
        gt_d_label =np.unique(gt_d)
        gt_d_label_ =np.unique(gt1_d)
        print(gt_d_label)
        #print(np.unique(gt1_d))
        classes = classes"""

        path_p = os.path.join(saving_dir,name[0],name[1])
        os.makedirs(path_p, exist_ok=True)
        print(path_p)
        outfile = open(os.path.join(path_p,name[1]+extention_pred),'wb')

        pickle.dump(pred_d,outfile)
        outfile.close()
        outfile_ = open(os.path.join(path_p,name[1]+extention_gt),'wb')

        pickle.dump(gt_d,outfile_)
        outfile_.close()

        """
        # PNG SAVING
        outfile_png = open(os.path.join(path_p,name[1]+extention_gt_png),'wb')
        rescaled = (255.0 / gt_d.max() * (gt_d - gt_d.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        img.save(outfile_png,"PNG" )"""

        """
        print(gt_d_label_)
        data.append(class_names[int(gt_d_label[1:])])
        data.append(name)
        for x in classes:
            summ.append(np.count_nonzero(pred_d == x))
            class_pred.append(class_names[int(x)])
            """
        """
        for x in classes_:
            summ_.append(np.count_nonzero(pred_d_ == x))
            #class_pred.append(class_names[int(x)])"""

        """
        #data.append(class_pred[1:])
        class_pred_ = ['t_0','t_30-60-90']
        #data.append(class_pred[1:])
        data.append(mean_score_for_control)
        data.append(median_score_for_control)
        data.append(mean_score_for_ischemic)
        data.append(median_score_for_ischemic)
        data.append(np.count_nonzero(pred_with_scores_1_control))
        data.append(np.count_nonzero(pred_with_scores_8_ischemic))
        data.append(mean_score_segmented_region)
        data.append(median_score_segmented_region)
        total = sum(summ[1:])
        #total_ = sum(summ_[1:])
        average = [round(x / int(total),2)for x in summ]
        #_average = [round(x / int(total_),2)for x in summ_]
        average_ = [0.0,0.0]"""
        """
        if len(class_pred[1:]) < 2:
            if "t_0" in class_pred[1:]:
                average_[0] = average[1]
            else :
                average_[1] = average[1]
            data.append(average_)
        else:
            data.append(average[1:])
            #data.append(_average[1:])"""



        #data.append(summ)
        #data_list.append(data)
        with open(os.path.join(saving_dir,'issue_50.csv'), 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_list)
