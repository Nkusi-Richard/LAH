# -*- coding: utf-8 -*-
import random
from PIL import Image as im
from numpy import asarray
from scipy import misc,ndimage
import numpy as np
import csv
from sklearn.metrics import confusion_matrix, roc_curve, auc
#  , precision_score, recall_score, accuracy_score
import sklearn.model_selection
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import mean_absolute_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from PIL import Image
# from sklearn.preprocessing import label_binarize

import cv2
import itertools
import spectral
# import visdom
import matplotlib.pyplot as plt
from mpltools import special
from scipy import io, misc
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from plot_matconf import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#from sklearn.metrics import plot_confusion_matrix
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
        return misc.imread(dataset)
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
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def display_predictions(pred, vis, info_data, gt=None, caption=""):
    img_name = info_data['img_name']
    [h, w] = info_data['img_shape']
    nb_img = len(img_name)
    pred__ = np.copy(pred)
    if pred.shape[1] > w+1:
        white = np.ones((h, 2 ))*255
        for x in range(1, nb_img):
            pred__[:, x*w:x*w+2] = white
    if gt is None:
        vis.images([pred__],
                   opts={'caption': caption})
    else:
        vis.images([pred,
                    gt],
                   nrow=2,
                   opts={'caption': caption,
                         'jpgquality': 50})


#def display_dataset(img, gt, bandse, vis,
#                    data_conf):
    """Display the specified dataset.

    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        display (optional): type of display, if any

    """
    """
    if len(data_conf['info_data']['img_name']) > 1:
        rgb = concat_rgb(data_conf['info_data'])
        caption = 'RGB'
    else:
        rgb = spectral.get_rgb(img, bands)
        rgb /= np.max(rgb)
        rgb = np.asarray(255 * rgb, dtype='uint8')

        caption = "RGB (bands {}, {}, {})".format(*bands)

    # send to visdom server
    vis.images([rgb],
               opts={'caption': caption})"""

def display_dataset(img, vis, info_data):
    """Display the specified hypercube.
    Args:
    - img: 3D hyperspectral image
    - vis (Visdom.visdom): visdom display
    - info_data (dict): dictionary containing dataset info
    """
    rgb_list, _ = rgb_generator_reg(info_data, img=img, greyscale=False)
    rgb = np.concatenate(tuple(rgb_list), axis=1)
    print("DATASET")
    print(rgb.shape)

    # send to visdom server
    vis.images([rgb],
               opts={'caption': 'RGB representation'})
    #save_figure( "Okay", "A", save_obj=rgb, folder='Err',
    #                modality='Visualization')


def explore_spectrums(img, complete_gt, class_names,palette, vis,
                      dataset_conf, ignored_labels=None,
                      plot_name='', exp_name='',
                      plot_scale=[0, 1]):
    """Plot sampled spectrums with mean + std for each class
    and the histogram of pixels values.

    Args:
    - img (np.array): 3D hyperspectral image
    - complete_gt (np.array): 2D array of labels
    - class_names (list[str]): list of class names
    - ignored_labels (list[int]): list of labels to ignore
    - vis (visdom.Visdom) : Visdom display
    - dataset_conf (dictionary): contains dataset information
    - plot_name (str): name of the plot to be saved
    - exp_name (str): experiment name
    - plot_scale (list[int]): y axis limits
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}
    csfont = {'fontname':'Arial'}

    # Configure x_scale:
    start = 0
    stop = img.shape[-1]
    if 'band_downsampling' in dataset_conf.keys():
        [start, stop] = dataset_conf['band_downsampling']
    bd_range = [500+start*5, 1000-(100-stop)*5]
    x_scale = np.arange(bd_range[0], bd_range[1], 5)

    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        color = [x/255.0 for x in palette[class_names.index(class_names[int(c)])]]
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        plt.figure()
        ax = plt.gca()
        ax.set_ylim(plot_scale[0], plot_scale[1])
        plt.title('{}: {}'.format(plot_name, class_names[int(c)]),fontsize=12,**csfont)
        # ax.set_ylim(0, 12)
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(x_scale, spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)

        # Plot the mean spectrum with thickness based on std
        #special.errorfill(x_scale, mean_spectrum, std_spectrum, label='mean',
        #                  label_fill='std')
        special.errorfill(x_scale, mean_spectrum, std_spectrum, label='mean',
                          color="#3F5D7D", alpha_fill=0.6)

        #plt.fill_between(x_scale, mean_spectrum- std_spectrum,mean_spectrum + std_spectrum)
        #plt.legend(label)
        #plt.errorbar(x_scale, mean_spectrum, yerr =std_spectrum )
        plt.xlabel('Wavelength (nm)', fontsize=12,**csfont)
        plt.ylabel('Relative Reflectance', fontsize=12,**csfont)
        if 'overall' in plot_name:
            vis.matplot(plt)
        mean_spectrums[class_names[int(c)]] = [mean_spectrum, std_spectrum]

        # save curve figures
        plot_dir = './report/{}/Curves/{}'.format(exp_name, plot_name)
        plot_nm = '{}.jpg'.format(class_names[int(c)])
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, plot_nm))

    mask = np.invert(np.isin(complete_gt, ignored_labels))
    # display_histo(img[mask], log=True)

    return mean_spectrums


def plot_spectrums(spectrums, vis, dataset_conf, label_values, exp_name,
                   palette, retained_class=['Nerve', 'Muscle'],
                   title="", plot_scale=[0, 1], plot_name=''):
    """Plot the specified dictionary of spectrums.

    Args:
    - spectrums (dict): dictionary (name -> spectrum) of spectrums to plot
    - vis (Visdom.visdom): Visdom display
    - dataset_conf (dictionary): contains dataset information
    - exp_name (str): experiment name
    - plot_scale (list[int]): y axis limits
    - plot_name (str): name of the plot to be saved
    """
    win = None
    fig = {}
    csfont = {'fontname':'Arial'}
    # Configure x_scale:
    start = 0
    stop = len(next(iter(spectrums.values()))[0])

    if 'band_downsampling' in dataset_conf.keys():
        [start, stop] = dataset_conf['band_downsampling']
    bd_range = [500+start*5, 1000-(100-stop)*5]
    x_scale = np.arange(bd_range[0], bd_range[1], 5)

    for x in range(2):
        fig[x] = plt.figure()
        ax = plt.gca()
        ax.set_ylim(plot_scale[0], plot_scale[1])
        plt.title('{}: mean spectral curves'.format(title),fontsize=12,**csfont)

        plt.xlabel('Wavelength (nm)',fontsize=12,**csfont)
        plt.ylabel('Relative Reflectance',fontsize=12,**csfont)

    for label, [mean_spec, std_spec] in spectrums.items():
        color = [x/255.0 for x in palette[label_values.index(label)]]

        # Sample and plot spectrums from the selected class
        plt.figure(fig[0].number)
        plt.plot(x_scale, mean_spec, color=color, label=label)
        #plt.plot(x_scale, std_spec, color=color, label=label)
        plt.figure(fig[1].number)
        #if not(label in retained_class):
        #    std_spec = 0
        #special.errorfill(x_scale, mean_spec, std_spec, label=label
        #                   )
        special.errorfill(x_scale, mean_spec, std_spec, label=label,
                          color=color, alpha_fill=0.5)

        #plt.fill_between(x_scale, mean_spec- std_spec,mean_spec+ std_spec)
        #plt.legend(label)
        #plt.errorbar(x_scale, mean_spec, yerr = std_spec)

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
        plot_dir = './report/{}/Curves/{}'.format(exp_name, plot_name)
        plot_nm = 'All_Classes{}.jpg'.format(preff)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, plot_nm))


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


def metrics(prediction, target, exp_name, ignored_labels=[], n_classes=None,
            labels=None, cm_id=''):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    print(np.count_nonzero(target))
    print(np.count_nonzero(prediction))
    print(target.shape)
    print(prediction.shape)
    print(ignored_labels)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]
    print(np.count_nonzero(target))
    print(np.count_nonzero(prediction))
    print("*****************************************")
    print(np.shape(prediction))
    print(np.shape(target))
    results = {}
    test_loss = mean_absolute_error(prediction,target)
    print("here")
    print(test_loss)
    #n_classes = np.max(target) + 1 if n_classes is None else n_classes
    #labels = ['business', 'health']
    #cm = confusion_matrix(
    #    target,
    #    prediction,
    #    labels=range(n_classes))

    report_dir = './report/{}/pkl_files'.format(exp_name)
    os.makedirs(report_dir, exist_ok=True)
    #pickle.dump(cm, open(report_dir+'/CM_k'+cm_id+'.pkl', 'wb'))
    # plot_confusion_reportrix(cm, labels, title=model)
    """

    # remove NaN lines and plot cm
    cm_disp = np.delete(cm, ignored_labels, axis=0)
    cm_disp = np.delete(cm_disp, ignored_labels, axis=1)
    labels_disp = np.delete(np.array(labels), ignored_labels)
    #plot_confusion_matrix(cm_disp, labels_disp, normalize=True,
    #                      title=exp_name+' cm_'+cm_id)

    cm_  = cm_disp.astype('float') / cm_disp.sum(axis=1)[:, np.newaxis]
    cm_ =  np.round(cm_, decimals=3)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_,
                              display_labels=labels_disp)
    """

        # NOTE: Fill all variables here with default values of the plot_confusion_matrix
    #disp.plot(include_values=True,
    #             cmap=plt.cm.Blues, ax=None, xticks_rotation='horizontal')
    """ fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()"""
    # Save confusion matrix figure
    """
    plot_dir = './report/{}/CM/'.format(exp_name)
    os.makedirs(plot_dir, exist_ok=True)
    plot_nm = 'k{}.jpg'.format(cm_id)
    plt.savefig(os.path.join(plot_dir, plot_nm))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy
    # results["sk_accuracy"] = accuracy_score(target, prediction)

    # results["sk_precision"] = precision_score(target, prediction)
    # results["sk_recall"]  = recall_score(target, prediction)

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores
   

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa
    """

    # Compute Precision and Recall
    # true_pos = np.diag(cm)
    # precision = np.sum(true_pos / np.sum(cm, axis=0))
    # recall = np.sum(true_pos / np.sum(cm, axis=1))

    # results["Precision"] = precision
    # results["Recall"] = recall

    return results


def show_results(results, vis, exp_name, ignored_labels, label_values=None,
                 agregated=False):
    text = "\n Results:\n"

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1 scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm_sum = np.sum([r["Confusion matrix"] for r in results], axis=0)

        # remove NaN lines and plot cm
        cm_disp = np.delete(cm_sum, ignored_labels, axis=0)
        cm_disp = np.delete(cm_disp, ignored_labels, axis=1)
        labels_disp = np.delete(np.array(label_values), ignored_labels)
        #plot_confusion_matrix(cm_disp, labels_disp, normalize=True,
        #                      title='{} cm_overall'.format(exp_name))
        cm_  = cm_disp.astype('float') / cm_disp.sum(axis=1)[:, np.newaxis]
        cm_ =  np.round(cm_, decimals=3)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm_,
                              display_labels=labels_disp)


        # NOTE: Fill all variables here with default values of the plot_confusion_matrix
        disp.plot(include_values=True,
                 cmap=plt.cm.Blues, ax=None, xticks_rotation='horizontal')
        # Save confusion matrix figure
        plot_dir = './report/{}/CM/'.format(exp_name)
        os.makedirs(plot_dir, exist_ok=True)
        plot_nm = 'overall.jpg'
        plt.savefig(os.path.join(plot_dir, plot_nm))

        # cm_mean = np.mean([r["Confusion matrix"] for r in results], axis=0)
        # plot_confusion_matrix(cm_mean, label_values, normalize=True,
        #                       title='cm_mean')
        text += "Agregated results :\n"

        cm = cm_sum
        pickle.dump(cm, open('./report/{}/pkl_files/CM_CrossVal.pkl'.format(
            exp_name), 'wb'))
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1 scores"]
        kappa = results["Kappa"]

    # vis.heatmap(cm, opts={'title': "Confusion matrix",
    #                       'marginbottom': 150,
    #                       'marginleft': 150,
    #                       'width': 500,
    #                       'height': 500,
    #                       'rownames': label_values,
    #                       'columnnames': label_values})
    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.03f} +- {:.03f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.03f}%\n".format(accuracy)
    text += "---\n"

    text += "F1 scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.03f} +- {:.03f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.03f}\n".format(kappa)

    # vis.text(text.replace('\n', '<br/>'))
    print(text)


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
        #train_indices, test_indices = sklearn.model_selection.train_test_split(
        #    X, train_size=train_size, stratify=y)
        train_indices, test_indices = sklearn.model_selection.train_test_split(
            X, train_size=train_size)

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
        print(cv_coord)
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


def data_stat(gt, label_values):
    stat = np.zeros((len(label_values,)))
    for c in np.unique(gt):
        stat[int(c)] = np.count_nonzero(gt == c)
    summ = np.sum(stat[1:])

    # TODO: check ZeroDivisionError
    print('Classes occurences (training set):')
    for lbl, val in zip(label_values[1:], stat[1:]):
        print("{} {} {}%".format(
            lbl, int(val), round(val/summ*100, 2)))


def display_legend(classes, vis, palette, final_shape, exp_name):
    fig = plt.figure()
    for i, label in enumerate(classes[1:]):
        square = np.ones((5, 5))*(i+1)
        color_square = convert_to_color_(square, palette=palette)
        ax = plt.subplot2grid((len(classes[1:]), 1), (i, 0))
        ax.yaxis.set_label_position("right")
        plt.imshow(color_square)
        y_lbl = plt.ylabel(label, labelpad=30)
        y_lbl.set_rotation(0)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        ax.set_yticklabels([])
        ax.set_xticklabels([])

    # save curve figures
    plot_dir = './report/{}/Visualization'.format(exp_name)
    plot_nm = 'legend.jpg'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, plot_nm))

    # save as image
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to
    # have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    w, h, d = buf.shape
    # image = np.array(
    #     Image.frombytes("RGBA", (w, h), buf.tostring()).resize(final_shape))
    # fig.close()
    # return image

def rgb_generator_reg(info_data, img=None, cross_dict=None, greyscale=True):
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

            """
            name = os.path.basename(fold).split(' ')[0]
            path = os.path.join(info_data['folder'], fold,
                               'reg' )
            # rotate
            list_png = [f for f in os.listdir(path) if f.endswith(
                    '.png')]
            path = os.path.join(path, list_png[0])
            #save_figure("OKay", fold, save_obj=scipy.ndimage.rotate((Image.open(path)), 270), folder='Err',
            #        modality='Visualization')
            rgb_list.append(scipy.ndimage.rotate((Image.open(path)), 270))"""
    except FileNotFoundError:
        if img is not None:
            rgb = calc_rgb(img)
            rgb_list = np.split(rgb, len(info_data['img_name']), 1)

    if cross_dict:
        cv_step = cross_dict['cv_step']
        folds_size = cross_dict['folds_size']

        end_steps = list(np.cumsum(folds_size))
        start_steps = [0] + end_steps[:-1]

        if end_steps[cv_step] < len(img_name):
            img_name = img_name[start_steps[cv_step]:end_steps[cv_step]]
            if rgb_list:
                rgb_list = rgb_list[start_steps[cv_step]:end_steps[cv_step]]
        else:
            img_name = img_name[-folds_size[-1]:]
            if rgb_list:
                rgb_list = rgb_list[-folds_size[-1]:]

    """
    if greyscale:
        for i, img in enumerate(rgb_list):
            # convert to greyscale
            rgb_list[i] = np.stack((np.dot(img, [0.2989, 0.5870, 0.1140]),)*3,
                                   axis=-1).astype(np.uint8)"""

    return rgb_list, img_name

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



def map_generator_reg(gt, info_data, cross_dict=None):
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

    nb_img = len(info_data['img_name'])
    if nb_img > 1:
        gt_list = np.split(gt, nb_img, 1)
        print(gt_list)
        if cross_dict:
            cv_coord = cross_dict['cv_coord']
            cv_step = cross_dict['cv_step']
            folds_size = cross_dict['folds_size']

            gt_crop = crop(cv_coord[cv_step])
            gt_list = np.split(gt_crop, folds_size[cv_step], 1)
    else:
        gt_list = [gt]

    return gt_list


def display_error_reg(img, pred, err, vis, info_data,
                  gt, exp_name='',
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
    rgb_list, img_name = rgb_generator_reg(info_data, img=img,
                                       cross_dict=cross_dict,
                                       greyscale=False)
    grey_list, _ = rgb_generator_reg(info_data, img=img,
                                 cross_dict=cross_dict)
    pred_list = map_generator_reg(pred, info_data, cross_dict=cross_dict)
    gt_list = map_generator_reg(gt, info_data, cross_dict=cross_dict)
    err_list = map_generator_reg(err, info_data, cross_dict=cross_dict)

    for i, (name, grey_d, rgb_d, pred_d, err_d, gt_d) in enumerate(zip(
            img_name, grey_list, rgb_list, pred_list, err_list, gt_list)):
        #white = np.ones((err_d.shape[0])) * 255
        #pred_d[:, -1] = white
        #gt_d[:, -1] = white
        #rgb_d[:, -1] = white
        #err_d[:, -1] = white
        print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        print(name)
        # display on visdom
        rg = np.concatenate([gt_d,pred_d,err_d],axis=1)
        vis.images([#np.transpose(rgb_d),
                    #np.transpose(gt_d),
                    #np.transpose(pred_d),
                    #np.transpose(err_d)
                    rg
                    ],
                   # np.transpose(legend, (2, 0, 1))],
                   nrow=2,
                   opts={'caption': name})

        # save image
        mask = gt_d == 0
        print(mask.shape)
        ma = np.nonzero(gt_d)
        print(ma)
        xcropp = ma[0]
        ycropp = ma[1]
        ymin, ymax = int(ycropp.min()), int(np.ceil(ycropp.max()))
        xmin, xmax = int(xcropp.min()), int(np.ceil(xcropp.max()))
        pred_mask = np.copy(pred_d)
        pred_mask[mask] = 0
        #pred_mask[:, -1] = white

        alpha = 1
        print("INTEREST")
        print(grey_d.shape)
        print(mask.shape)
        pred_d = (pred_d /110)*200
        gt_d = (gt_d /110)*200
        #pred_d = im.fromarray(pred_d)
        #pred_d = asarray(pred_d.convert('RGB'))
        #gt_d = im.fromarray(gt_d)
        #gt_d = asarray(gt_d.convert('RGB'))
        pred_d = cv2.applyColorMap(pred_d.astype(np.uint8), cv2.COLORMAP_JET)
        gt_d= cv2.applyColorMap(gt_d.astype(np.uint8), cv2.COLORMAP_JET)
        rgb_d = cv2.cvtColor(rgb_d, cv2.COLOR_BGR2RGB)
        pred_img = overlayed_img(rgb_d, pred_d, ~mask, alpha=alpha)
        gt_img = overlayed_img(rgb_d, gt_d, ~mask, alpha=alpha)
        print("XMIN, XMAX, YMIN, YMAX")
        print(xmin-5, xmax+5, ymin-5, ymax+5)
        rgb_d = rgb_d[xmin-5:xmax+5, ymin-5:ymax+5]
        gt_img = gt_img[xmin-5:xmax+5, ymin-5:ymax+5]
        pred_img = pred_img[xmin-5:xmax+5, ymin-5:ymax+5]
        print(type(pred_d))
        #if ARGS.regression:
        #gt_img_= (cv2.applyColorMap(((gt_img/np.max(gt_img))).astype(np.uint8)*255, cv2.COLORMAP_JET))
        #gt_img_ =np.uint8(plt.cm.jet(gt_img/np.max(gt_img))*255)
        #pred_img_ = (cv2.applyColorMap(((pred_img/np.max(pred_img))).astype(np.uint8)*255, cv2.COLORMAP_JET))
        #pred_img_ =np.uint8(plt.cm.jet(pred_img/np.max(pred_img))*255)
        ##  err_d_ = cv2.applyColorMap(((err_d/110)*255).astype(np.uint8), cv2.COLORMAP_JET)
        #err_d_ = (cv2.applyColorMap(((err_d/np.max(err_d))).astype(np.uint8)*255, cv2.COLORMAP_JET))
        #err_d_=np.uint8(plt.cm.jet(err_d/np.max(err_d))*255)
        ##   pred_d_ = cv2.applyColorMap(((pred_d/110)*255).astype(np.uint8), cv2.COLORMAP_JET)
        #pred_d_ = (cv2.applyColorMap(((pred_d/np.max(pred_d))).astype(np.uint8)*255, cv2.COLORMAP_JET))
        #pred_d_ =np.uint8(plt.cm.jet(pred_d/np.max(pred_d))*255)

        zoom_factor = 10
        out_img = np.concatenate((ndimage.zoom(rgb_d, (zoom_factor,) * 2 + (1,) * (rgb_d.ndim - 2)), ndimage.zoom(gt_img, (zoom_factor,) * 2 + (1,) * (gt_img.ndim - 2)),ndimage.zoom(pred_img, (zoom_factor,) * 2 + (1,) * (pred_img.ndim - 2))),
                                 axis=1)
        #out_img = np.concatenate((ndimage.zoom(rgb_d, (zoom_factor,) * 2 + (1,) * (rgb_d.ndim - 2)), ndimage.zoom(gt_img, (zoom_factor,) * 2 + (1,) * (gt_img.ndim - 2))),
        #                         axis=1)
        #print(gt_img.shape)
        #out_img = np.concatenate((gt_img,pred_img),
        #                         axis=1)
        print("Arrayyyyyyyyyyyyyyyyyyyyyyyyyy")
        #print(gt_img_.shape)
        #print(np.max(gt_img/np.max(gt_img)))
        #print(np.max(err_d/np.max(err_d)))
        #else:
        #    out_img = np.concatenate((gt_img, pred_img, pred_d_),
        #                         axis=1)

        #out_img = np.concatenate((pred_d_),
        #                         axis=1)

        #out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        save_figure(exp_name, name, save_obj=out_img, folder='Err',
                    modality='Visualization')
        #save_figure(exp_name, name, save_obj=pred_d_, folder='Err',
        #            modality='Visualization')

def save_figure(exp_name, out_nm, save_obj=None, folder='', modality=''):
    """ Save the specified image, dictionary or array or the current plt.figure().
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
    out_dir = os.path.join('report', exp_name, modality, folder)
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


def display_error(img, pred, err,gt_, vis, bands, info_data,
                  classes, palette, gt, exp_name='',
                  cross_dict=None):
    # lgd_shape = tuple(list(np.shape(gt_list[0]))[:2])[::-1]
    # legend = display_legend(classes, vis, palette, lgd_shape)[:, :, :3]

    img_nm = 'err_visu'
    if cross_dict:
        img_nm = 'k{}'.format(1+cross_dict['cv_step'])

    rgb_list, img_name = rgb_generator(img, bands, info_data,
                                       cross_dict=cross_dict)
    pred_list = map_generator(pred, info_data, cross_dict=cross_dict)
    gt_list = map_generator(gt, info_data, cross_dict=cross_dict)
    err_list = map_generator(err, info_data, cross_dict=cross_dict)
    gt__list = map_generator(gt_, info_data, cross_dict=cross_dict)

    suffx = (len(img_name) > 1)
    for i, (name, rgb_d, pred_d, err_d, gt_d,gt_d_) in enumerate(zip(
            img_name, rgb_list, pred_list, err_list, gt_list,gt__list)):
        """ cHANGING THE LABEL COLOR OF THE ANNOTATIONS"""

        print("OKAY")
        #gt_d = convert_from_color_(gt_d,palette)
        print(np.unique(gt_d))
        original_label = np.unique(gt_d_)[1]
        print(original_label)
        mask_ =  np.zeros(gt_d.shape, dtype='bool')
        mask_[gt_d == 8] = True
        gt_d[mask_]= int(original_label)
        gt_d = convert_to_color_(gt_d,palette)

        white = np.ones((err_d.shape[0], 3))*255
        pred_d[:, -1, :] = white
        gt_d[:, -1, :] = white
        rgb_d[:, -1, :] = white
        err_d[:, -1, :] = white
        maski = np.zeros(gt_d.shape, dtype='bool')
        pred_di =  np.copy(pred_d) 
        maski[gt_d == 0] = True
        pred_di[maski] = 0
        """ cHANGING THE LABEL COLOR OF THE ANNOTATIONS 

        print("OKAY")
        gt_d = convert_from_color_(gt_d,palette)
        print(np.unique(gt_d))
        original_label = np.unique(gt_d_)[1]
        print(original_label)        
        mask_ =  np.zeros(gt_d.shape, dtype='bool')
        mask_[gt_d == 8] = True
        gt_d[mask_]= int(original_label)
        gt_d = convert_to_color_(gt_d,palette)"""


        # display on visdom
        vis.images([np.transpose(pred_d, (2, 0, 1)),
                    np.transpose(gt_d, (2, 0, 1)),
                    np.transpose(err_d, (2, 0, 1)),
                    np.transpose(rgb_d, (2, 0, 1))],
                   # np.transpose(legend, (2, 0, 1))],
                   nrow=4,
                   opts={'caption': name})

        # save image
        mask = gt_d == 0
        pred_mask = np.copy(pred_d)
        pred_mask[mask] = 0
        pred_mask[:, -1, :] = white

        alpha = 1
        pred_img = overlayed_img(rgb_d, pred_d, ~mask, alpha=alpha)
        gt_img = overlayed_img(rgb_d, gt_d, ~mask, alpha=alpha)
        out_img = np.concatenate((rgb_d, gt_img, pred_img,   pred_di, pred_d),
                                 axis=1)

        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        img_dir = './report/{}/Visualization/Err'.format(exp_name)

        if suffx:
            out_nm = '{}_({}_{})'.format(name,img_nm, str(i))
        else:
            out_nm = img_nm
        os.makedirs(img_dir, exist_ok=True)
        cv2.imwrite(os.path.join(img_dir, '{}.jpg'.format(out_nm)), out_img)


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
    plt.show()


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
    rgb = np.zeros(tuple(info_data['cube_shape']))
    for i, acquisition in enumerate(info_data['img_name']):
        print(acquisition)
        img_name = os.path.basename(acquisition).split(' ')[0]
        print(img_name)
        path = os.path.join(info_data['folder'], acquisition
                            )
        list_png = [f for f in os.listdir(path) if f.endswith(
                    '.dat')]
        path = os.path.join(path, list_png[0])

        print(path)
        img = Image.open(path)
        # rotate
        rgb[:, i*width:(i+1)*width] = scipy.ndimage.rotate(img, 270)
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
    img = (img - np.mean(img))/np.std(img)
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
        exp_var, 100*np.sum(exp_var)))

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


def class_fusion(fused_class, gt, ignored_labels, label_values,
                 class_name='Fused'):
    """ Function to fuse classes for hierarchy classification.
    Please pay attention that if on class inside fused_class is ignored, all
    other fused classes will be ignored also. Moreover, to keep a pleasant
    display of the prediction, please do not includ unclassified class in
    the fused classes.
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
    gt[gt == -1] = len(label_values)-1

    ignored_labels = list(ignored_labels)

    if fused_ignored:
        ignored_labels.append(len(label_values)-1)
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

    def __init__(self, weight=None, alpha=1, gamma=2,  reduction='mean'):
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
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss

        if (self.reduction != 'none'):
            return torch.mean(F_loss)
        else:
            return F_loss


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
    {'standardization', 'normalization'}
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
            norm_img[:, :, band] = abs(img[:, :, band]-mean_bd)
        if mode == 'rel_diff':
            mean_bd = np.mean(retained_pix[:, band])
            norm_img[:, :, band] = (img[:, :, band])/mean_bd
    return norm_img


def ROC_curve(probabilities, gt, ignored_labels, label_values, exp_name,
              palette, plot_name='', verbose=True, compact=True):
    """ Compute and possibly plot the ROC curve for each class
    Args:
    - probabilities (np.array): class confidence scores of each pixel
    - gt (np.array): ground truth mask
    - ignored_labels (list(int)): list containing label indexes to be ignored
    - label_values (list(str)): list containing class names
    - exp_name (str): name of the experiment
    - palette (dict): dicitonary of colors
    - plot_name (str): name of the graph plotted
    - verbose (bool): to chose to plot the curves or not
    - compact (bool): to chose to have a reduced display
    """
    report_dir = './report/'+exp_name

    ignored_mask = np.zeros(gt.shape[:2], dtype=np.bool)
    for l in ignored_labels:
        ignored_mask[gt == l] = True
    ignored_mask = ~ignored_mask
    y_true = gt[ignored_mask]
    y_score = probabilities[ignored_mask]

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

        if not compact:
            color = [x/255.0 for x in palette[class_id]]

            lw = 2
            plt.figure()
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve (area = {0:0.2f})'.format(roc_auc[i]))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('{} {}: {}'.format(exp_name, plot_name,
                                         label_values[class_id]))
            plt.legend(loc="lower right")

            # save ROC curve figure
            plot_dir = report_dir+'/ROC/indiv_ROC/'+plot_name
            plot_nm = '{}_{}.jpg'.format(plot_name, label_values[class_id])
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, plot_nm))

    if verbose:
        lw = 2
        plt.figure()
        for i, class_id in enumerate(np.unique(y_true)):
            color = [x/255.0 for x in palette[class_id]]
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(label_values[int(class_id)], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(exp_name+' '+plot_name)
        plt.legend(loc="lower right")

        # save ROC curve figure
        plot_dir = report_dir+'/ROC/indiv_ROC/'+plot_name
        plot_nm = '{}_classes.jpg'.format(plot_name)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, plot_nm))

    report_dir += '/pkl_files'
    os.makedirs(report_dir, exist_ok=True)
    pickle.dump([fpr, tpr, thresh, roc_auc, np.unique(y_true), palette],
                open(report_dir+'/ROC.pkl', 'wb'))

    return [fpr, tpr, thresh, class_ind]


def ROC_combined(ROC_info, exp_name):
    """ Plot the patient-specific ROC curves on the same plot for each class
    Args:
    - ROC_info (dict): dictionary containing the patient-specific fpr and tpr
    - exp_name (str): name of the experiment
    """
    lw = 2

    unique_class = dict()
    for _, [_, _, _, class_ind] in ROC_info.items():
        unique_class.update(class_ind)

    fig = dict()
    for ind, label in unique_class.items():
        fig[ind] = plt.figure()
        plt.title('{}, patient-specific ROC curves: {}'.format(
            exp_name, label))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    for index, [fpr, tpr, _, class_ind] in ROC_info.items():
        for ind in class_ind.keys():
            roc_auc = auc(fpr[ind], tpr[ind])
            plt.figure(fig[ind].number)
            plt.plot(fpr[ind], tpr[ind], lw=lw,
                     label='k{0} ROC curve (area = {1:0.2f})'.format(
                         index, roc_auc))
            plt.legend(loc="lower right")

    plot_dir = './report/{}/ROC/'.format(exp_name)
    os.makedirs(plot_dir, exist_ok=True)

    for ind, label in unique_class.items():
        plt.figure(fig[ind].number)
        plot_nm = '{}_indiv.jpg'.format(label)
        plt.savefig(os.path.join(plot_dir, plot_nm))

    report_dir = './report/{}/pkl_files'.format(exp_name)
    os.makedirs(report_dir, exist_ok=True)
    pickle.dump(ROC_info, open(report_dir+'/ROC_combined.pkl', 'wb'))


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
            print(np.unique(temp_gt))
        return set_gt, set_tf

    if pix_ds_dict['set']:
        # Setup downsampling sets:
        if pix_ds_dict['set'] == 'train':
            train_gt, test_gt = sampling(train_gt, test_gt)
            return train_gt, test_gt

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
    print("MASK")
    print(overlay.shape)
    print(mask.shape)
    print(color_pred.shape)
    overlay[mask] = color_pred[mask]

    # Aplly overlay:
    overlay_img = cv2.addWeighted(overlay, alpha, rgb_img, 1-alpha, 0)

    return overlay_img


def display_overlay(img, pred, gt, gt_mask,  pred_mask, vis, bands,
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
    - bands (list[int]): list containing indexes of spectral bands
    used to build the rgb image dislayed
    - info_data (dict): dictionary containing data information
    - palette (dict): dictionary of colors
    - cross_dict (dict): dictionary containing cross validation information
    - alpha (float): alpha coefficient for transparency overlay
    """
    img_nm = 'overlay_visu'
    if cross_dict:
        img_nm = 'k{}'.format(1+cross_dict['cv_step'])

    rgb_list, img_name = rgb_generator(img, bands, info_data,
                                       cross_dict=cross_dict)
    pred_list = map_generator(pred, info_data, cross_dict=cross_dict)
    gt_list = map_generator(gt, info_data, cross_dict=cross_dict)
    pred_mask_list = map_generator(pred_mask, info_data, cross_dict=cross_dict)
    gt_mask_list = map_generator(gt_mask, info_data, cross_dict=cross_dict)

    suffx = (len(img_name) > 1)
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
        img_dir = './report/{}/Visualization/Overlay'.format(exp_name)

        if suffx:
            out_nm = '{}_({}_{})'.format(name,img_nm, str(i))
        else:
            out_nm = img_nm
        os.makedirs(img_dir, exist_ok=True)
        cv2.imwrite(os.path.join(img_dir, '{}.jpg'.format(out_nm)), out_img)


#def snv_norm(data):
    """ Apply SNV normalization. For more information, please refer to this
    review:
        http://wiki.eigenvector.com/index.php?title=Advanced_Preprocessing:_Sample_Normalization
    """
    """
    number_variables = data.shape[2]
    data_reshaped = np.reshape(data, (-1, number_variables))
    mean = np.mean(data_reshaped, axis=1)
    weight = np.sqrt(np.sum(np.square(
        data_reshaped-np.vstack(mean)), axis=1)/(number_variables-1))
    data_norm = np.multiply(data_reshaped, np.reciprocal(
        weight)[:, np.newaxis])
    data_norm = data_norm.reshape(data.shape)
    return data_norm
    """


def load_cv_ckpt(model_name, folder, cross_dict):
    """ Load the corresponding checkpoint of the current
    cross validation fold.
    Args:
    - model_name (str): model name
    - folder (str): folder containing ckpt
    - cross_dict (dictionary): contains cross_val infos
    Returns:
    - model_path (torch.nn): path to the ckpt to load
    """
    cv_step = cross_dict['cv_step']
    ckpt_dir = os.path.join(folder, 'k{}'.format(cv_step+1))
    if 'SVM' in model_name:
        extension = 'pkl'
    else:
        extension = 'pth'
    ckpt_name = 'BEST_{}.{}'.format(model_name, extension)

    return os.path.join(ckpt_dir, ckpt_name)


def rgb_generator(img, bands, info_data, cross_dict=None):
    """ Generate rgb images to be displayed.
    Args:
    - img (np.array): HSI hypercube
    - bands (list[int]): list containing indexes of spectral bands
    used to build the rgb image dislayed
    - info_data (dict): dictionary containing data information
    - cross_dict (dict): dictionary containing cross validation information
    Returns:
    - rgb_list (list[np.array]): list containing rgb images to be displayed
    - img_name (list[str]): list containing image names to be displayed
    """
    rgb_list = []
    rgb_tmpl = '{0}_RGB-Image.png'

    if len(info_data['img_name']) > 1:
        img_name = info_data['img_name']
        img_shape = info_data['img_shape']
        for fold in img_name:
            name = os.path.basename(fold).split(' ')[0]
            path = os.path.join(info_data['folder'], fold,
                                rgb_tmpl.format(name))
            # rotate
            rgb_list.append(scipy.ndimage.rotate(Image.open(path), 270))

        if cross_dict:
            cv_step = cross_dict['cv_step']
            cv_coord = cross_dict['cv_coord'][cv_step]
            fold_size = cross_dict['fold_size']
            rgb_list_disp = []
            it_step = cv_coord[0]//img_shape[1]
            for x in range(fold_size):
                rgb_list_disp.append(rgb_list[it_step+x])
            rgb_list = rgb_list_disp
            if (cv_step+1)*fold_size < len(img_name):
                img_name = img_name[cross_dict['cv_step']*fold_size:
                                    (1+cross_dict['cv_step'])*fold_size]
            else:
                img_name = img_name[-fold_size:]
    else:
        img_name = ['']
        rgb = spectral.get_rgb(img, bands)
        rgb /= np.max(rgb)
        rgb_list = [np.asarray(255 * rgb, dtype='uint8')]

    return rgb_list, img_name


def map_generator(gt, info_data, cross_dict=None):
    """ Display the prediction pass through a mask overlayed on the
    rgb image of the HSI hypercube.
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
                coord_img = [cv_coord[0]+img_shape[1]*x]
                coord_img.append(coord_img[-1]+img_shape[1])
                gt_list.append(crop(coord_img))
        else:
            gt_list = np.split(gt, nb_img, 1)
    else:
        gt_list = [gt]
    return gt_list


def heatmap_img(rgb_img, proba, alpha=0.4, thresh=0):
    """ Generate heatmap overlayed with rgb image.
    Args:
    - rgb_img (np.array): rgb image
    - proba (np.array): probability map
    - alpha (float): alpha coefficient for transparency overlay
    - thresh (float): probability threshold
    Returns:
    - hm_img (np.array): heatmap image
    """
    # Generate overlay
    overlay = np.copy(rgb_img)
    heatmap = (proba * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay[proba > thresh] = heatmap[proba > thresh]

    # Apply overlay:
    overlay_img = cv2.addWeighted(overlay, alpha, rgb_img, 1-alpha, 0)

    return overlay_img
    # return heatmap


def display_heatmap(img, proba, retained_class, vis, bands,
                    info_data, cross_dict=None, alpha=0.4,
                    exp_name=''):
    """ Display the heatmap based on the prediction probability of the desired
    class.
    Args:
    - img (np.array): HSI hypercube
    - proba (np.array): colored prediction matrix
    - vis (visdom.Visdom): visdom visualizer
    - bands (list[int]): list containing indexes of spectral bands
    used to build the rgb image dislayed
    - info_data (dict): dictionary containing data information
    - cross_dict (dict): dictionary containing cross validation information
    """
    img_nm = 'heatmap_visu'
    if cross_dict:
        img_nm = 'k{}'.format(1+cross_dict['cv_step'])

    rgb_list, img_name = rgb_generator(img, bands, info_data,
                                       cross_dict=cross_dict)
    class_proba = proba[:, :, retained_class]
    proba_list = map_generator(class_proba, info_data, cross_dict=cross_dict)

    if len(img_name) > 1:
        suffx = True

    for i, (name, rgb_d, proba_d) in enumerate(
            zip(img_name, rgb_list, proba_list)):
        hm_img = heatmap_img(rgb_d, proba_d)

        vis.heatmap(
            X=proba_d,
            opts=dict(
                colormap='jet',
                jpgquality=50,
            )
        )
        plt.figure()
        plt.imshow(proba_d, cmap='jet')
        plt.colorbar(cmap='jet')
        plt.show()

        vis.images([np.transpose(hm_img, (2, 0, 1))],
                   opts={'caption': 'heatmap ' + name})

        # save image
        out_img = cv2.cvtColor(hm_img, cv2.COLOR_BGR2RGB)
        img_dir = './report/{}/Visualization/Heatmap'.format(exp_name)

        if suffx:
            out_nm = '{}_{}'.format(img_nm, str(i))
        os.makedirs(img_dir, exist_ok=True)
        cv2.imwrite(os.path.join(img_dir, '{}.jpg'.format(out_nm)), out_img)

def saving_results_analysis(prediction,gt,info_data,cross_dict,exp_name,img,bands):
    img_name = info_data['img_name']
    img_shape = info_data['img_shape']
    saving_dir = './report/{}/results_analysis/issue_50/'.format(exp_name)
    os.makedirs(saving_dir, exist_ok=True)
    extention_gt = '_gt.pkl'
    extention_pred = '_pred.pkl'
    extention_gt_png = '_gt.png'
    extention_pred_png = '_pred.png'

    data_list = []
    rgb_list, img_name = rgb_generator_reg(info_data, img=img,
                                       cross_dict=cross_dict,
                                       greyscale=False)
                                       
    pred_list = map_generator_reg(prediction, info_data, cross_dict=cross_dict)
    #proba_list = map_generator(proba, info_data, cross_dict=cross_dict)
    gt_list = map_generator_reg(gt, info_data, cross_dict=cross_dict)
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


def read_mask(mask_path):#, label_info):
    """ Read png mask.
    Args:
    - mask_path (str): path to the png image
    - label_info (list[int, str, list[str]]): list containing labels
    index, name and color
    Returns:
    - mask (np.array): ground truth mask
    """
    img = np.asarray(Image.open(mask_path))#.convert('RGB'))
    #mask = np.zeros(tuple(img.shape[:2]))

    #for [index, label, color] in label_info:
    #    mask[(img == color).all(-1)] = int(index)
    return img 

def name_mapping(info_data, exp_name, cross_dict):
    """ Generate a .txt file with the cross validation name mapping
    Args:
    - info_data (dict): ditionary containing dataset infos
    - exp_name (str): experiment name
    """
    img_name = info_data['img_name']
    folds_size = cross_dict['folds_size']
    nb_img = len(img_name)

    end_steps = list(np.cumsum(folds_size))
    start_steps = [0] + end_steps[:-1]

    nmap_path = os.path.join('report', exp_name, 'name_mapping.txt')
    with open(nmap_path, 'w+') as f:
        for i, (strt, end) in enumerate(zip(start_steps, end_steps)):
            if end < nb_img:
                f.write('k{} {}\n'.format(
                    i+1, '; '.join(img_name[strt:end])))
            else:
                f.write('k{} {}\n'.format(
                    1+1, '; '.join(img_name[-folds_size[-1]:])))


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

