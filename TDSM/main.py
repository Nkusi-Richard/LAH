# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR HYPERSPECTRAL DATA.

This script allows the user to run several deep models (and SVM baselines)
against various hyperspectral datasets. It is designed to quickly benchmark
state-of-the-art CNNs on various public hyperspectral datasets.

This code is released under the GPLv3 license for non-commercial and research
purposes only.
For commercial use, please contact the authors.
"""

# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary
from PIL import Image
# from torch.utils.tensorboard import SummaryWriter

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
import joblib
# from sklearn.externals import joblib
# from skimage import io

# Visualization
import seaborn as sns
from skimage import io, morphology
import visdom
import matplotlib.pyplot as plt
import datetime

from termios import tcflush, TCIFLUSH
import sys
import os
import yaml
import argparse
import copy
import shutil
import random

from utils import metrics, convert_to_color_, convert_from_color_,\
    display_dataset, display_predictions, explore_spectrums, plot_spectrums,\
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device,\
    data_stat, display_error, normalization, standardization,\
    ROC_curve, ROC_combined, channel_wised_normalization,\
    band_downsampling, pixel_downsampling, display_legend, \
    load_cv_ckpt, camel_to_snake, display_overlay, name_mapping,\
    save_score, class_normalization, snv_norm, calc_rgb,saving_results_analysis   # , HSI_ml

from datasets import get_dataset, HyperX, DATASETS_CONFIG
from models import get_model, train, test, save_model

EXP_NAME = None
ENV_VISDOM = None
EPOCH = None

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.set_deterministic(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)



def data_loader():
    """ Load the desired dataset and apply the required pre-process.
    Returns:
    - img (np.array): HSI hypercube
    - gt (np.array): associated label matrix
    - disp_img (np.array): HSI hypercube used for displaying rgb images
    - label_values (list[str]): list containing label names
    - ignored_labels (list[int]): list containing indexes of labels to
    be ignored
    """
    img, gt, label_values, ignored_labels, new_conf = get_dataset(
        DATASET, CONF_DATASET['loading'])
    # filtering out nan values in gt by setting default values
    gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
    disp_img = np.copy(img)
    norm_config = CONF_DATASET['normalization']
    bd_downsampling = CONF_DOWNSAMPLE['bd_downsampling']

    # pre-processing
    if bd_downsampling:
        bd_info = {'band_downsampling': bd_downsampling}
        img = band_downsampling(img, bd_downsampling)
        new_conf[DATASET]['info_data'].update(bd_info)

    if norm_config['normalized']:
        pre_proc_info = {'pre-processing': {
            'process:': 'normalization',
            'threshold': norm_config['norm_thresh']}}
        # TODO: check following line
        # disp_img = np.copy(img)
        img = normalization(img, thresh=norm_config['norm_thresh'])
        new_conf[DATASET].update(pre_proc_info)
    elif norm_config['standardized']:
        pre_proc_info = {'pre-processing': {
            'process:': 'standardization',
            'threshold': norm_config['norm_thresh']}}
        # TODO: check following line
        # disp_img = np.copy(img)
        img = standardization(img, thresh=norm_config['norm_thresh'])
        new_conf[DATASET].update(pre_proc_info)
    elif norm_config['channel_norm']:
        pre_proc_info = {'pre-processing': {
            'process:': 'Channel_wised_norm'}}
        img = channel_wised_normalization(img)
        new_conf[DATASET].update(pre_proc_info)
    elif norm_config['class_norm']:
        pre_proc_info = {'pre-processing': {
            'process:': 'Class_wised_norm'}}
        img = class_normalization(img, gt, norm_config['class_norm'],
                                  label_values,
                                  mode='standardization')
        new_conf[DATASET].update(pre_proc_info)
    elif norm_config['snv_norm']:
        pre_proc_info = {'pre-processing': {
            'process:': 'SNV_wised_norm)'}}
        img = snv_norm(img)
        new_conf[DATASET].update(pre_proc_info)

    DATASETS_CONFIG.update(new_conf)
    return img, gt, disp_img, label_values, ignored_labels


def training_process(img, train_gt, hyperparams, ignored_labels,
                     cross_dict, viz):
    """ Launch the training on the Train set
    Args:
    - img (np.array): HSI cube
    - train_gt (np.array): train label matrix
    - hyperparams (dict) dictionary containing all the model hyperparameters
    - ignored_labels (list[int]): list containing the indexes of labels to
    be ignored
    - viz (visdom.Visdom): visdom visualizer
    Returns:
    - model (sklearn.svm.SVC or nn.Module): trained model
    - hyperparams (dict): updated hyperparams dictionary
    """
    n_classes = hyperparams['n_classes']
    best_path = ''
    model_name = MODEL

    if MODEL == 'SVM_grid':
        # Parameters for the SVM grid search
        SVM_GRID_PARAMS = [{'kernel': ['rbf'],
                            'gamma': [1e-1, 1e-2, 1e-3],
                            'C': [1, 10, 100, 1000]},
                           {'kernel': ['linear'],
                            'C': [0.1, 1, 10, 100, 1000]},
                           {'kernel': ['poly'], 'degree': [3],
                            'gamma': [1e-1, 1e-2, 1e-3]}]

        print("Running a grid search SVM")
        # Grid search SVM (linear and RBF)
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=ignored_labels)
        class_weight = 'balanced' if CLASS_BALANCING else None
        model = sklearn.svm.SVC(class_weight=class_weight)
        model = sklearn.model_selection.GridSearchCV(model, SVM_GRID_PARAMS,
                                                     verbose=5, n_jobs=4)
        model.fit(X_train, y_train)
        print("SVM best parameters : {}".format(model.best_params_))
        save_model(model, MODEL, DATASET)

    elif MODEL == 'SVM':
        print('Hyperparams:{}'.format(hyperparams))
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=ignored_labels)
        class_weight = 'balanced' if CLASS_BALANCING else None

        if CHECKPOINT is not None:
            if SAMPLING_MODE == 'cross_val':
                best_path = load_cv_ckpt(model_name, CHECKPOINT, cross_dict)
            else:
                best_path = CHECKPOINT
            model = joblib.load(best_path)
        else:
            model = sklearn.svm.SVC(class_weight=class_weight, verbose=True,
                                    probability=True)
        if hyperparams['epoch'] != 0:
            model.fit(X_train, y_train)
            best_path = save_model(model, MODEL, DATASET, EXP_NAME, cross_dict,
                                   is_best=True)

    elif MODEL == 'SGD':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=ignored_labels)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        class_weight = 'balanced' if CLASS_BALANCING else None
        model = sklearn.linear_model.SGDClassifier(class_weight=class_weight,
                                                   learning_rate='optimal',
                                                   tol=1e-3, average=10)
        model.fit(X_train, y_train)
        save_model(model, MODEL, DATASET)

    elif MODEL == 'nearest':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=ignored_labels)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        class_weight = 'balanced' if CLASS_BALANCING else None
        model = sklearn.neighbors.KNeighborsClassifier(weights='distance')
        model = sklearn.model_selection.GridSearchCV(
            model, {'n_neighbors': [1, 3, 5, 10, 20]}, verbose=5, n_jobs=4)
        model.fit(X_train, y_train)
        model.fit(X_train, y_train)
        save_model(model, MODEL, DATASET)

    else:
        # Neural network
        if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, n_classes, ignored_labels)
            hyperparams['weights'] = torch.from_numpy(weights).float()

        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
        model_name = camel_to_snake(str(model.__class__.__name__))

        # Print model information
        print('Hyperparams:{}'.format(hyperparams))

        last_epoch = 0
        monitor_train = None
        if CHECKPOINT is not None:
            if SAMPLING_MODE == 'cross_val':
                best_path = load_cv_ckpt(
                    model_name, CHECKPOINT,
                    cross_dict,
                    full_resume=hyperparams['full_resume'])
            else:
                best_path = CHECKPOINT
            if best_path:
                ckpt_dict = torch.load(best_path)
                if hyperparams['full_resume']:
                    last_epoch = ckpt_dict['epoch']
                    monitor_train = ckpt_dict['monitor_train']
                    # Load model/optimizer states
                    model.load_state_dict(ckpt_dict['model_state_dict'])
                    optimizer.load_state_dict(
                        ckpt_dict['optimizer_state_dict'])
                else:
                    try:
                        model.load_state_dict(
                            ckpt_dict['model_state_dict'])
                    except KeyError:
                        # Maintain the loading of obsolete model checkpoints
                        model.load_state_dict(ckpt_dict)

        if hyperparams['epoch'] != 0 and 'BEST' not in best_path:
            # Split train set in train/val
            train_gt, val_gt = sample_gt(gt=train_gt, train_size=0.90,
                                         mode='random')

            # Generate the dataset
            train_dataset = HyperX(img, train_gt, **hyperparams)
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=hyperparams['batch_size'],
                # pin_memory=hyperparams['device'],
                worker_init_fn=seed_worker,
                generator=g,
                shuffle=True)
            val_dataset = HyperX(img, val_gt, **hyperparams)
            val_loader = data.DataLoader(val_dataset,
                                         worker_init_fn=seed_worker,
                                         generator=g,
                                         # shuffle=True,
                                         # pin_memory=hyperparams['device'],
                                         batch_size=hyperparams['batch_size'])
            with torch.no_grad():
                for input, _ in train_loader:
                    break

            if CUDA_DEVICE == 0:
                print("\nNetwork :")
                summary(model.to(hyperparams['device']), input.size()[1:])
                # We would like to use device=hyperparams['device'] altough we
                # have to wait for torchsummary to be fixed first.

            # Launch the training
            try:
                best_path = train(model, optimizer, loss, train_loader,
                                  hyperparams, EXP_NAME,
                                  cross_dict=cross_dict,
                                  val_loader=val_loader,
                                  display=viz,
                                  monitor_train=monitor_train,
                                  last_epoch=last_epoch)
            except KeyboardInterrupt:
                # Allow the user to stop the training
                pass

    return model, hyperparams, best_path, model_name


def inference(model, img, test_gt, best_checkpoint, hyperparams):
    """ Realize a prediction on the whole input hypercube
    Args:
    - model (nn.module): trained model
    - img (np.array): HSI hypercube
    - hyperparams(dict): dictionary containing model hyperparameters
    Returns:
    - prediction (np.array): prediction mask
    """
    n_bands = hyperparams['n_bands']
    n_classes = hyperparams['n_classes']
    ignored_labels = hyperparams['ignored_labels']

    if MODEL == 'SVM_GRID' or MODEL == 'SVM' or MODEL == 'nearest':
        if best_checkpoint.endswith('.pkl'):
            model = joblib.load(best_checkpoint)
        svm_classes = n_classes-len(ignored_labels)
        print('Inference on the image...')
        proba_svm = model.predict_proba(img.reshape(-1, n_bands))
        proba_svm = proba_svm.reshape(img.shape[:2]+(svm_classes,))

        # Fill ignored class probabilities:
        probabilities = np.zeros(img.shape[:2]+(n_classes,))
        ind_svm = 0
        for class_ind in range(n_classes):
            if class_ind in ignored_labels:
                continue
            else:
                probabilities[:, :, class_ind] = proba_svm[:, :, ind_svm]
                ind_svm += 1
        prediction = np.argmax(probabilities, axis=-1)

    elif MODEL == 'SGD':
        scaler = sklearn.preprocessing.StandardScaler()
        prediction = model.predict(scaler.transform(img.reshape(-1, n_bands)))
        prediction = prediction.reshape(img.shape[:2])

    else:
        if best_checkpoint.endswith('.pth'):
            ckpt_dict = torch.load(best_checkpoint)
            if('model_state_dict' in ckpt_dict.keys()):
                model.load_state_dict(ckpt_dict['model_state_dict'])
            else:
                model.load_state_dict(torch.load(best_checkpoint))
        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        prediction_ = np.copy(prediction)
        
        
        arr = prediction == 3 
        print("CONNECTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
        print(np.count_nonzero(arr))
        print(arr)
        print(arr.shape)
        arr[prediction == 2] = True
        print(np.count_nonzero(arr))
        #arr[test_gt == 0] = False

        cleaned = morphology.remove_small_objects(arr,min_size=50)
        #cleaned = morphology.remove_small_holes(cleaned, area_threshold=50)
        cleaned = morphology.remove_small_holes(cleaned, min_size=50)
        print(cleaned)
        prediction_[~cleaned] = 1
        prediction__ = prediction_
        """

       
        arr_ = prediction_ == 2
        cleaned_ = morphology.remove_small_objects(arr_,min_size=20000)
        cleaned_ = morphology.remove_small_holes(cleaned_, min_size=20000)
        prediction__[~cleaned_] =1 """

    return prediction__, probabilities


def compute_results(prediction, test_gt, label_values, hyperparams,
                    info_data, cross_dict=None):
    """Compute the results accoding to the pre-defined metrics and
    prepare the display of the error map
    Args:
    - prediction (np.array): prediction mask
    - test_gt (np.array): ground-truth mask
    - label_values (list[str]): label names
    - hyperparams (dict): dictionary containing model hyperparameters
    - info_data (dict): dictionary containing dataset info
    - cross_dict (dict): dictionary containing cross val info
    Returns:
    - run_results (dict): dictionary containing the different metrics
    scores
    - error_map (np.array): error mask
    """
    ignored_labels = hyperparams['ignored_labels']
    print("IGNORED LABELS")
    print(ignored_labels)
    n_classes = hyperparams['n_classes']

    # compute metric scores
    # Patient specific
    run_results = metrics(prediction, test_gt,
                          EXP_NAME, info_data,
                          ignored_labels=ignored_labels,
                          n_classes=n_classes,
                          labels=label_values,
                          cross_dict=cross_dict,
                          patient_specific=True)
    # Kfold specific
    kfold_results = metrics(prediction, test_gt,
                            EXP_NAME, info_data,
                            ignored_labels=ignored_labels,
                            n_classes=n_classes,
                            labels=label_values,
                            cross_dict=cross_dict,
                            patient_specific=False)

    # compute error
    disp_prediction = np.copy(prediction)
    mask = np.zeros(test_gt.shape, dtype='bool')
    for l in ignored_labels:
        mask[test_gt == l] = True
    disp_prediction[mask] = 0
    error_map = np.copy(disp_prediction)
    # error_map[np.nonzero(train_gt)] = 0
    err_mask = disp_prediction == test_gt
    error_map[err_mask] = 0

    return run_results, kfold_results, error_map


def main():
    """
    Main.
    """
    # # Prepare dataset-------------------------------------------------------
    # Load dataset
    img, gt, disp_img, label_init, ignored_labels = data_loader()
    disp_img_ = calc_rgb(disp_img)
    disp_img_ = np.dot(disp_img_[...,:3], [0.2989, 0.5870, 0.1140])
    print(np.max(disp_img_))

    # Number of classes
    n_classes = len(label_init)
    label_values = copy.deepcopy(label_init)

    if (CONF_DOWNSAMPLE['pix_downsampling'] and
            CONF_DOWNSAMPLE['pix_ds_config']['mode'] == 'patch'):
        assert 'FT_patch' in label_values, "'FT_patch' class is missing for"
        " patch downsampling"
        n_classes -= 1
        label_values.remove('FT_patch')

    data_info = DATASETS_CONFIG[DATASET]['info_data']

    # # Prepare displaying parameters-----------------------------------------
    # start Visdom connection
    viz = visdom.Visdom(env=ENV_VISDOM, port=CONF_VISU['visdom_port'])

    if not viz.check_connection:
        print("Visdom is not connected. Did you run 'python -m visdom.server'"
              " ?")

    # Generate color palette
    trait = '\n'+'='*100
    palette = {0: (0, 0, 0)}

    if CONF_VISU['palette']:
        for i, (lbl, color) in enumerate(CONF_VISU['palette'].items()):
            lbl_ind = label_init.index(lbl)
            palette[lbl_ind] = tuple(np.asarray(color, dtype='uint8'))

    for k, color in enumerate(
            sns.color_palette("hls", len(label_init) - 1)):
        palette.setdefault(k+1, tuple(np.asarray(255 * np.array(color),
                                                 dtype='uint8')))
    invert_palette = {v: k for k, v in palette.items()}

    def convert_to_color(x):
        y = np.copy(x)
        return convert_to_color_(y, palette=palette)

    def convert_from_color(x):
        return convert_from_color_(x, palette=invert_palette)

    # # Display dataset used--------------------------------------------------
    # Show the image and the ground truth
    print("HELLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print(disp_img.shape)
    display_dataset(disp_img, viz, data_info)
    display_predictions(convert_to_color(gt), viz, data_info,
                        caption="Available Ground Truth")
    # Display color code:
    display_legend(label_init, viz, palette, EXP_NAME,
                   ignored_labels=ignored_labels)

    # Show classe's spectral characteristics
    if CONF_EXPLORATION['with_exploration']:
        plot_scale = CONF_EXPLORATION['plot_scale']
        curve_class = CONF_EXPLORATION['curve_class']
        viz_spec = visdom.Visdom(env=ENV_VISDOM+'_Spectrums',
                                 port=CONF_VISU['visdom_port'])
        # Data exploration : compute and show the mean spectrums
        mean_spectrums = explore_spectrums(img, gt, label_values, viz_spec,
                                           data_info,
                                           ignored_labels=ignored_labels,
                                           plot_name='overall',
                                           exp_name=EXP_NAME,
                                           plot_scale=plot_scale)

        plot_spectrums(mean_spectrums, viz_spec,
                       data_info,
                       label_values, EXP_NAME, palette,
                       title='overall',
                       plot_name='overall',
                       plot_scale=plot_scale,
                       retained_class=curve_class)

        if VERBOSE:
            plt.show()
        else:
            plt.close('all')

        if REPORT_MODE:
            img_name = data_info['img_name']
            [h, w] = data_info['img_shape']
            for x, name in enumerate(img_name):
                info_spectrum = explore_spectrums(
                    img[:, x*w:(x+1)*w, :],
                    gt[:, x*w:(x+1)*w],
                    label_values, viz,
                    data_info,
                    ignored_labels=ignored_labels,
                    plot_name=name, exp_name=EXP_NAME,
                    plot_scale=plot_scale)
                plot_spectrums(info_spectrum, viz_spec,
                               data_info,
                               label_values, EXP_NAME, palette,
                               plot_scale=plot_scale,
                               retained_class=curve_class,
                               plot_name=name,
                               title='{} (acquisition {})'.format(
                                   name, str(x)))
                if VERBOSE:
                    plt.show()
                else:
                    plt.close('all')

    # Prepare model hyperparameters-----------------------------------------
    print("KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
    print(img.shape)
    print(img.shape[-1])
    hyperparams = {'dataset': DATASET}
    hyperparams.update({'n_classes': n_classes,
                        'n_bands': img.shape[-1],
                        'ignored_labels': ignored_labels,
                        'device': CUDA_DEVICE,
                       'label_values': label_values})
    hyperparams.update(CONF_MODEL)
    hyperparams.update(CONF_DATASET['data_augmentation'])
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    # # Launch training process-----------------------------------------------
    indiv_results = {}
    cv_results = {}
    prediction = np.zeros(gt.shape)
    proba = np.zeros(gt.shape+(n_classes,))
    ROC_info_kfold = dict()

    # Remove ignored labels
    for x in ignored_labels:
        print("HEREEEEEEEEEEEEEEEEEEEEEE")
        gt[gt == x] = 0

    # set-up cross validation mode
    n_runs = 1
    cross_dict = None
    if SAMPLING_MODE == 'cross_val':
        n_runs = CONF_DATASET['sampling']['kfolds']
        n_images = len(data_info['img_name'])
        fold_size = n_images // n_runs
        if (n_images % n_runs) != 0:
            fold_size += 1
        cv_coord = []
        img_shape = data_info['img_shape']
        for k in range(n_runs):
            if ((fold_size * (k + 1)) > (n_images)):
                cv_coord.append([img.shape[1]-fold_size*img_shape[1],
                                 img.shape[1]])
            else:
                cv_coord.append([img_shape[1]*k*fold_size,
                                 img_shape[1]*(k+1)*fold_size])

        cross_dict = {'kfold': n_runs,
                      'fold_size': fold_size,
                      'cv_step': 0,
                      'cv_coord': cv_coord}
        gt_cv = np.zeros(gt.shape)
        name_mapping(data_info, EXP_NAME, cross_dict)

    # run the experiment several times
    for run in range(n_runs):
        print("\nRunning an experiment with the {} model".format(MODEL),
              "run {}/{}".format(run + 1, n_runs))

        # Split the data
        if INFERENCE_MODE:
            train_gt = None
            test_gt = gt
            hyperparams['epoch'] = 0
        else:
            train_gt, test_gt = sample_gt(
                gt,
                CONF_DATASET['sampling']['proportion'],
                mode=SAMPLING_MODE,
                cross_dict=cross_dict)

        if CONF_DOWNSAMPLE['pix_downsampling']:
            # Apply pixel downsampling
            print("HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            pix_ds_config = CONF_DOWNSAMPLE['pix_ds_config']
            print(pix_ds_config)
            train_gt, test_gt = pixel_downsampling(
                train_gt, test_gt,
                pix_ds_config,
                label_init)
            DATASETS_CONFIG[DATASET]['info_data'].update(
                {'pix_downsampling': pix_ds_config})

        if not INFERENCE_MODE:
            # Display the split
            display_predictions(convert_to_color(train_gt), viz, data_info,
                                caption="Train ground truth")

            # Display dataset information
            print(trait)
            print('\n{0}: {1}'.format(DATASET, DATASETS_CONFIG[DATASET]))
            print(trait)
            data_stat(train_gt, gt, img, label_init)

        display_predictions(convert_to_color(test_gt), viz, data_info,
                            caption="Test ground truth")
        print(trait)

        # Train the model
        # TODO: decorrelate model loading and training
        model, hyperparams, best_ckpt, model_name = training_process(
            img, train_gt, hyperparams, ignored_labels, cross_dict, viz)
        print(trait)

        # Test the model
        if cross_dict:
            if REPORT_MODE:
                if CHECKPOINT:
                    best_ckpt = load_cv_ckpt(model_name,
                                             CHECKPOINT,
                                             cross_dict)
                else:
                    # Comment the following line before launching this script
                    # on GPU-server:
                    tcflush(sys.stdin, TCIFLUSH)
                    best_ckpt = input("{}-fold inference, please specify"
                                      "the checkpoint to load:\n".format(
                                          cross_dict['cv_step']+1))
                print("model loaded: {}".format(best_ckpt))

            cv_coord = cross_dict['cv_coord'][cross_dict['cv_step']]
            eval_img = np.copy(img[:, cv_coord[0]:cv_coord[1], :])
            prediction[:, cv_coord[0]:cv_coord[1]], proba[
                :, cv_coord[0]:cv_coord[1]] = inference(
                model, eval_img, test_gt, best_ckpt, hyperparams)

            # Save ground truth used
            gt_cv[:, cv_coord[0]:cv_coord[1]] = test_gt[
                :, cv_coord[0]:cv_coord[1]]
        else:
            prediction, proba = inference(model, img, test_gt, best_ckpt, hyperparams)


        #prediction[prediction == 2] = 3
        #test_gt[test_gt == 2] = 3
        print(np.unique(prediction))
        color_prediction = convert_to_color(prediction)
        print("Preddddddddddddddddddddddddddddddddd")
        predd = convert_from_color(prediction)
        print(np.unique(predd))
        print(prediction.shape)
        if REPORT_MODE and VERBOSE:
            plt.show()
        else:
            plt.close('all')

        # Compute kfold specific ROC curves
        ROC_info_kfold.update(ROC_curve(
            proba, test_gt,
            ignored_labels, label_values,
            EXP_NAME, palette, viz, data_info,
            cross_dict=cross_dict))

        # Compute result metrics
        print(label_values)
        run_results, kfold_results, error_map = compute_results(
            prediction,
            test_gt,
            label_values,
            hyperparams,
            data_info,
            cross_dict=cross_dict)

        # Display results and error
        """
        test_gt1 = np.copy(test_gt)
        test_gt1[test_gt > 0] = 255
        mask1 = Image.fromarray(test_gt1)
        mask1.show()
        mask = test_gt == 0
        prediction[mask] = 0
        prediction_ = np.copy(prediction)
        arr = prediction == 3

        cleaned = morphology.remove_small_objects(arr,min_size=1000)
        cleaned = morphology.remove_small_holes(cleaned, min_size=1000)
        prediction_[cleaned] = 1
        prediction = prediction_
        """

        viz.matplot(plt)
        print("OTKKKK")
        print(np.unique(test_gt))
        gt_err = np.copy(test_gt)
        """
        display_error(disp_img,
                      color_prediction,
                      convert_to_color(error_map),
                      viz,
                      data_info,
                      label_values, palette,
                      exp_name=EXP_NAME,
                      cross_dict=cross_dict,
                      gt=convert_to_color(gt_err))

        """
        #disp_img_ = np.dot(disp_img[...,:3], [0.2989, 0.5870, 0.1140])
        """
        mask_disp = disp_img_ >= 180
        prediction[mask_disp] = 1
        color_prediction = convert_to_color(prediction)"""
        # Display overlay prediction and heatmap
        try:
            lbl_indx = label_values.index(CONF_VISU['lbl_visu'])
            pred_mask = prediction == lbl_indx
            gt_mask = test_gt == lbl_indx
            # display_heatmap(disp_img, proba,
            #                 lbl_indx,
            #                 viz,
            #                 data_info, cross_dict=cross_dict, alpha=0.8,
            #                 exp_name=EXP_NAME)
        except ValueError:
            print("Warning: '{}' is not a class label, all classes will be "
                  "overlayed.".format(CONF_VISU['lbl_visu']))
            gt_mask = test_gt != 0
            pred_mask = gt_mask

        print("HEREEEEEEEEEEEEEEEEEEEEEEEEEEE")
        """
        disp_img_ = np.dot(disp_img[...,:3], [0.2989, 0.5870, 0.1140])
        mask_disp = disp_img_ >= 240
        prediction[mask_disp] = 1
        color_prediction = convert_to_color(prediction)
        print(disp_img.shape)
        """
        display_error(disp_img,
                      color_prediction,
                      convert_to_color(error_map),
                      gt_mask,
                      pred_mask,
                      viz,
                      data_info,
                      label_values, palette,
                      exp_name=EXP_NAME,
                      cross_dict=cross_dict,
                      gt=convert_to_color(gt_err))

        display_overlay(disp_img,
                        color_prediction,
                        convert_to_color(test_gt),
                        gt_mask,
                        pred_mask,
                        viz,
                        data_info,
                        palette,
                        cross_dict=cross_dict,
                        alpha=1,
                        exp_name=EXP_NAME)

        saving_results_analysis(prediction,gt_err,data_info,cross_dict,EXP_NAME,disp_img)
        indiv_results.update(run_results)
        cv_results.update(kfold_results)
        print(trait)

        _ = show_results(kfold_results, viz, EXP_NAME,
                         ignored_labels=ignored_labels,
                         label_values=label_values)

        if VERBOSE:
            plt.show()
        else:
            plt.close('all')

        if not cross_dict:
            display_predictions(color_prediction, viz, data_info,
                                gt=convert_to_color(test_gt),
                                caption="Prediction vs. test ground truth")
        else:
            cross_dict['cv_step'] += 1
        torch.cuda.empty_cache()

    save_score(proba, data_info, label_values, exp_name=EXP_NAME)
    if cross_dict:
        # Overall ROC curves
        _ = ROC_curve(proba, gt_cv, ignored_labels, label_values,
                      EXP_NAME, palette, viz, data_info)

        if VERBOSE:
            plt.show()
        else:
            plt.close('all')
        # Patient specific ROC curves
        ROC_info_patient = ROC_curve(proba, gt_cv, ignored_labels,
                                     label_values, EXP_NAME, palette,
                                     viz, data_info, patient_specific=True)

        if VERBOSE:
            plt.show()
        else:
            plt.close('all')

        # Kfold specific ROC combined curves
        ROC_combined(ROC_info_kfold, EXP_NAME)

        if VERBOSE:
            plt.show()
        else:
            plt.close('all')

        color_prediction = convert_to_color(prediction)
        # run_results, error_map = compute_results(prediction,
        #                                          gt,
        #                                          label_values,
        #                                          hyperparams)
        # Display results and error
        viz.matplot(plt)
        display_predictions(color_prediction, viz, data_info,
                            gt=convert_to_color(gt_cv),
                            caption="Prediction vs. test ground truth")

    else:
        # Patient specific ROC curves
        ROC_info_patient = ROC_curve(proba, test_gt, ignored_labels,
                                     label_values, EXP_NAME, palette,
                                     viz, data_info, patient_specific=True)

        # Kfold specific ROC combined curves
        ROC_combined(ROC_info_patient, EXP_NAME, patient_specific=True)

    if VERBOSE:
        plt.show()
    else:
        plt.close('all')

    if n_runs > 1:
        # Generate performance report per subset
        subsets = ['']
        try:
            subsets += data_info['subsets']
        except KeyError:
            pass

        for subset in subsets:
            # Subset Confusion matrices
            try:
                perf_text = show_results(indiv_results, viz, EXP_NAME,
                                         ignored_labels=ignored_labels,
                                         label_values=label_values,
                                         agregated=True, subset=subset)
                # Subset Patient-specific ROC combined curves
                ROC_combined(ROC_info_patient, EXP_NAME,
                             patient_specific=True,
                             subset=subset)
            except:
                print("Warning: report on '{}' will be skipped, this subset"
                      " seems to be missing in your dataset. To remove this"
                      " warning please remove it from the subsets specified"
                      " in the configuration file.".format(subset))

            if VERBOSE:
                plt.show()
            else:
                plt.close('all')

        # Temporary replacement of log file
        with open(TEMP_LOG, 'w+') as f:
            f.write(EXP_NAME + '\n')
            f.write(str(hyperparams))
            f.write(trait)
            f.write('\n{}: {}'.format(DATASET, DATASETS_CONFIG[DATASET]))
            f.write(trait)
            f.write(perf_text)


def environment_name():
    """ Define visdom and report environment names
    """
    global EXP_NAME
    global ENV_VISDOM
    info_xp = [DATASET, MODEL, SAMPLING_MODE]

    if INFERENCE_MODE:
        info_xp.pop()

    EXP_NAME = '_'.join(info_xp)
    ENV_VISDOM = ' '.join(info_xp)

    if not(REPORT_MODE or INFERENCE_MODE):
        time_now = str(datetime.datetime.now()).split(' ')
        EXP_NAME += '_'.join(time_now)
        ENV_VISDOM += '_'.join(time_now)
    else:
        ENV_VISDOM += '_Report'
        global EPOCH
        EPOCH = 0

        viz = visdom.Visdom(port=int(CONF_VISU['visdom_port']))
        list_env = viz.get_env_list()
        ind = 0
        for env_name in list_env:
            if ENV_VISDOM == env_name[:-2]:
                ind = max(ind, int(env_name[-1])+1)
        EXP_NAME += '_{}'.format(ind)
        ENV_VISDOM += '_{}'.format((ind))

    if CONF_PROCESS['debug']:
        EXP_NAME = 'test'
        ENV_VISDOM = 'test'
    print('Environment:', EXP_NAME)
    EXP_NAME = os.path.join(CONF_PROCESS['output_path'], EXP_NAME)


if __name__ == '__main__':
    dataset_names = [v['name'] if 'name' in v.keys() else k for k,
                     v in DATASETS_CONFIG.items()]

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="Specify configuration path")
    args = parser.parse_args()
    PARSED_YML = yaml.load(open(args.config_file),
                           Loader=yaml.FullLoader)

    CONF_PROCESS = PARSED_YML['process']
    CONF_EXPLORATION = PARSED_YML['spectrum_exploration']
    CONF_DATASET = PARSED_YML['dataset']
    CONF_DOWNSAMPLE = CONF_DATASET['downsampling']
    CONF_MODEL = PARSED_YML['model']
    CONF_VISU = PARSED_YML['visualization']

    CUDA_DEVICE = get_device(CONF_PROCESS['cuda'])
    DATASET = CONF_DATASET['dataset_name']
    MODEL = CONF_MODEL['model_name']
    CLASS_BALANCING = CONF_MODEL['class_balancing']
    CHECKPOINT = CONF_MODEL['checkpoint']
    SAMPLING_MODE = CONF_DATASET['sampling']['mode']

    VERBOSE = CONF_VISU['verbose']
    REPORT_MODE = CONF_PROCESS['report']
    INFERENCE_MODE = CONF_PROCESS['inference']

    # Setup env names
    environment_name()

    # Temporary replacement of log file
    TEMP_LOG = os.path.join(EXP_NAME, 'report/log.out')

    # Save configuration file
    os.makedirs(EXP_NAME, exist_ok=True)
    shutil.copy(args.config_file, os.path.join(
        EXP_NAME, os.path.basename(args.config_file)))
    main()
