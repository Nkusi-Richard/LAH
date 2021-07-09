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
from sklearn.svm import SVR
import cv2
from skimage import io, morphology
import torch.utils.data as data
from torchsummary import summary
# from torch.utils.tensorboard import SummaryWriter

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
#from sklearn.externals import joblib
# from skimage import io

# Visualization
import seaborn as sns
import visdom
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import datetime
import joblib
# Comment the following line before launching this script on GPU-server:
from termios import tcflush, TCIFLUSH
import sys
import random
import os

from utils import metrics, convert_to_color_, convert_from_color_,\
    display_dataset, display_predictions, explore_spectrums, plot_spectrums,\
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device,\
    data_stat, display_error, normalization, standardization,\
    channel_wised_normalization, class_normalization, ROC_curve,\
    ROC_combined, display_overlay, snv_norm, pixel_downsampling, \
    load_cv_ckpt, camel_to_snake, display_legend, band_downsampling,name_mapping,\
    display_heatmap,saving_results_analysis,display_error_reg  # , HSI_ml
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model
from sklearn.metrics import plot_confusion_matrix
import argparse

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
    - LABEL_VALUES (list[str]): list containing label names
    - IGNORED_LABELS (list[int]): list containing indexes of labels to
    be ignored
    - RGB_BANDS (list[int]): list containing indexes of spectral bands
    used to build the rgb image dislayed
    """
    img, gt,IGNORED_LABELS,RGB_BANDS, \
        new_conf = get_dataset(DATASET, FOLDER)
    disp_img = np.copy(img)

    # pre-processing
    if ARGS.bd_downsampling:
        bd_info = {'band_downsampling': ARGS.bd_downsampling}
        img = band_downsampling(img, ARGS.bd_downsampling)
        new_conf[DATASET]['info_data'].update(bd_info)

    if ARGS.normalized:
        pre_proc_info = {'pre-processing': {
            'process:': 'normalization',
            'threshold': ARGS.norm_thresh}}
        # TODO: check following line
        # disp_img = np.copy(img)
        img = normalization(img, thresh=ARGS.norm_thresh)
        new_conf[DATASET].update(pre_proc_info)
    elif ARGS.standardized:
        pre_proc_info = {'pre-processing': {
            'process:': 'standardization',
            'threshold': ARGS.norm_thresh}}
        # TODO: check following line
        # disp_img = np.copy(img)
        img = standardization(img, thresh=ARGS.norm_thresh)
        new_conf[DATASET].update(pre_proc_info)
    elif ARGS.channel_norm:
        pre_proc_info = {'pre-processing': {
            'process:': 'Channel_wised_norm'}}
        img = channel_wised_normalization(img)
        new_conf[DATASET].update(pre_proc_info)
    elif ARGS.class_norm:
        pre_proc_info = {'pre-processing': {
            'process:': 'Class_wised_norm)'}}
        # by default, channel wised standardization is used:
        img = class_normalization(img, gt, 'Healthy', LABEL_VALUES,
                                  mode='rel_diff')
    elif ARGS.snv_norm:
        pre_proc_info = {'pre-processing': {
            'process:': 'SNV_wised_norm)'}}
        img = snv_norm(img)
        new_conf[DATASET].update(pre_proc_info)
    DATASETS_CONFIG.update(new_conf)
    return img, gt, disp_img, IGNORED_LABELS, RGB_BANDS


def data_split(gt, cross_dict=None):
    """ Split Train/Test dataset
    Args:
    - gt (np.array): label matrix
    - kfold (int): kfold iteration
    - cv_step (int): current iteration
    Returns:
    - test_gt (np.array): test label matrix
    - train_gt (np.array): train label matrix
    """
    if TRAIN_GT is not None and TEST_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = open_file(TEST_GT)
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w, :h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
        # Sample random training spectra
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE,
                                      mode=SAMPLING_MODE,
                                      cross_dict=cross_dict)
    return train_gt, test_gt


def training_process(img, train_gt, hyperparams,  IGNORED_LABELS,
                     cross_dict, viz):
    """ Launch the training on the Train set
    Args:
    - img (np.array): HSI cube
    - train_gt (np.array): train label matrix
    - hyperparams (dict) dictionary containing all the model hyperparameters
    - LABEL_VALUES (list[str]): list containing the classe names
    - IGNORED_LABELS (list[int]): list containing the indexes of labels to
    be ignored
    - viz (visdom.Visdom): visdom visualizer
    Returns:
    - model (sklearn.svm.SVC or nn.Module): trained model
    - hyperparams (dict): updated hyperparams dictionary
    """
    N_CLASSES = hyperparams['n_classes']
    MODEL_NAME = MODEL

    best_path = ''
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
                                         ignored_labels=IGNORED_LABELS)
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
                                         ignored_labels=IGNORED_LABELS)
        class_weight = 'balanced' if CLASS_BALANCING else None
        print("OOOOOOOOOOOOOOOO")
        print(X_train.shape)
        print(y_train.shape)

        if CHECKPOINT is not None:
            if SAMPLING_MODE == 'cross_val':
                best_path = load_cv_ckpt(MODEL_NAME, CHECKPOINT, cross_dict)
            else:
                best_path = CHECKPOINT
            model = joblib.load(best_path)
        else:
            if ARGS.regression:
                print("REGRESSION")
                model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
            else:
                model = sklearn.svm.SVC(class_weight=class_weight, verbose=True,
                                    probability=True)

        if hyperparams['epoch'] != 0:
            print("REGREEEEEEEEEEEEEEE")
            print(model)
            model.fit(X_train, y_train)
            print(model)
            best_path = save_model(model, MODEL, DATASET, EXP_NAME, cross_dict,
                                   is_best=True)

    elif MODEL == 'SGD':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
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
                                         ignored_labels=IGNORED_LABELS)
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
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams['weights'] = torch.from_numpy(weights).float()
        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
        if ARGS.regression:
            if ARGS.reg_loss == "mse":
                loss = torch.nn.MSELoss()
            elif ARGS.reg_loss == "mae":
                loss = torch.nn.L1Loss()

        MODEL_NAME = camel_to_snake(str(model.__class__.__name__))

        # Split train set in train/val
        train_gt, val_gt = sample_gt(gt=train_gt,train_size=0.90,
                                     mode='random')
        # Generate the dataset
        train_dataset = HyperX(img, train_gt, **hyperparams)
        train_loader = data.DataLoader(train_dataset,
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

        # Print model information
        print('Hyperparams:{}'.format(hyperparams))
        if ARGS.cuda == 0:
            print("\nNetwork :")
            summary(model.to(hyperparams['device']), input.size()[1:])
            # We would like to use device=hyperparams['device'] altough we
            # have to wait for torchsummary to be fixed first.
        if CHECKPOINT is not None:
            if SAMPLING_MODE == 'cross_val':
                best_path = load_cv_ckpt(MODEL_NAME, CHECKPOINT, cross_dict)
            else:
                best_path = CHECKPOINT
            model.load_state_dict(torch.load(best_path))

        # Launch the training
        try:
            best_path = train(model, optimizer, loss, train_loader,
                              hyperparams['epoch'],
                              EXP_NAME,
                              cross_dict=cross_dict,
                              scheduler=hyperparams['scheduler'],
                              device=hyperparams['device'],
                              supervision=hyperparams['supervision'],
                              val_loader=val_loader,
                              display=viz)
        except KeyboardInterrupt:
            # Allow the user to stop the training
            pass

    return model, hyperparams, best_path, MODEL_NAME

def undesired_objects (image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    print(nb_components)
    sizes = stats[:, -1]; nb_components = nb_components 

    max_label = 1
    max_size = 150#sizes[1]
    print(stats)
    img2 = np.zeros(output.shape)
    for i in range(0, nb_components):
        print("Here")
        print(sizes[i])
        if sizes[i] > 25000:
            max_label = i
            img2[output == i + 1] = 1 

    #img2 = np.zeros(output.shape)
    #img2[output == max_label] =2 
    print("OKAY")
    print(np.unique(img2))
    return img2

def inference(model, img, best_checkpoint, hyperparams):
    """ Realize a prediction on the whole input hypercube
    Args:
    - model (nn.module): trained model
    - img (np.array): HSI hypercube
    - hyperparams(dict): dictionary containing model hyperparameters
    Returns:
    - prediction (np.array): prediction mask
    """
    print("INFERENCE")
    print(img.shape)
    N_BANDS = hyperparams['n_bands']
    #N_CLASSES = hyperparams['n_classes']
    IGNORED_LABELS = hyperparams['ignored_labels']

    if MODEL == 'SVM_GRID' or MODEL == 'SVM' or MODEL == 'nearest':
        if best_checkpoint.endswith('.pkl'):
            model = joblib.load(best_checkpoint)
        if ARGS.regression:
            SVM_CLASSES = 1
            N_CLASSES = SVM_CLASSES
        else:
            SVM_CLASSES = N_CLASSES-len(IGNORED_LABELS)
        prediction = model.predict(img.reshape(-1, N_BANDS))
        # prediction = prediction.reshape(img.shape[:2])
        print(N_BANDS)
        print(img.shape)
        #proba_svm = model.predict_proba(img.reshape(-1, N_BANDS))
        proba_svm = prediction 
        print("SVMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
        print(proba_svm.shape)
        #proba_svm = proba_svm.reshape(img.shape[:2]+(SVM_CLASSES,))
        proba_svm = proba_svm.reshape(img.shape[:2])
        print(proba_svm.shape)

        #probabilities = np.zeros(img.shape[:2]+(N_CLASSES,))
        probabilities = np.zeros(img.shape[:2])
        ind_svm = 0
        if ARGS.regression:
            prediction_ = proba_svm
        else :
            for class_ind in range(N_CLASSES):
                if class_ind in IGNORED_LABELS:
                    continue
                else:
                    probabilities[:, :, class_ind] = proba_svm[:, :, ind_svm] 
                    ind_svm += 1
            prediction = np.argmax(probabilities, axis=-1)

    elif MODEL == 'SGD':
        scaler = sklearn.preprocessing.StandardScaler()
        prediction = model.predict(scaler.transform(img.reshape(-1, N_BANDS)))
        prediction = prediction.reshape(img.shape[:2])

    else:
        if best_checkpoint.endswith('.pth'):
            model.load_state_dict(torch.load(best_checkpoint))
        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        if ARGS.regression:
            prediction = probabilities
        prediction_ = prediction 
        print("OKAYYYYYYYYYYYYYYYYYYYY")
        print(prediction.shape)
        #print(np.unique(np.max(probabilities, axis=-1)))
        """ 
        ##print((prediction))
        ##prediction = morphology.remove_small_objects(prediction, min_size=2, connectivity=4, in_place=False)
        ##prediction = morphology.remove_small_holes(prediction, area_threshold=0, connectivity=4, in_place=False)
        #prediction = undesired_objects (prediction)
        arr = prediction == 1 
        ##arr =  prediction.astype('bool')
       
        cleaned = morphology.remove_small_objects(arr,min_size=20000)
        cleaned = morphology.remove_small_holes(cleaned, min_size=20000)
        ##cleaned[cleaned==0] = 1
        ##cleaned[cleaned==1] =2 
        ##prediction = cleaned
        prediction_[~cleaned] = 2
        ##prediction_[cleaned_] = 2
        ##prediction_[~cleaned_] =1 
        ##prediction_[~cleaned] = 2
        prediction__ = prediction_
        
        arr_ = prediction_ == 2
        cleaned_ = morphology.remove_small_objects(arr_,min_size=20000)
        cleaned_ = morphology.remove_small_holes(cleaned_, min_size=20000)
        prediction__[~cleaned_] =1 """

    return prediction_, probabilities


def compute_results(prediction, test_gt, hyperparams,
                    iteration=0):
    """Compute the results accoding to the pre-defined metrics and
    prepare the display of the error map
    Args:
    - prediction (np.array): prediction mask
    - test_gt (np.array): ground-truth mask
    - LABEL_VALUES (list[str]): label names
    - hyperparams (dict): dictionary containing model hyperparameters
    Returns:
    - run_results (dict): dictionary containing the different metrics
    scores
    - error_map (np.array): error mask
    """
    ignored_labels = hyperparams['ignored_labels']
    #N_CLASSES = hyperparams['n_classes']

    # compute metric scores
    run_results = metrics(prediction,
                          test_gt,
                          EXP_NAME,
                          ignored_labels=ignored_labels,
                          cm_id=str(1+iteration))#,
                          #n_classes=N_CLASSES,
                          #labels=LABEL_VALUES)

    # compute error
    disp_prediction = np.copy(prediction)
    mask = np.zeros(test_gt.shape, dtype='bool')
    for l in ignored_labels:
        mask[test_gt == l] = True
    disp_prediction[mask] = 0
    error_map = np.copy(disp_prediction)
    # error_map[np.nonzero(train_gt)] = 0
    """err_mask = disp_prediction == test_gt
    error_map[err_mask] = 0"""

    # # error exploration
    # ignored_lbl = list(np.unique(gt))
    # ignored_lbl.remove(1.0)
    # ignored_lbl.remove(2.0)
    # ignored_lbl.remove(8.0)
    # ignored_lbl.remove(12.0)
    # mask = np.zeros(gt.shape, dtype='bool')
    # for l in ignored_lbl:
    #     print(l, gt == int(l))
    #     mask[gt == int(l)] = True
    # test_gt[mask] = 0
    # prediction[mask] = 0
    # error_map[mask] = 0
    return run_results, error_map

def cross_val_generator(data_info, subset_specific=False):
    """ Set-up cross validation mode
    Args:
    - data_info (dict): dictionary containing data information
    - subset_specific (bool): generate a subset-based cross validation
    Returns:
    - cross_dict (dict): dictionary containing cross validation info
    - n_runs (int): number of training iteration
    """
    # Subset based cross validation
    if ARGS.kfolds == -1:
        subsets = data_info['subsets']
        img_name = data_info['img_name']
        n_runs = len(subsets)
        folds_size = []
        for subset in subsets:
            # Count images per subset
            folds_size.append(
                sum(map(lambda i: i.startswith(subset), img_name)))
    else:
        n_runs = ARGS.kfolds
        n_images = len(data_info['img_name'])
        folds_size = [n_images // n_runs] * n_runs
        if (n_images % n_runs) != 0:
            folds_size = [x + 1 for x in folds_size]

    cv_coord = []
    img_shape = data_info['img_shape']
    cube_shape = data_info['cube_shape']

    end_steps = list(np.cumsum(folds_size))
    start_steps = [0] + end_steps[:-1]

    cv_coord = [[strt*img_shape[1], end*img_shape[1]] for strt, end in zip(
        start_steps, end_steps)]

    if cv_coord[-1][-1] > cube_shape[1]:
        cv_coord[-1] = [cube_shape[1]-folds_size[-1]*img_shape[1],
                        cube_shape[1]]

    cross_dict = {'kfold': n_runs,
                  'folds_size': folds_size,
                  'cv_step': 0,
                  'cv_coord': cv_coord}
    #name_mapping(data_info, EXP_NAME, cross_dict)

    return cross_dict, n_runs



def main():
    """
    Main.
    """
    # # Prepare dataset-------------------------------------------------------
    # Download dataset
    if ARGS.download is not None and len(ARGS.download) > 0:
        for dataset in ARGS.download:
            get_dataset(dataset, target_folder=FOLDER)
            quit()
    # Load dataset
    img, gt, disp_img, IGNORED_LABELS, RGB_BANDS = data_loader()
    print("DATA IMG")
    print(img.shape)
    print(disp_img.shape)
    # Number of classes
    if ARGS.regression:
        N_CLASSES = 1
    else:
        N_CLASSES = len(LABEL_VALUES)
    # Number of bands (last dimension of the image tensor)
    N_BANDS = img.shape[-1]
    DATA_INFO = DATASETS_CONFIG[DATASET]['info_data']

    # # Prepare displaying parameters-----------------------------------------
    # start Visdom connection
    viz = visdom.Visdom(env=ENV_VISDOM, port=ARGS.visdom_port, log_to_filename='visdom')
    print('Environment:', EXP_NAME)
    if not viz.check_connection:
        print("Visdom is not connected. Did you run 'python -m visdom.server'"
              " ?")
    trait = '\n'+'='*100
    """
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(
            sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color),
                                          dtype='uint8'))
    invert_palette = {v: k for k, v in palette.items()}

    def convert_to_color(x):
        return convert_to_color_(x, palette=palette)

    def convert_from_color(x):
        return convert_from_color_(x, palette=invert_palette)"""

    # # Display dataset used--------------------------------------------------
    # Show the image and the ground truth
    
    #display_dataset(disp_img, gt, RGB_BANDS, viz,
    #                DATASETS_CONFIG[DATASET])
    print(disp_img.shape)
    #display_dataset(disp_img, viz, DATA_INFO)
    #display_predictions(gt, viz,
    #                    DATA_INFO,
    #                    caption="Available Ground Truth")
    # Display color code:
    #display_legend(LABEL_VALUES, viz, palette, None, EXP_NAME)

    # Show classe's spectral characteristics
    """
    if DATAVIZ:
        # Data exploration : compute and show the mean spectrums
        mean_spectrums = explore_spectrums(
            img, gt, LABEL_VALUES,palette, viz,
            DATASETS_CONFIG[DATASET]['info_data'],
            ignored_labels=IGNORED_LABELS,
            plot_name='overall',
            exp_name=EXP_NAME,
            plot_scale=PLOT_SCALE)
        plot_spectrums(mean_spectrums, viz,
                       DATASETS_CONFIG[DATASET]['info_data'],
                       LABEL_VALUES, EXP_NAME, palette,
                       title='overall',
                       plot_name='overall',
                       plot_scale=PLOT_SCALE)
        if VERBOSE:
            plt.show()
        else:
            plt.close('all')

        if REPORT_MODE:
            img_name = DATA_INFO['img_name']
            [h, w] = DATA_INFO['img_shape']
            for x, name in enumerate(img_name):
                mean_spectrum = explore_spectrums(
                    img[:, x*w:(x+1)*w, :],
                    gt[:, x*w:(x+1)*w],
                    LABEL_VALUES, palette, viz,
                    DATASETS_CONFIG[DATASET]['info_data'],
                    ignored_labels=IGNORED_LABELS,
                    plot_name=name, exp_name=EXP_NAME,
                    plot_scale=PLOT_SCALE)
                plot_spectrums(mean_spectrum, viz,
                               DATASETS_CONFIG[DATASET]['info_data'],
                               LABEL_VALUES, EXP_NAME, palette,
                               plot_scale=PLOT_SCALE,
                               plot_name=name,
                               title='{} (acquisition {})'.format(
                                   name, str(x)))
                if VERBOSE:
                    plt.show()
                else:
                    plt.close('all')"""

    # # Prepare model hyperparameters-----------------------------------------
    for x in IGNORED_LABELS:
        gt[gt == x] = 0

    hyperparams = vars(ARGS)
    """
    hyperparams.update({'n_classes': N_CLASSES,
                        'n_bands': N_BANDS,
                        'ignored_labels': IGNORED_LABELS,
                        'device': CUDA_DEVICE,
                       'label_values': LABEL_VALUES})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)"""
    hyperparams.update({'n_classes': N_CLASSES,
                       'n_bands': N_BANDS, 
                        'ignored_labels': IGNORED_LABELS,
                        'device': CUDA_DEVICE
                      })
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    # # Launch training process-----------------------------------------------
    results = []
    prediction = np.zeros(gt.shape)
    #print(prediction.shape)
    #prediction[prediction == 0] = 2
    #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    #print(np.unique(prediction))
    #proba = np.zeros(gt.shape+(len(LABEL_VALUES),))
    proba = np.zeros(gt.shape)
    #print(proba.shape)
    ROC_info = dict()
    # Number of runs (for cross-validation)
    N_RUNS = ARGS.runs

    # set-up cross validation mode
    cross_dict = None
    if SAMPLING_MODE == 'cross_val':
        # TO DO : replace in this next line N_RUNS by a CL option KFOLDS
        # N_RUNS = len(DATA_INFO['img_name'])
        # I keep the N_RUNS var to keep it compatible with the normal run
        """
        N_RUNS = ARGS.kfolds
        n_images = len(DATA_INFO['img_name'])
        fold_size = n_images // N_RUNS
        if (n_images % N_RUNS) != 0:
            fold_size += 1
        cv_coord = []
        img_shape = DATA_INFO['img_shape']
        for k in range(N_RUNS):
            if ((fold_size * (k + 1)) > (n_images)):
                cv_coord.append([img.shape[1]-fold_size*img_shape[1],
                                 img.shape[1]])
            else:
                cv_coord.append([img_shape[1]*k*fold_size,
                                 img_shape[1]*(k+1)*fold_size])

        cross_dict = {'kfold': ARGS.kfolds,
                      'fold_size': fold_size,
                      'cv_step': 0,
                      'cv_coord': cv_coord}"""

        cross_dict, N_RUNS = cross_val_generator(DATA_INFO,True)
        gt_cv = np.zeros(gt.shape)


    # run the experiment several times
    for run in range(N_RUNS):
        print("\nRunning an experiment with the {} model".format(MODEL),
              "run {}/{}".format(run + 1, N_RUNS))

        # Split the data
        train_gt, test_gt = data_split(gt, cross_dict=cross_dict)

        # Pixel downsampling:
        if PIX_DS:
            # Apply pixel downsampling
            train_gt, test_gt = pixel_downsampling(train_gt, test_gt, PIX_DS,
                                                   LABEL_VALUES)
            DATASETS_CONFIG[DATASET]['info_data'].update(
                {'pix_downsampling': PIX_DS})

        # ===================================================================
        # # Exp Liver:
        # train_gt, _ = sample_gt(train_gt, 0.5,
        #                         mode='random',
        #                         cross_dict=cross_dict)
        #train_gt, _ = pixel_downsampling(
        #     train_gt, 0.5,
        #     retained_class=LABEL_VALUES.index('No_Liver'),
        #     sampling_mode='random')

        # # Exp Neck:
        # train_gt, _ = pixel_downsampling(
        #     train_gt, 0.5,
        #     retained_class=LABEL_VALUES.index('Skin'),
        #     sampling_mode='random')
        # train_gt, _ = pixel_downsampling(
        #     train_gt, 0.5,
        #     retained_class=LABEL_VALUES.index('Muscle'),
        #     sampling_mode='random')

        # # Iss46:
        # train_gt, _ = pixel_downsampling(
        #     train_gt, 0.5,
        #     retained_class=LABEL_VALUES.index('Healthy_Tissue'),
        #     sampling_mode='random')

        # # Exp 3:
        # train_gt, _ = pixel_downsampling(
        #     train_gt, 0.5,
        #     retained_class=LABEL_VALUES.index('Healthy_f'),
        #     sampling_mode='random')

        # # Exp 4:
        # train_gt, _ = pixel_downsampling(
        #     train_gt, 0.5,
        #     retained_class=LABEL_VALUES.index('Healthy_S'),
        #     sampling_mode='random')

        # # Exp 5 a):
        # test_gt, train_gt = pixel_downsampling(
        #     test_gt, 0.66, transfer_gt=train_gt,
        #     retained_class=LABEL_VALUES.index('Healthy'),
        #     sampling_mode='disjoint')

        # # Exp 5 b):
        # test_gt, train_gt = pixel_downsampling(
        #     test_gt, 0.66, transfer_gt=train_gt,
        #     retained_class=LABEL_VALUES.index('Healthy_Tissue'),
        #     sampling_mode='disjoint')
        # ============================================================
        # # Pre-processing for Liver dataset:

        # train_gt, _ = sample_gt(train_gt, 0.5,
        #                         mode='random',
        #                         cross_dict=cross_dict)

        # ============================================================

        # Display the split
        
        display_predictions(train_gt, viz, DATA_INFO,
                            caption="Train ground truth")
        display_predictions(test_gt, viz, DATA_INFO,
                            caption="Test ground truth")

        # Display dataset information
        print(trait)
        samples = np.count_nonzero(train_gt)
        total = np.count_nonzero(gt)
        print('\n{0}: {1}'.format(DATASET, DATASETS_CONFIG[DATASET]))
        print(trait)
        print("Image has dimensions {}x{} and {} channels".format(
            *img.shape))
        print("{} samples selected for training over {} ({}%)".format(
            samples, total, round(samples*100/total, 2)))
        """
        print('\n{0} classes, {1}'.format(len(LABEL_VALUES), LABEL_VALUES))
        data_stat(train_gt, LABEL_VALUES)
        print(trait)

        # # ML analysis
        # HSI_ml(img, gt, IGNORED_LABELS, LABEL_VALUES, palette,
        #        n_components=2, mode='LDA')
        # HSI_ml(img, gt, IGNORED_LABELS, LABEL_VALUES, palette,
        #        n_components=3, mode='PCA')"""

        # Train the model
        model, hyperparams, best_ckpt, MODEL_NAME = training_process(
            img, train_gt, hyperparams, IGNORED_LABELS,
            cross_dict, viz)

        # Test the model
        if cross_dict:
            if REPORT_MODE:
                if CHECKPOINT:
                    best_ckpt = load_cv_ckpt(MODEL_NAME,
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

            print("TESTING")
            cv_coord = cross_dict['cv_coord'][cross_dict['cv_step']]
            eval_img = np.copy(img[:, cv_coord[0]:cv_coord[1], :])
            prediction[:, cv_coord[0]:cv_coord[1]], proba[
                :, cv_coord[0]:cv_coord[1]] = inference(
                model, eval_img, best_ckpt, hyperparams)
            """
            ROC_info[cross_dict['cv_step']+1] = ROC_curve(
                proba[:, cv_coord[0]:cv_coord[1], :],
                test_gt[:, cv_coord[0]:cv_coord[1]],
                IGNORED_LABELS, LABEL_VALUES,
                EXP_NAME, palette, compact=False,
                plot_name='k{}'.format(1+cross_dict['cv_step']))"""

            if REPORT_MODE and VERBOSE:
                plt.show()
            else:
                plt.close('all')

        else:
            prediction, proba = inference(model, img, best_ckpt, hyperparams)

        #color_prediction = convert_to_color(prediction)
        run_results, error_map = compute_results(prediction,
                                                 test_gt,
                                                 
                                                 hyperparams,
                                                 iteration=run)

        # Display results and error
        viz.matplot(plt)
        #gt_err = np.copy(gt)
        gt_err = np.copy(gt)
        for x in IGNORED_LABELS:
            gt_err[gt_err == x] = 0
        """
        print("OKAY")
        #gt_d = convert_from_color_(gt_d,palette)
        #print(np.unique(gt_d))
        original_label = np.unique(gt)[1]
        print(original_label)
        mask_ =  np.zeros(gt_err.shape, dtype='bool')
        mask_[gt_err == 8] = True
        gt_err[mask_]= int(original_label)
        #gt_d = convert_to_color_(gt_d,palette)"""
        """
        display_error(disp_img,
                      prediction,
                      convert_to_color(error_map),
                      gt,
                      viz,
                      RGB_BANDS, DATA_INFO,
                      LABEL_VALUES, palette,
                      exp_name=EXP_NAME,
                      cross_dict=cross_dict,
                      gt=gt_err)"""
        print(prediction.shape)
        display_error_reg(disp_img,
                      prediction,
                      error_map,
                      viz,
                      DATA_INFO,
                      
                      exp_name=EXP_NAME,
                      cross_dict=cross_dict,
                      gt=gt_err)


        saving_results_analysis(prediction,gt_err,DATA_INFO,cross_dict,EXP_NAME,disp_img,RGB_BANDS)
        """plot_confusion_matrix(model, prediction, test_gt,
                                 display_labels=LABEL_VALUES,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
        cm  = run_results["Confusion matrix"].astype('float') / run_results["Confusion matrix"].sum(axis=1)[:, np.newaxis]
        cm =  np.round(cm, decimals=3)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=LABEL_VALUES)


        # NOTE: Fill all variables here with default values of the plot_confusion_matrix
        disp = disp.plot(include_values=True,
                 cmap=plt.cm.Blues, ax=None, xticks_rotation='horizontal')

        plt.show()"""
        """
        # Display overlay prediction and heatmap
        try:
            lbl_indx = LABEL_VALUES.index(ARGS.lbl_visu)
            pred_mask = prediction == lbl_indx
            gt_mask = gt == lbl_indx
            print(ARGS.lbl_visu)
            display_heatmap(disp_img, proba,
                            lbl_indx,
                            viz, RGB_BANDS,
                            DATA_INFO, cross_dict=cross_dict, alpha=0.8,
                            exp_name=EXP_NAME)
        except ValueError:
            print("Warning: '{}' is not a class label, all classes will be "
                  "overlayed.".format(ARGS.lbl_visu))
            gt_mask = gt != 0
            pred_mask = gt_mask

        display_overlay(disp_img,
                        color_prediction,
                        convert_to_color(gt),
                        gt_mask,
                        pred_mask,
                        viz,
                        RGB_BANDS, DATA_INFO,
                        palette,
                        cross_dict=cross_dict,
                        alpha=1, exp_name=EXP_NAME)

        results.append(run_results)
        print(trait)
        show_results(run_results, viz, EXP_NAME,
                     ignored_labels=IGNORED_LABELS,
                     label_values=LABEL_VALUES)

        if not cross_dict:
            _ = ROC_curve(proba, test_gt, IGNORED_LABELS, LABEL_VALUES,
                          EXP_NAME, palette, compact=False,
                          plot_name='overall')
            display_predictions(color_prediction, viz, DATA_INFO,
                                gt=convert_to_color(test_gt),
                                caption="Prediction vs. test ground truth")
        else:
            cross_dict['cv_step'] += 1
        torch.cuda.empty_cache()

    if cross_dict:
        _ = ROC_curve(proba, gt, IGNORED_LABELS, LABEL_VALUES,
                      EXP_NAME, palette, compact=False,
                      plot_name='overall')

        ROC_combined(ROC_info, EXP_NAME)

        color_prediction = convert_to_color(prediction)
        # run_results, error_map = compute_results(prediction,
        #                                          gt,
        #                                          LABEL_VALUES,
        #                                          hyperparams,
        #                                          iteration=run)
        # Display results and error
        viz.matplot(plt)
        # show_results(run_results, viz, EXP_NAME, label_values=LABEL_VALUES)
        display_predictions(color_prediction, viz, DATA_INFO,
                            gt=convert_to_color(gt),
                            caption="Prediction vs. test ground truth")

    if N_RUNS > 1:
        show_results(results, viz, EXP_NAME,
                     ignored_labels=IGNORED_LABELS,
                     label_values=LABEL_VALUES,
                     agregated=True)
    if VERBOSE:
        plt.show()
    else:
        plt.close('all')

    """
        cross_dict['cv_step'] += 1

if __name__ == '__main__':
    dataset_names = [v['name'] if 'name' in v.keys() else k for k,
                     v in DATASETS_CONFIG.items()]

    # Argument parser for CLI interaction
    parser = argparse.ArgumentParser(
        description="Run deep learning experiments on various hyperspectral"
        " datasets")
    parser.add_argument('--dataset', type=str, default=None,
                        choices=dataset_names,
                        help="Dataset to use.")
    parser.add_argument('--model', type=str, default=None,
                        help="Model to train. Available:\n"
                        "SVM (linear), "
                        "SVM_grid (grid search on linear, poly and "
                        "RBF kernels), "
                        "baseline (fully connected NN), "
                        "hu (1D CNN), "
                        "hamida (3D CNN + 1D classifier), "
                        "lee (3D FCN), "
                        "chen (3D CNN), "
                        "li (3D CNN), "
                        "he (3D CNN), "
                        "luo (3D CNN), "
                        "sharma (2D CNN), "
                        "boulch (1D semi-supervised CNN), "
                        "liu (3D semi-supervised CNN), "
                        "mou (1D RNN)")
    parser.add_argument('--folder', type=str, help="Folder where to store the "
                        "datasets (defaults to the current working directory)",
                        default="./Datasets/")
    parser.add_argument('--cuda', type=int, default=-1,
                        help="Specify CUDA device (defaults to -1, \
                        which learns on CPU)")
    parser.add_argument('--runs', type=int, default=1,
                        help="Number of runs (default: 1)")
    parser.add_argument('--kfolds', type=int, default=1,
                        help="Number of k folds for cross-validation\
                        (default: 1)")
    parser.add_argument('--restore', type=str, default=None,
                        help="Weights to use for initialization, \
                        e.g. a checkpoint path. For cross validation \
                        experiment, just specify the path to the folder \
                        containing  all ckpts.")
    parser.add_argument('--debug', action='store_true',
                        help="activate debug mode")
    parser.add_argument('--report', action='store_true',
                        help="activate report mode for cross validation")
    parser.add_argument('--verbose', action='store_true',
                        help="activate the plot display")
    parser.add_argument('--visdom_port', default=8097,
                        help="set specific port for visdom writer")
    parser.add_argument('--lbl_visu', default=None,
                        help="set specific label to generate the overlayed of\
                        the class prediction")

    # Dataset options
    group_dataset = parser.add_argument_group('Dataset')
    group_dataset.add_argument('--training_sample', type=float, default=10,
                               help="Percentage of samples to use for training\
                               (default: 10%%)")
    group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode\
    (random sampling or disjoint or cross_val, default: random)",
                               default='random')
    group_dataset.add_argument('--train_set', type=str, default=None,
                               help="Path to the train ground truth (optional,\
                               this supersedes the --sampling_mode option)")
    group_dataset.add_argument('--test_set', type=str, default=None,
                               help="Path to the test set (optional, by default the\
                               test_set is the entire ground truth minus the\
                               training)")

    # Training options
    group_train = parser.add_argument_group('Training')
    group_train.add_argument('--epoch', type=int,
                             help="Training epochs (optional, if absent will be\
                             set by the model)")
    group_train.add_argument('--patch_size', type=int,
                             help="Size of the spatial neighbourhood (optional, if\
                             absent will be set by the model)")
    group_train.add_argument('--learning_rate', type=float,
                             help="Learning rate, set by the model if not\
                             specified.")
    group_train.add_argument('--class_balancing', action='store_true',
                             help="Inverse median frequency class balancing\
                             (default = False)")
    group_train.add_argument('--batch_size', type=int,
                             help="Batch size (optional, if absent will be set by\
                             the model")
    group_train.add_argument('--test_stride', type=int, default=1,
                             help="Sliding window step stride during inference\
                             (default = 1)")
    group_train.add_argument('--regression', action='store_true',
                          help="regression task")
    group_train.add_argument('--reg_loss', default=None,
                          choices=['mae', 'mse'],
                          help="applying one of the regression loss")
                         

    # Data augmentation parameters
    group_da = parser.add_argument_group('Data augmentation')
    group_da.add_argument('--flip_augmentation', action='store_true',
                          help="Random flips (if patch_size > 1)")
    group_da.add_argument('--radiation_augmentation', action='store_true',
                          help="Random radiation noise (illumination)")
    group_da.add_argument('--mixture_augmentation', action='store_true',
                          help="Random mixes between spectra")
    group_da.add_argument('--normalized', action='store_true',
                          help="Normalize the hypercube values before the training\
                          (i.e. scale values between 0 & 1).")
    group_da.add_argument('--standardized', action='store_true',
                          help="Standardized the hypercube values before the training\
                          (i.e. mean=0 & std=1).")
    group_da.add_argument('--norm_thresh', type=float, default=None,
                          help="Threshold to be applied to the hypercube values before\
                          the normalization or after the standardization\
                          (default=5)")
    group_da.add_argument('--channel_norm', action='store_true',
                          help="Apply channel wised normalization on\
                          hypercube")
    group_da.add_argument('--class_norm', action='store_true',
                          help="Apply class wised normalization on\
                          hypercube")
    group_da.add_argument('--snv_norm', action='store_true',
                          help="Apply SNV normalization on\
                          hypercube")

    group_da.add_argument('--bd_downsampling', nargs='+', default=None,
                          type=int, help="Apply a band selection respecting \
                          the band range specified \
                          (i.e. 2 values between 0 & nb_bands)")
    group_da.add_argument('--pix_downsampling', action='store_true',
                          help="Apply a pixel downsampling")
    group_da.add_argument('--ds_rate', default=0.5,
                          help="Proportion of pixels to downsample")
    group_da.add_argument('--ds_set', default=None,
                          choices=['train', 'test'],
                          help="Downsampling is applied on the set specified,\
                          by default, downsampling is applied on both sets")
    group_da.add_argument('--ds_class', default=None,
                          help="Downsampling is applied on the class specified,\
                          by default, dowsampling is applied on all classes")
    group_da.add_argument('--ds_transfer', action='store_true',
                          help="Transfer the pixels retained during the \
                          downsampling to the other set.")
    group_da.add_argument('--ds_mode', default='random',
                          choices=['random', 'disjoint'],
                          help="Specify the downsampling mode")

    parser.add_argument('--with_exploration', action='store_true',
                        help="See data exploration visualization")
    parser.add_argument('--plot_scale', nargs='+', default=[0, 1],
                        help="Scale used to plot spectrums curves")
    parser.add_argument('--download', type=str, default=None, nargs='+',
                        choices=dataset_names,
                        help="Download the specified datasets and quits.")

    ARGS = parser.parse_args()

    CUDA_DEVICE = get_device(ARGS.cuda)

    # % of training samples
    SAMPLE_PERCENTAGE = ARGS.training_sample
    # Data augmentation ?
    FLIP_AUGMENTATION = ARGS.flip_augmentation
    RADIATION_AUGMENTATION = ARGS.radiation_augmentation
    MIXTURE_AUGMENTATION = ARGS.mixture_augmentation
    # Dataset name
    DATASET = ARGS.dataset
    # Model name
    MODEL = ARGS.model
    # Spatial context size (number of neighbours in each spatial direction)
    PATCH_SIZE = ARGS.patch_size
    # Add some visualization of the spectra ?
    DATAVIZ = ARGS.with_exploration
    PLOT_SCALE = [int(x) for x in ARGS.plot_scale]
    # Target folder to store/download/load the datasets
    FOLDER = ARGS.folder
    # Number of epochs to run
    EPOCH = ARGS.epoch
    # Sampling mode, e.g random sampling
    SAMPLING_MODE = ARGS.sampling_mode
    # Pre-computed weights to restore
    CHECKPOINT = ARGS.restore
    # Learning rate for the SGD
    LEARNING_RATE = ARGS.learning_rate
    # Automated class balancing
    CLASS_BALANCING = ARGS.class_balancing
    # Training ground truth file
    TRAIN_GT = ARGS.train_set
    # Testing ground truth file
    TEST_GT = ARGS.test_set
    TEST_STRIDE = ARGS.test_stride

    VERBOSE = ARGS.verbose
    REPORT_MODE = ARGS.report

    PIX_DS = dict()
    if ARGS.pix_downsampling:
        PIX_DS.update({
            'set': ARGS.ds_set,
            'rate': float(ARGS.ds_rate),
            'class': ARGS.ds_class,
            'mode': ARGS.ds_mode,
            'transfer': ARGS.ds_transfer})

    time_now = str(datetime.datetime.now()).split(' ')
    info_xp = [DATASET, MODEL, SAMPLING_MODE]

    EXP_NAME = '_'.join(info_xp)
    ENV_VISDOM = ' '.join(info_xp)
    # if not(REPORT_MODE):
    EXP_NAME += '_'.join(time_now)
    ENV_VISDOM += ' '.join(time_now)

    if ARGS.debug:
        EXP_NAME = 'test'
        ENV_VISDOM = 'test'

    main()
