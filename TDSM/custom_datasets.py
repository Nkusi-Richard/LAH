import os
import pickle
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import open_file, class_fusion, read_mask, read_cube
from PIL import Image

DATASETS_CONFIG = {
    'DFC2018_HSI': {
        'img': '2018_IEEE_GRSS_DFC_HSI_TR.HDR',
        'gt': '2018_IEEE_GRSS_DFC_GT_TR.tif',
        'download': False,
        'loader': lambda folder:
        dfc2018_loader(folder)},
    'Neck': {
        'loader': lambda loading_config, dataset_conf:
        classic_loader('Neck', loading_config, dataset_conf)},
    'Abdomen': {
        'loader': lambda loading_config, dataset_conf:
        classic_loader('Abdomen', loading_config, dataset_conf)},
    'Leipzig_S': {
        'loader': lambda loading_config, dataset_conf:
        classic_loader('Leipzig_S', loading_config, dataset_conf)},
    'Leipzig_C': {
        'loader': lambda loading_config, dataset_conf:
        classic_loader('Leipzig_C', loading_config, dataset_conf)},
    'Leipzig_SC': {
        'loader': lambda loading_config, dataset_conf:
        subsets_loader('Leipzig_SC', loading_config, dataset_conf)},
    'Liver': {
        'loader': lambda loading_config, dataset_conf:
        classic_loader('Liver', loading_config, dataset_conf)},
    'Laser': {
        'download': False,
        'loader': lambda loading_config, dataset_conf:
        subsets_loader('Laser', loading_config, dataset_conf)}
}


def dfc2018_loader(folder):
    img = open_file(folder + '2018_IEEE_GRSS_DFC_HSI_TR.HDR')[:, :, :-2]
    gt = open_file(folder + '2018_IEEE_GRSS_DFC_GT_TR.tif')
    gt = gt.astype('uint8')

    rgb_bands = (47, 31, 15)

    label_values = ["Unclassified",
                    "Healthy grass",
                    "Stressed grass",
                    "Artificial turf",
                    "Evergreen trees",
                    "Deciduous trees",
                    "Bare earth",
                    "Water",
                    "Residential buildings",
                    "Non-residential buildings",
                    "Roads",
                    "Sidewalks",
                    "Crosswalks",
                    "Major thoroughfares",
                    "Highways",
                    "Railways",
                    "Paved parking lots",
                    "Unpaved parking lots",
                    "Cars",
                    "Trains",
                    "Stadium seats"]
    ignored_labels = [0]
    return img, gt, rgb_bands, ignored_labels, label_values


def classic_loader(dataset, loading_config, dataset_config):
    """
    Data loader, load multiple hypercubes and their associated labels
    to form one global hyperspectral image with the corresponding
    annotation mask
    Args:
    - dataset (str): dataset name
    - loading_config (dict): dictionary containing loading specification
    from .yml configuration file
    - dataset_config (dict): dictionary containing dataset information
    Returns:
    -img (np.array): 3D hyperspectral image (WxHxB)
    -gt (np.array): 2D int array of labels (WxH)
    -label_values (list[str]): list of class names
    -ignored_labels (list[int]): list of int classes to ignore
    """
    folder = loading_config['folder']
    img_tmpl = loading_config['img_tmpl']
    mask_tmpl = loading_config['mask_tmpl']
    images = []
    masks = []
    img_name = []

    if loading_config['file_order']:
        folders = loading_config['file_order']
    else:
        folders = os.listdir(folder)
        folders.sort()

    for i, acquisition in enumerate(folders):
        try:
            acquisition_path = os.path.join(folder, acquisition,
                                            img_tmpl.format(acquisition))
            cube = pickle.load(open(acquisition_path, 'rb'))
            images.append(cube)
            mask_path = os.path.join(
                folder, acquisition, mask_tmpl.format(acquisition))
            mask = pickle.load(open(mask_path, 'rb'))
            masks.append(mask)
            img_name.append(acquisition)
        except NotADirectoryError:
            pass

    # form 2 single matrices: img and gt
    shape = np.shape(images[0])
    nb_img = len(images)
    new_shape = (shape[0], shape[1]*nb_img, shape[2])

    img = np.zeros(new_shape)
    gt = np.zeros(new_shape[:2])

    for i in range(nb_img):
        img[:, i*shape[1]:(i+1)*shape[1], :] = images[i]
        gt[:, i*shape[1]:(i+1)*shape[1]] = np.fliplr(masks[i].T)

    label_values = [x.split(' ')[-1].replace('\n', '') for x in open(
        loading_config['class_index'], 'r')]

    ignored_classes = loading_config['ignored_labels']
    if ignored_classes:
        try:
            ignored_labels = [
                label_values.index(x) for x in ignored_classes]
        except ValueError:
            print("Error: One class label was not recognized, please check the"
                  " labels you need to ignore in the configuration file.")
            exit()

    fused_class = loading_config['fused_labels']
    if fused_class:
        for name, group in fused_class.items():
            gt, ignored_labels, label_values = class_fusion(
                group, gt, ignored_labels, label_values,
                class_name=name)

    info_data = {dataset: {'info_data': {
        'folder': folder,
        'cube_shape': list(np.shape(img))[:2],
        'img_shape': list(shape)[:2],
        'img_name': img_name,
        'fused_class': fused_class}}}
    dataset_config.update(info_data)

    return img, gt, ignored_labels, label_values


def subsets_loader(dataset_name, loading_config, dataset_conf):
    """
    Data loader, load multiple hypercubes and their associated labels
    to form one global hyperspectral image with the corresponding
    annotation mask
    Args:
    - dataset (str): dataset name
    - loading_config (dict): dictionary containing loading specification
    from .yml configuration file
    - dataset_config (dict): dictionary containing dataset information
    Returns:
    -img (np.array): 3D hyperspectral image (WxHxB)
    -gt (np.array): 2D int array of labels (WxH)
    -label_values (list[str]): list of class names
    -ignored_labels (list[int]): list of int classes to ignore
    """
    folder = loading_config['folder']
    img_tmpl = loading_config['img_tmpl']
    mask_tmpl = loading_config['mask_tmpl']
    subsets = dataset_conf[dataset_name]['subsets']
    shuffle_subsets = dataset_conf[dataset_name]['shuffle_subsets']
    annot_dir = dataset_conf[dataset_name]['annotation']
    raw_data = dataset_conf[dataset_name]['raw_data']

    images, masks, img_names = {}, {}, {}
    folder_subsets = []
    for subset in subsets:
        images.update({subset: []})
        masks.update({subset: []})
        img_names.update({subset: []})
        folder_subsets.append(os.path.join(folder, subset))

    # Load label names and colors (optional) generate a global index:
    class_index_paths = loading_config['class_index']
    if isinstance(class_index_paths, str):
        class_index_paths = [class_index_paths]

    if len(class_index_paths) == 1:
        label_info = [x.replace('\n', '').split(' ') for x in open(
            class_index_paths[0], 'r')]
        lbl_offset = [0] * len(subsets)
    else:
        label_info = []
        lbl_offset = [0]
        for path in class_index_paths:
            subset_info = [x.split(' ').replace('\n', '') for x in open(
                path, 'r')]
            label_info += [
                    [x[0]+lbl_offset[-1], x[1], x[2]] for x in subset_info]
            lbl_offset.append(lbl_offset[-1] + len(subset_info))

    if raw_data:
        for i, (_, _, color) in enumerate(label_info):
            color = color.replace('[', '').replace(']', '')
            label_info[i][2] = np.array(
                [int(x) for x in list(color.split(','))])

    # Generate subset samples order:
    if loading_config['file_order']:
        folders = loading_config['file_order']
        if isinstance(folders[0], str):
            folders = [folders] * len(subsets)
    else:
        folders = []
        for folder in folder_subsets:
            folders.append(os.listdir(folder))

    # Load subset samples separately:
    nb_img = 0
    for w, (fold, subset, offset) in enumerate(
            zip(folder_subsets, subsets, lbl_offset)):
        print(folders[w])
        for i, acquisition in enumerate(folders[w]):
            acq_folder = os.path.join(fold, acquisition)
            try:
                # Load raw data
                if raw_data:
                    # List acq. since there is no template name for raw data
                    list_acq = [f for f in os.listdir(
                        acq_folder) if f.endswith('.dat')]
                    acquisition_path = os.path.join(acq_folder, list_acq[0])
                    mask_folder = os.path.join(acq_folder, annot_dir)
                    list_acq_ = [f for f in os.listdir(
                        mask_folder) if f.endswith('.png')]
                    list_acq_1 = [f for f in os.listdir(
                        acq_folder) if f.endswith('.png')]
                    acquisition_mask = os.path.join(acq_folder, list_acq_1[0])

                    #mask_path = os.path.join(mask_folder, os.path.basename(
                    #    acquisition_path.replace('dat', 'png')))
                    mask_path = os.path.join(mask_folder, list_acq_[0])
                    cube = read_cube(acquisition_path)
                    img1 = np.asarray(Image.open(acquisition_mask))
                    mask = read_mask(mask_path, label_info)
                    #print(mask.shape)
                    #print(img1.shape)
                    mask[img1 == 0] = 0
                    #mask1 = Image.fromarray(mask)
                    #mask1.show()
                    #mask[mask]

                # Load prepared data
                else:
                    acquisition_path = os.path.join(
                        acq_folder, img_tmpl.format(acquisition))
                    cube = pickle.load(open(acquisition_path, 'rb'))
                    mask_path = os.path.join(acq_folder, annot_dir,
                                             mask_tmpl.format(acquisition))
                    mask = pickle.load(open(mask_path, 'rb'))
                    cl_mask = mask[mask != 0]
                    mask[mask != 0] = cl_mask+offset*np.ones(cl_mask.shape)

                images[subset].append(cube)
                masks[subset].append(mask)
                img_names[subset].append(os.path.join(subset, acquisition))
                nb_img += 1
            except NotADirectoryError:
                print("Warning: '{}' directory was not found and will be"
                      " skipped".format(acq_folder))
                pass
            except FileNotFoundError:
                print("Warning: '{}' mask or cube was not found and will be"
                      " skipped".format(acq_folder))
                pass

    # shuffle or concatenate subsets
    if shuffle_subsets:
        # Interleave the following lists:
        # 1. Extract the value of the dictionaries
        # 2. Depackage these as an argument to 'zip_longest'
        # 3. 'zip_longest' generates a tuple iteration filled with
        # None for missing elements
        # 4. 'chain' concatenate those tuple into a list
        # 5. Iteration over the generated list to remove None elements
        img_name = [x for x in itertools.chain(*itertools.zip_longest(
            *list(img_names.values()))) if x is not None]
        images_list = [x for x in itertools.chain(*itertools.zip_longest(
            *list(images.values()))) if x is not None]
        mask_list = [x for x in itertools.chain(*itertools.zip_longest(
            *list(masks.values()))) if x is not None]
    else:
        img_name, images_list, mask_list = [], [], []
        for subset in subsets:
            img_name += img_names[subset]
            images_list += images[subset]
            mask_list += masks[subset]

    # form 2 single matrices: img and gt
    shape = np.shape(images[subsets[0]][0])
    new_shape = (shape[0], shape[1]*nb_img, shape[2])
    img = np.zeros(new_shape)
    gt = np.zeros(new_shape[:2])
    for i, (hsi, mask) in enumerate(zip(images_list, mask_list)):
        img[:, i*shape[1]:(i+1)*shape[1], :] = hsi
        gt[:, i*shape[1]:(i+1)*shape[1]] = np.fliplr(mask.T)

    # Generate ignored class indexes
    label_values = [x[1] for x in label_info]
    ignored_classes = loading_config['ignored_labels']
    if ignored_classes:
        try:
            ignored_labels = [
                label_values.index(x) for x in ignored_classes]
        except ValueError:
            print("Error: One class label was not recognized, please check the"
                  " labels you need to ignore in the configuration file.")
            exit()

    # Fuse classes and associated indexes
    fused_class = loading_config['fused_labels']
    if fused_class:
        for name, group in fused_class.items():
            gt, ignored_labels, label_values = class_fusion(
                group, gt, ignored_labels, label_values,
                class_name=name)

    # Update configuration dictionary
    info_data = {dataset_name: {'info_data': {
        'folder': folder,
        'cube_shape': list(np.shape(img))[:2],
        'img_shape': list(shape)[:2],
        'img_name': img_name,
        'fused_class': fused_class,
        'subsets': subsets}}}
    dataset_conf.update(info_data)

    print("OKKKKKKKKKKKKKKKKKKKKKK")
    print(np.unique(gt))
    return img, gt, ignored_labels, label_values
