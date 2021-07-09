from utils import open_file, class_fusion, read_mask, read_cube
import numpy as np
import torch
import numpy as np
import os
import pickle
import scipy.misc
import scipy.ndimage

CUSTOM_DATASETS_CONFIG = {
    'DFC2018_HSI': {
        'img': '2018_IEEE_GRSS_DFC_HSI_TR.HDR',
        'gt': '2018_IEEE_GRSS_DFC_GT_TR.tif',
        'download': False,
        'loader': lambda folder:
        dfc2018_loader(folder)},
    'Neck': {
        'download': False,
        'loader': lambda folder, dataset_conf:
        NeckTissue_loader(folder, dataset_conf)},
    'Abdomen': {
        'download': False,
        'loader': lambda folder, dataset_conf:
        AbdomenTissue_loader(folder, dataset_conf)},
    'AbdoNeck': {
        'download': False,
        'loader': lambda folder, dataset_conf:
        AbdoNeckTissue_loader(folder, dataset_conf)},
    'Leipzig_S': {
        'download': False,
        'loader': lambda folder, dataset_conf:
        LeipzigS_Tissue_loader(folder, dataset_conf)},
    'Leipzig_C': {
        'download': False,
        'loader': lambda folder, dataset_conf:
        LeipzigC_Tissue_loader(folder, dataset_conf)},
    'Leipzig_SC': {
        'download': False,
        'loader': lambda folder, dataset_conf:
        LeipzigFused_Tissue_loader(folder, dataset_conf)},
    'Liver': {
        'download': False,
        'loader': lambda folder, dataset_conf:
        LiverTissue_loader(folder, dataset_conf)},
    'Liver2': {
        'download': False,
        'loader': lambda folder, dataset_conf:
        Liver2Tissue_loader(folder, dataset_conf),
        'subsets': ['t_0', 't_150'],
        'corr_classInd': [1, 2],
        'corr_label': ['Unclassified', 't_0', 'Healthy']},
    'Laser': {
        'download': False,
        'loader': lambda folder, dataset_conf:
        LaserTissue_loader(folder, dataset_conf),
        'subsets': ['PATIENT_pig_1_ROI_8_30-08-2019',
            'PATIENT_pig_1_ROI_7_30-08-2019',
            'PATIENT_pig_1_ROI_6_30-08-2019',
            'PATIENT_pig_1_ROI_5_30-08-2019',
            'PATIENT_pig_1_ROI_2_30-08-2019',
            'PATIENT_pig_1_ROI_1_30-08-2019'
            ],
        # 'PATIENT_pig_1_ROI_roi3_30-08-2019',
                    # 'PATIENT_pig_1_ROI_roi2_30-08-2019'],
        'folders_order': ['35C', '60C', '70C', '80C', '90C', '100C', '110C',
                           't1', 't3', 't5','t4'],
        #'threshold': '55-65'
        }

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


def NeckTissue_loader(folder, DATASETS_CONFIG):
    img_tmpl = '{0}_cube.pkl'
    mask_tmpl = '{0}_mask.pkl'
    images = []
    masks = []
    img_name = []

    folders = os.listdir(folder)
    # folders.sort()

    folders = ['2020_02_13_12_01_27',
               '2020_02_10_11_56_06',
               '2020_02_13_12_30_31',
               '2020_02_10_12_32_57',
               '2020_02_11_12_07_02',
               '2020_02_12_11_51_39',
               '2020_02_12_12_23_00',
               '2020_02_11_12_38_57']
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

    rgb_bands = [40, 15, 0]

    label_values = [x.split(' ')[-1].replace('\n', '') for x in open(
        os.path.join(folder, 'Neck_list.txt'), 'r')]
    # os.path.join(folder, 'neck_class.txt'), 'r')]

    ignored_labels = [0]
    fused_class = []

    # # Hierarchy classification(Artery - Fat - Nerve - Vein):
    # ignored_labels = [0, 2, 3, 4, 6, 7, 9, 10, 11]

    # # Hierarchy classification(Artery - Nerve - Vein):
    # ignored_labels = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11]

    # Exp Neck:
    # # set new camera:

    ignored_labels = [0, 2, 3, 4, 7, 6, 10, 11]

    # fused_class = ['Nerve', 'Artery', 'Vein']

    if fused_class:
        gt, ignored_labels, label_values = class_fusion(
            fused_class, gt, ignored_labels, label_values)

    info_data = {'Neck': {'info_data': {
        'folder': folder,
        'cube_shape': list(np.shape(img))[:2],
        'img_shape': list(shape)[:2],
        'img_name': img_name,
        'fused_class': fused_class}}}
    DATASETS_CONFIG.update(info_data)

    return img, gt, rgb_bands, ignored_labels, label_values


def AbdomenTissue_loader(folder, DATASETS_CONFIG):
    img_tmpl = '{0}_cube.pkl'
    mask_tmpl = '{0}_mask.pkl'
    images = []
    masks = []
    img_name = []

    folders = os.listdir(folder)
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

    rgb_bands = [40, 15, 0]

    label_values = [x.split(' ')[-1].replace('\n', '') for x in open(
        os.path.join(folder, 'abdomen_class.txt'), 'r')]

    ignored_labels = [0]
    fused_class = []

    ignored_labels = [0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16,
                      17, 18, 20, 22, 23, 25]

    fused_class = ['Adrenal', 'Bladder', 'Burnt_liver',
                   'Clean_green_gauze', 'Colon', 'Drape',
                   'Duodenum', 'Gallbladder', 'Glove', 'Human_skin',
                   'Kidney', 'Liver', 'Lymphnodes', 'Metal',
                   'Pancreas', 'Skin', 'Stomach', 'Surrenal_gland', 'Spleen']

    if fused_class:
        gt, ignored_labels, label_values = class_fusion(
            fused_class, gt, ignored_labels, label_values)

    info_data = {'Abdomen': {'info_data': {
        'folder': folder,
        'cube_shape': list(np.shape(img))[:2],
        'img_shape': list(shape)[:2],
        'img_name': img_name,
        'fused_class': fused_class}}}
    DATASETS_CONFIG.update(info_data)

    return img, gt, rgb_bands, ignored_labels, label_values


def AbdoNeckTissue_loader(folder, DATASETS_CONFIG):
    img_tmpl = '{0}_cube.pkl'
    mask_tmpl = '{0}_mask.pkl'
    images = []
    masks = []
    img_name = []

    folders = os.listdir(folder)
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

    rgb_bands = [40, 15, 0]

    label_values = [x.split(' ')[-1].replace('\n', '') for x in open(
        os.path.join(folder, 'AbdoNeck_list.txt'), 'r')]

    ignored_labels = [0, 1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16,
                      17, 18, 21, 23, 25, 26, 27, 28, 30]

    fused_class = ['Unclassified', 'Adrenal', 'Bladder', 'Burnt_liver',
                   'Cartilage', 'Clean_green_gauze', 'Colon', 'Drape',
                   'Duodenum', 'Gallbladder', 'Glove', 'Human_skin',
                   'Kidney', 'Liver', 'Lymphnodes', 'Metal', 'Pancreas',
                   'Skin', 'Stomach', 'Surrenal_gland', 'Thymus',
                   'Thyroid', 'Spleen']

    if fused_class:
        gt, ignored_labels, label_values = class_fusion(
            fused_class, gt, ignored_labels, label_values)

    info_data = {'AbdoNeck': {'info_data': {
        'folder': folder,
        'cube_shape': list(np.shape(img))[:2],
        'img_shape': list(shape)[:2],
        'img_name': img_name,
        'fused_class': fused_class}}}
    DATASETS_CONFIG.update(info_data)

    return img, gt, rgb_bands, ignored_labels, label_values


def LeipzigS_Tissue_loader(folder, DATASETS_CONFIG):
    img_tmpl = '{0}_cube.pkl'
    mask_tmpl = '{0}_mask.pkl'
    images = []
    masks = []
    img_name = []

    # folders = os.listdir(folder)
    folders = ['2019_04_30_15_34_56', '2019_10_17_15_54_04',
               '2019_10_23_14_33_05', '2019_08_23_11_46_45',
               '2019_07_04_12_00_42', '2019_08_23_16_38_54',
               '2019_08_14_13_06_08', '2019_10_25_11_43_32',
               '2019_08_26_13_51_35', '2020_03_09_17_31_24']

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

    rgb_bands = [40, 15, 0]

    label_values = [x.split(' ')[-1].replace('\n', '') for x in open(
        os.path.join(folder, 'Leipzig_S_list.txt'), 'r')]

    ignored_labels = [0]
    fused_class = {}

    # Iss46_model1-3:
    ignored_labels = [0, 4]
    fused_class = {'Healthy_Tissue': ['Stomach', 'Esophagus']}

    # Iss46_model2-4:
    ignored_labels = [0]
    fused_class = {'Healthy_Tissue': ['Stomach', 'Esophagus'],
                   'Tumor_M': ['Tumor', 'Margin']}

    for name, group in fused_class.items():
        gt, ignored_labels, label_values = class_fusion(
            group, gt, ignored_labels, label_values,
            class_name=name)

    info_data = {'Leipzig_S': {'info_data': {
        'folder': folder,
        'cube_shape': list(np.shape(img))[:2],
        'img_shape': list(shape)[:2],
        'img_name': img_name,
        'fused_class': fused_class}}}

    DATASETS_CONFIG.update(info_data)

    return img, gt, rgb_bands, ignored_labels, label_values


def LeipzigC_Tissue_loader(folder, DATASETS_CONFIG):
    img_tmpl = '{0}_cube.pkl'
    mask_tmpl = '{0}_mask.pkl'
    images = []
    masks = []
    img_name = []

    # folders = os.listdir(folder)
    folders = ['2019_09_04_12_43_40', '2019_07_25_11_56_38',
               '2019_07_17_15_38_14', '2019_08_12_10_52_33',
               '2019_08_28_14_00_03', '2019_07_12_11_14_41',
               '2019_08_23_12_15_26', '2019_08_15_13_08_58',
               '2019_08_09_12_17_55', '2019_07_15_11_33_28']
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

    rgb_bands = [40, 15, 0]

    label_values = [x.split(' ')[-1].replace('\n', '') for x in open(
        os.path.join(folder, 'Leipzig_C_list.txt'), 'r')]
    # os.path.join(folder, 'neck_class.txt'), 'r')]

    # Exp 1:
    ignored_labels = [0, 3]
    fused_class = []

    if fused_class:
        gt, ignored_labels, label_values = class_fusion(
            fused_class, gt, ignored_labels, label_values)

    info_data = {'Leipzig_C': {'info_data': {
        'folder': folder,
        'cube_shape': list(np.shape(img))[:2],
        'img_shape': list(shape)[:2],
        'img_name': img_name,
        'fused_class': fused_class}}}

    # # band downsampling
    # bd_range = [30, 100]
    # info_data['Leipzig_C']['info_data'].update({
    #     'band_downsampling': bd_range})
    # img = band_downsampling(img, bd_range)

    DATASETS_CONFIG.update(info_data)
    return img, gt, rgb_bands, ignored_labels, label_values


def LeipzigFused_Tissue_loader(folder, DATASETS_CONFIG):
    img_tmpl = '{0}_cube.pkl'
    mask_tmpl = '{0}_mask.pkl'
    images = {'Colon': [], 'Stomach': []}
    masks = {'Colon': [], 'Stomach': []}
    img_names = {'Colon': [], 'Stomach': []}

    folder_C = os.path.join(folder, 'Colon')
    folder_S = os.path.join(folder, 'Stomach')
    folder_SC = [folder_C, folder_S]
    datasets = ['Colon', 'Stomach']

    label_values_C = [x.split(' ')[-1].replace('\n', '') for x in open(
        os.path.join(folder_C, 'Leipzig_C_list.txt'), 'r')]
    label_values_S = [x.split(' ')[-1].replace('\n', '') for x in open(
        os.path.join(folder_S, 'Leipzig_S_list.txt'), 'r')]
    # os.path.join(folder, 'neck_class.txt'), 'r')]

    label_values = label_values_C + label_values_S

    lbl_offset = [0, len(label_values_C)]

    fix = [['2019_09_04_12_43_40', '2019_07_25_11_56_38',
            '2019_07_17_15_38_14', '2019_08_12_10_52_33',
            '2019_08_28_14_00_03', '2019_07_12_11_14_41',
            '2019_08_23_12_15_26', '2019_08_15_13_08_58',
            '2019_08_09_12_17_55', '2019_07_15_11_33_28'],
           ['2019_04_30_15_34_56', '2019_10_17_15_53_04',
            '2019_10_23_14_33_05', '2019_08_23_11_46_45',
            '2019_07_04_12_00_42', '2019_08_23_16_38_54',
            '2019_08_14_13_06_08', '2019_10_25_11_43_32',
            '2019_08_26_13_51_35', '2020_03_09_17_31_24']]

    w = 0
    for fold, dataset, offset in zip(folder_SC, datasets, lbl_offset):
        # folders = os.listdir(fold)
        folders = fix[w]
        w += 1
        for i, acquisition in enumerate(folders):
            try:
                acquisition_path = os.path.join(fold, acquisition,
                                                img_tmpl.format(acquisition))
                cube = pickle.load(open(acquisition_path, 'rb'))
                images[dataset].append(cube)
                mask_path = os.path.join(
                    fold, acquisition, mask_tmpl.format(acquisition))
                mask = pickle.load(open(mask_path, 'rb'))

                cl_mask = mask[mask != 0]
                mask[mask != 0] = cl_mask+offset*np.ones(cl_mask.shape)
                masks[dataset].append(mask)
                img_names[dataset].append(acquisition)
            except NotADirectoryError:
                pass

    # form 2 single matrices: img and gt
    shape = np.shape(images[datasets[0]][0])
    nb_img = len(images[datasets[0]]) + len(images[datasets[1]])

    new_shape = (shape[0], shape[1]*nb_img, shape[2])

    img = np.zeros(new_shape)
    gt = np.zeros(new_shape[:2])
    img_name = []
    count = 0
    for i in range(nb_img//2):
        for dataset in datasets:
            img[:, count*shape[1]:(count+1)*shape[1], :] = images[dataset][i]
            gt[:, count*shape[1]:(count+1)*shape[1]] = np.fliplr(
                masks[dataset][i].T)
            img_name.append(os.path.join(dataset, img_names[dataset][i]))
            count += 1

    rgb_bands = [40, 15, 0]

    # Exp 3:
    ignored_labels = [0, 3, 4, 8]
    fused_class = {'Healthy_f': ['Healthy_C', 'Stomach', 'Esophagus'],
                   'Tumor_f': ['Tumor_C', 'Tumor_S']}

    # # Exp 4:
    # ignored_labels = [0, 2, 3, 4, 7, 8]
    # fused_class = {'Healthy_S': ['Stomach', 'Esophagus']}

    if fused_class:
        for name, group in fused_class.items():
            gt, ignored_labels, label_values = class_fusion(
                group, gt, ignored_labels, label_values,
                class_name=name)

    info_data = {'Leipzig_SC': {'info_data': {
        'folder': folder,
        'cube_shape': list(np.shape(img))[:2],
        'img_shape': list(shape)[:2],
        'img_name': img_name,
        'fused_class': fused_class}}}

    DATASETS_CONFIG.update(info_data)

    return img, gt, rgb_bands, ignored_labels, label_values


def LiverTissue_loader(folder, DATASETS_CONFIG):
    img_tmpl = '{0}_cube.pkl'
    mask_tmpl = '{0}_mask.pkl'
    phase1_mask = '{0}_mas.pkl'
    images = []
    masks = []
    pha1_masks = []
    img_name = []

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
            
            pha1_mask_path =  os.path.join(
                folder, acquisition, phase1_mask.format(acquisition))
            pha1_mask = pickle.load(open(pha1_mask_path, 'rb'))
            print(np.unique(pha1_mask))
            pha1_masks.append(pha1_mask)
            img_name.append(acquisition)
        except NotADirectoryError:
            pass

    # form 2 single matrices: img and gt
    shape = np.shape(images[0])
    nb_img = len(images)
    new_shape = (shape[0], shape[1]*nb_img, shape[2])

    img = np.zeros(new_shape)
    gt = np.zeros(new_shape[:2])
    gt1 = np.zeros(new_shape[:2])

    for i in range(nb_img):
        img[:, i*shape[1]:(i+1)*shape[1], :] = images[i]
        gt[:, i*shape[1]:(i+1)*shape[1]] = np.fliplr(masks[i].T)
        gt1[:, i*shape[1]:(i+1)*shape[1]] = pha1_masks[i]
    rgb_bands = [40, 15, 0]

    label_values = [x.split(' ')[-1].replace('\n', '') for x in open(
        os.path.join(folder, 'Liver_list.txt'), 'r')]

    ignored_labels = [0,5,6,7,8,9,10]
    #ignored_labels = [0,10]
    fused_class = dict()

    # # Exp 2:
    fused_class = {'t_30-60-90': ['t_30', 't_60','t_90']}
    #fused_class = {'t_90-390': [ 't_90','t_390']}

    if fused_class:
        for name, group in fused_class.items():
            gt, ignored_labels, label_values = class_fusion(
                group, gt, ignored_labels, label_values,
                class_name=name)

    info_data = {'Liver': {'info_data': {
        'folder': folder,
        'cube_shape': list(np.shape(img))[:2],
        'img_shape': list(shape)[:2],
        'img_name': img_name,
        'fused_class': fused_class}}}
    DATASETS_CONFIG.update(info_data)
    print("###############################################################")
    print(img.shape)
    #img = scipy.ndimage.median_filter(img, size=3)
    return img, gt, rgb_bands, ignored_labels, label_values,gt1


def Liver2Tissue_loader(folder, DATASETS_CONFIG):
    img_tmpl = '{0}_cube.pkl'
    mask_tmpl = '{0}_mask.pkl'

    subsets = DATASETS_CONFIG['Liver2']['subsets']
    lbl_correction = DATASETS_CONFIG['Liver2']['corr_classInd']

    images, masks, img_names = {}, {}, {}
    folder_subsets = []
    for subset in subsets:
        images.update({subset: []})
        masks.update({subset: []})
        img_names.update({subset: []})
        folder_subsets.append(os.path.join(folder, subset))

    label_values = DATASETS_CONFIG['Liver2']['corr_label']
    if label_values is None:
        label_values = [x.split(' ')[-1].replace('\n', '') for x in open(
            os.path.join(folder, 'Liver_list.txt'), 'r')]

    # fix = [['2019_09_04_12_43_40', '2019_07_25_11_56_38',
    #         '2019_07_17_15_38_14', '2019_08_12_10_52_33',
    #         '2019_08_28_14_00_03', '2019_07_12_11_14_41',
    #         '2019_08_23_12_15_26', '2019_08_15_13_08_58',
    #         '2019_08_09_12_17_55', '2019_07_15_11_33_28'],
    #        ['2019_04_30_15_34_56', '2019_10_17_15_53_04',
    #         '2019_10_23_14_33_05', '2019_08_23_11_46_45',
    #         '2019_07_04_12_00_42', '2019_08_23_16_38_54',
    #         '2019_08_14_13_06_08', '2019_10_25_11_43_32',
    #         '2019_08_26_13_51_35', '2020_03_09_17_31_24']]

    w = 0
    for j, (fold, subset) in enumerate(zip(folder_subsets, subsets)):
        folders = os.listdir(fold)
        # folders = fix[w]
        w += 1
        for i, acquisition in enumerate(folders):
            try:
                acquisition_path = os.path.join(fold, acquisition,
                                                img_tmpl.format(acquisition))
                cube = pickle.load(open(acquisition_path, 'rb'))
                images[subset].append(cube)
                mask_path = os.path.join(
                    fold, acquisition, mask_tmpl.format(acquisition))
                mask = pickle.load(open(mask_path, 'rb'))

                # correct the class index
                if lbl_correction:
                    true_ind = lbl_correction[j]
                    cl_mask = mask[mask != 0]
                    mask[mask != 0] = true_ind*np.ones(cl_mask.shape)

                masks[subset].append(mask)
                img_names[subset].append(acquisition)
            except NotADirectoryError:
                pass

    # form 2 single matrices: img and gt
    shape = np.shape(images[subsets[0]][0])
    nb_img = len(images[subsets[0]]) + len(images[subsets[1]])

    new_shape = (shape[0], shape[1]*nb_img, shape[2])

    img = np.zeros(new_shape)
    gt = np.zeros(new_shape[:2])
    img_name = []
    count = 0
    for i in range(nb_img//2):
        for subset in subsets:
            img[:, count*shape[1]:(count+1)*shape[1], :] = images[subset][i]
            gt[:, count*shape[1]:(count+1)*shape[1]] = np.fliplr(
                masks[subset][i].T)
            img_name.append(os.path.join(subset, img_names[subset][i]))
            count += 1

    rgb_bands = [40, 15, 0]

    ignored_labels = [0]
    fused_class = {}

    if fused_class:
        for name, group in fused_class.items():
            gt, ignored_labels, label_values = class_fusion(
                group, gt, ignored_labels, label_values,
                class_name=name)

    info_data = {'Liver2': {'info_data': {
        'folder': folder,
        'cube_shape': list(np.shape(img))[:2],
        'img_shape': list(shape)[:2],
        'img_name': img_name,
        'fused_class': fused_class}}}

    DATASETS_CONFIG.update(info_data)

    print("class indices", np.unique(gt))
    return img, gt, rgb_bands, ignored_labels, label_values


def LaserTissue_loader(folder, DATASETS_CONFIG):
    annot_dir = 'reg'

    subsets = DATASETS_CONFIG['Laser']['subsets']
    #deg_thresh = DATASETS_CONFIG['Laser']['threshold']

    images, masks, img_names = {}, {}, {}
    folder_subsets = []
    for subset in subsets:
        images.update({subset: []})
        masks.update({subset: []})
        img_names.update({subset: []})
        folder_subsets.append(os.path.join(folder, subset))

    #label_info = [x.replace('\n', '').split(' ') for x in open(
    #    os.path.join(folder, 'class_index', 'th{}_list.txt'.format(
    #        deg_thresh)), 'r')]
    #for i, (_, _, color) in enumerate(label_info):
    #    color = color.replace('[', '').replace(']', '')
    #    label_info[i][2] = np.array([int(x) for x in list(color.split(','))])

    nb_img = 0
    for j, (fold, subset) in enumerate(zip(folder_subsets, subsets)):
        folders = DATASETS_CONFIG['Laser']['folders_order']
        for i, acquisition in enumerate(folders):
            acq_folder = os.path.join(fold, acquisition)
            #print(acq_folder)

            try:
                list_acq = [f for f in os.listdir(acq_folder) if f.endswith(
                    '.dat')]
                acquisition_path = os.path.join(acq_folder, list_acq[0])

                mask_folder = os.path.join(fold, acquisition,
                                          annot_dir)
                list_png = [f for f in os.listdir(mask_folder) if f.endswith(
                    '.png')]

                mask_path = os.path.join(mask_folder, list_png[0])
                print(acquisition_path)

                cube = read_cube(acquisition_path)
                print(cube.shape)
                mask = read_mask(mask_path)#, label_info)
                images[subset].append(cube)
                masks[subset].append(mask)
                img_names[subset].append(acquisition)
                nb_img += 1
            except NotADirectoryError:
                print("Warning: '{}' directory was not found and would be"
                      " skipped".format(acq_folder))
                pass
            except FileNotFoundError:
                print("Warning: '{}' mask or cube was not found and would be"
                      " skipped".format(acq_folder))
                pass

    # form 2 single matrices: img and gt
    shape = np.shape(images[subsets[0]][0])
    new_shape = (shape[0], shape[1]*nb_img, shape[2])

    #label_values = [x[1] for x in label_info]
    img = np.zeros(new_shape)
    gt = np.zeros(new_shape[:2])
    img_name = []
    count = 0

    for subset in subsets:
        for i in range(len(img_names[subset])):
            rot_mask = np.fliplr(masks[subset][i].T)
            print("THE SHAPE OF IMAGE AND MASK")
            print(images[subset][i].shape)
            print(rot_mask.shape)
            img[:, count*shape[1]:(count+1)*shape[1], :] = images[subset][i]
            gt[:, count*shape[1]:(count+1)*shape[1]] = rot_mask
            assert rot_mask.shape[:2] == images[subset][i].shape[:2], \
                'Error: hypercube and mask shapes are not matching.'

            img_name.append(os.path.join(subset, img_names[subset][i]))
            count += 1

    rgb_bands = [40, 15, 0]

    ignored_labels = [0]
    fused_class = {}

    if fused_class:
        for name, group in fused_class.items():
            gt, ignored_labels, label_values = class_fusion(
                group, gt, ignored_labels, label_values,
                class_name=name)

    info_data = {'Laser': {'info_data': {
        'folder': folder,
        'cube_shape': list(np.shape(img))[:2],
        'img_shape': list(shape)[:2],
        'img_name': img_name,
        'fused_class': fused_class,
        'subsets': subsets}}}

    DATASETS_CONFIG.update(info_data)

    return img, gt, rgb_bands, ignored_labels#, label_values

