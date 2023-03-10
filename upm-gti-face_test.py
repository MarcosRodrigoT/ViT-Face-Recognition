import os
import re
import pickle
import numpy as np
import tensorflow as tf
from vit_keras import vit
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc


def get_label(file_path):
    return tf.strings.split(file_path, sep=os.path.sep)[-1], file_path


def dataset2dict(ds):
    ds_dict = {}
    for file_name_, file_path_ in ds:
        ds_dict[file_name_.numpy().decode('ascii')] = file_path_.numpy().decode('ascii')

    return ds_dict


def only_subjects_of_interest(ds_dict, soi, display=False):
    dict_aux = ds_dict.copy()

    for key, val in ds_dict.items():
        if key.split(' -')[0] not in soi:
            dict_aux.pop(key)

    if display:
        for key, val in dict_aux.items():
            print(f"KEY: {key:<35s} \t\t VAL: {val}")

    return dict_aux


def summarize(ds_dict, mode, display=False):
    dict_aux = {}
    distances_gt = ['1', '3', '6', '9', '12', '15', '18', '21', '24', '27', '30']

    for key in ds_dict.keys():
        dict_aux[key.split(' -')[0]] = {}

    for key, val in ds_dict.items():
        if mode == 'gallery':
            dict_aux[key.split(' -')[0]] = val
        elif mode == 'probe':
            dict_aux[key.split(' -')[0]][re.findall('[0-9]*C', key.split('- ')[-1])[0][:-1]] = val

    for key, val in dict_aux.items():
        if display:
            print(f"KEY: {key:<10s} \t\t VAL: {val}")
        if mode == 'gallery':
            if val is None:
                print('Missing this person!!')
        elif mode == 'probe':
            if len(dict_aux[key]) != 11:
                distances = list(dict_aux[key].keys())
                d_aux = []
                for d in distances_gt:
                    if d not in distances:
                        d_aux.append(d)
                print(f"DISTANCES MISSING: {d_aux}")

    return dict_aux


def remove_2(ds_dict, mode, display=False):
    dict_aux = {}

    for key, val in ds_dict.items():
        if re.findall('_2', key):
            dict_aux[key[:-2]] = val
        else:
            dict_aux[key] = val

    if display:
        if mode == 'gallery':
            print('<><><><><><><><><><><><><><><> GALLERY <><><><><><><><><><><><><><><>')
        elif mode == 'probe':
            print('<><><><><><><><><><><><><><><> PROBE <><><><><><><><><><><><><><><>')
        for key, val in dict_aux.items():
            print(f"KEY: {key:<10s} \t\t VAL: {val}")

    return dict_aux


def preprocess(path, mode):
    img_ = tf.io.read_file(path)
    if mode == 'gallery':
        img_ = tf.image.decode_jpeg(img_, channels=3)
    elif mode == 'probe':
        img_ = tf.image.decode_png(img_, channels=3)
    img_ = tf.image.convert_image_dtype(img_, dtype=tf.float32)
    img_ = tf.image.resize(img_, [224, 224])
    img_ = tf.expand_dims(img_, axis=0)

    return img_


def compute_embeddings(ds_gallery_dict, ds_probe_dict, display=False):
    try:
        with open(f"./saved_results/Tests/UPM-GTI-Face/embeddings_FC_{CASE_OF_STUDY}.pickle", 'rb') as file:
            gallery_dict_aux, probe_dict_aux = pickle.load(file)
    except FileNotFoundError:
        gallery_dict_aux = {}
        for person_name, img_file_path in ds_gallery_dict.items():
            img = preprocess(img_file_path, mode='gallery')
            print(f"[INFO] Image of {person_name}")
            print(
                f"Image shape: {img.shape}\n"
                f"Image dtype: {img.dtype}\n"
                f"Image Min val: {tf.reduce_min(img).numpy()}\n"
                f"Image Max val: {tf.reduce_max(img).numpy()}"
            )

            embeddings_vit = vit_model(img).numpy()
            embeddings_resnet = resnet_model(img).numpy()
            embeddings_vgg = vgg_model(img).numpy()

            gallery_dict_aux[person_name] = {'file_name': img_file_path}
            gallery_dict_aux[person_name]['embeddings_vit'] = embeddings_vit
            gallery_dict_aux[person_name]['embeddings_resnet'] = embeddings_resnet
            gallery_dict_aux[person_name]['embeddings_vgg'] = embeddings_vgg

        probe_dict_aux = {}
        for person_name, dict_of_distances in ds_probe_dict.items():
            probe_dict_aux[person_name] = {}
            for distance, img_file_path in dict_of_distances.items():
                img = preprocess(img_file_path, mode='gallery')

                embeddings_vit = vit_model(img).numpy()
                embeddings_resnet = resnet_model(img).numpy()
                embeddings_vgg = vgg_model(img).numpy()

                probe_dict_aux[person_name][distance] = {'file_name': img_file_path}
                probe_dict_aux[person_name][distance]['embeddings_vit'] = embeddings_vit
                probe_dict_aux[person_name][distance]['embeddings_resnet'] = embeddings_resnet
                probe_dict_aux[person_name][distance]['embeddings_vgg'] = embeddings_vgg

        with open(f"./saved_results/Tests/UPM-GTI-Face/embeddings_FC_{CASE_OF_STUDY}.pickle", 'wb') as file:
            pickle.dump([gallery_dict_aux, probe_dict_aux], file)

    if display:
        print('<><><><><><><><><><><><><><><> EMBEDDINGS GALLERY <><><><><><><><><><><><><><><>')
        for key, val in gallery_dict_aux.items():
            print(f"KEY: {key}")
            print(f"\t - file_name: {val['file_name']}")
            print(f"\t - embeddings_vit: {val['embeddings_vit'].shape}")
            print(f"\t - embeddings_resnet: {val['embeddings_resnet'].shape}")
            print(f"\t - embeddings_vgg: {val['embeddings_vgg'].shape}")
        print('<><><><><><><><><><><><><><><> EMBEDDINGS PROBE <><><><><><><><><><><><><><><>')
        for key, val in probe_dict_aux.items():
            print(f"KEY: {key}")
            for dist in val.keys():
                print(f"\t - Distance: {dist}")
                print(f"\t\t - file_name: {val[dist]['file_name']}")
                print(f"\t\t - embeddings_vit: {val[dist]['embeddings_vit'].shape}")
                print(f"\t\t - embeddings_resnet: {val[dist]['embeddings_resnet'].shape}")
                print(f"\t\t - embeddings_vgg: {val[dist]['embeddings_vgg'].shape}")

    return gallery_dict_aux, probe_dict_aux


def create_all_pairs_no_mask(skip_distances):
    try:
        with open(f"./saved_results/Tests/UPM-GTI-Face/all_pairs_FC_N.pickle", 'rb') as file:
            all_pairs_ = pickle.load(file)
    except FileNotFoundError:
        gallery_dict_non_masked = {'FC': {}}
        probe_dict_non_masked = {'FC': {}}
        all_gallery = {}
        all_probe = {}

        with open('./saved_results/Tests/UPM-GTI-Face/embeddings_FC_I_N.pickle', 'rb') as file:
            gallery_dict_FC_indoor, probe_dict_FC_indoor = pickle.load(file)
        with open('./saved_results/Tests/UPM-GTI-Face/embeddings_FC_O_N.pickle', 'rb') as file:
            gallery_dict_FC_outdoor, probe_dict_FC_outdoor = pickle.load(file)

        for person in gallery_dict_FC_indoor.keys():
            gallery_dict_non_masked['FC'][person] = {'Indoor': deepcopy(gallery_dict_FC_indoor[person]),
                                                     'Outdoor': deepcopy(gallery_dict_FC_outdoor[person])}

            probe_dict_non_masked['FC'][person] = {'Indoor': deepcopy(probe_dict_FC_indoor[person]),
                                                   'Outdoor': deepcopy(probe_dict_FC_outdoor[person])}

        # create dictionary with all gallery items
        for person in gallery_dict_non_masked['FC'].keys():
            all_gallery[person + '_indoor'] = {}
            all_gallery[person + '_indoor'] = deepcopy(gallery_dict_non_masked['FC'][person]['Indoor'])

            all_gallery[person + '_outdoor'] = {}
            all_gallery[person + '_outdoor'] = deepcopy(gallery_dict_non_masked['FC'][person]['Outdoor'])

        # create dictionary with all probe items
        for person in probe_dict_non_masked['FC'].keys():
            for distance in probe_dict_non_masked['FC'][person]['Indoor'].keys():
                # skip distances
                if distance in skip_distances:
                    continue
                all_probe[person + '_indoor_' + distance] = {}
                all_probe[person + '_indoor_' + distance] = deepcopy(probe_dict_non_masked['FC'][person]['Indoor'][distance])

            for distance in probe_dict_non_masked['FC'][person]['Outdoor'].keys():
                # skip distances
                if distance in skip_distances:
                    continue
                all_probe[person + '_outdoor_' + distance] = {}
                all_probe[person + '_outdoor_' + distance] = deepcopy(probe_dict_non_masked['FC'][person]['Outdoor'][distance])

        # all_pairs_ = all_gallery.copy()
        all_pairs_ = deepcopy(all_gallery)
        for key in all_pairs_.keys():
            all_pairs_[key].update({'probes': deepcopy(all_probe)})

        with open(f"./saved_results/Tests/UPM-GTI-Face/all_pairs_{'FC'}_N.pickle", 'wb') as file:
            pickle.dump(all_pairs_, file)

    return all_pairs_


def compute_score_embeddings(embeddings1, embeddings2):
    cosine_distance = cosine(embeddings1, embeddings2)
    score = 1 - cosine_distance

    return score


def compute_score_activation_maps(act_maps1, act_maps2):
    act_maps1 = tf.reshape(act_maps1, shape=-1)
    act_maps2 = tf.reshape(act_maps2, shape=-1)
    score = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.AUTO)(act_maps1, act_maps2)
    score = score.numpy() * -1

    return score


def compute_scores_and_ground_truths(ds_dict):
    for gallery_person in ds_dict.keys():
        for probe_person in ds_dict[gallery_person]['probes'].keys():
            score_vit = compute_score_embeddings(
                ds_dict[gallery_person]['embeddings_vit'],
                ds_dict[gallery_person]['probes'][probe_person]['embeddings_vit']
            )
            score_resnet = compute_score_embeddings(
                ds_dict[gallery_person]['embeddings_resnet'],
                ds_dict[gallery_person]['probes'][probe_person]['embeddings_resnet']
            )
            score_vgg = compute_score_embeddings(
                ds_dict[gallery_person]['embeddings_vgg'],
                ds_dict[gallery_person]['probes'][probe_person]['embeddings_vgg']
            )

            # ground truth equal to 1 if same person, 0 if different
            if gallery_person.split('_')[0] == probe_person.split('_')[0]:
                gt = 1
            else:
                gt = 0

            ds_dict[gallery_person]['probes'][probe_person].update({'ground_truth': gt})
            ds_dict[gallery_person]['probes'][probe_person].update({'score_vit': score_vit})
            ds_dict[gallery_person]['probes'][probe_person].update({'score_resnet': score_resnet})
            ds_dict[gallery_person]['probes'][probe_person].update({'score_vgg': score_vgg})

    return ds_dict


def compute_roc(ds_dict, fig_name, positive_label=1):
    vit_results = []
    resnet_results = []
    vgg_results = []
    gt_results = []

    for gallery_person in ds_dict.keys():
        for probe_person in ds_dict[gallery_person]['probes'].keys():
            vit_results.append(ds_dict[gallery_person]['probes'][probe_person]['score_vit'])
            resnet_results.append(ds_dict[gallery_person]['probes'][probe_person]['score_resnet'])
            vgg_results.append(ds_dict[gallery_person]['probes'][probe_person]['score_vgg'])
            gt_results.append(ds_dict[gallery_person]['probes'][probe_person]['ground_truth'])

    # Figures
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    lw = 3

    # ViT
    fpr_vit, tpr_vit, thresholds_vit = roc_curve(gt_results, vit_results, pos_label=positive_label)
    auc_vit = auc(fpr_vit, tpr_vit)
    fnr_vit = 1 - tpr_vit
    eer_vit = fpr_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))]
    eer_vit_threshold = thresholds_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))]

    ax.plot(fpr_vit, tpr_vit, linestyle='-', lw=lw, color='blue', label='ViT_B32 (EER=%s, AUC=%s)' % ('{0:.2f}'.format(eer_vit), '{0:.2f}'.format(auc_vit)))
    ax.scatter(eer_vit, tpr_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))], color='blue', linewidths=8, zorder=10)

    # ResNet
    fpr_resnet, tpr_resnet, thresholds_resnet = roc_curve(gt_results, resnet_results, pos_label=positive_label)
    auc_resnet = auc(fpr_resnet, tpr_resnet)
    fnr_resnet = 1 - tpr_resnet
    eer_resnet = fpr_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))]
    eer_resnet_threshold = thresholds_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))]

    ax.plot(fpr_resnet, tpr_resnet, linestyle='-', lw=lw, color='orange', label='ResNet_50 (EER=%s, AUC=%s)' % ('{0:.2f}'.format(eer_resnet), '{0:.2f}'.format(auc_resnet)))
    ax.scatter(eer_resnet, tpr_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))], color='orange', linewidths=8, zorder=10)

    # VGG
    fpr_vgg, tpr_vgg, thresholds_vgg = roc_curve(gt_results, vgg_results, pos_label=positive_label)
    auc_vgg = auc(fpr_vgg, tpr_vgg)
    fnr_vgg = 1 - tpr_vgg
    eer_vgg = fpr_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))]
    eer_vgg_threshold = thresholds_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))]

    ax.plot(fpr_vgg, tpr_vgg, linestyle='-', lw=lw, color='green', label='VGG_16 (EER=%s, AUC=%s)' % ('{0:.2f}'.format(eer_vgg), '{0:.2f}'.format(auc_vgg)))
    ax.scatter(eer_vgg, tpr_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))], color='green', linewidths=8, zorder=10)

    # ROC fig params
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title('Receiver Operating Characteristics (ROC)', fontsize=15)
    ax.set_xlabel('FPR', fontsize=15)
    ax.set_ylabel('TPR', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.legend(loc='lower right', prop={"size": 11})

    plt.savefig(f"./saved_results/Tests/UPM-GTI-Face/{fig_name}.png", bbox_inches='tight')

    results = {'vit': {}, 'resnet': {}, 'vgg': {}, 'gt_results': gt_results}
    results['vit'].update({'results': vit_results, 'thresholds': thresholds_vit, 'eer_threshold': eer_vit_threshold})
    results['resnet'].update({'results': resnet_results, 'thresholds': thresholds_resnet, 'eer_threshold': eer_resnet_threshold})
    results['vgg'].update({'results': vgg_results, 'thresholds': thresholds_vgg, 'eer_threshold': eer_vgg_threshold})

    return results


def compute_metrics(scores_, ground_truth_):
    tp_ = 0
    tn_ = 0
    fp_ = 0
    fn_ = 0
    for score_, gt_ in zip(scores_, ground_truth_):
        if score_ == 1 and gt_ == 1:
            tp_ += 1
        elif score_ == 1 and gt_ == 0:
            fp_ += 1
        elif score_ == 0 and gt_ == 1:
            fn_ += 1
        elif score_ == 0 and gt_ == 0:
            tn_ += 1

    recall_ = tp_ / (tp_ + fn_ + tf.keras.backend.epsilon())
    precision_ = tp_ / (tp_ + fp_ + tf.keras.backend.epsilon())
    fscore_ = (2 * tp_) / (2 * tp_ + fp_ + fn_ + tf.keras.backend.epsilon())

    return tp_, tn_, fp_, fn_, recall_, precision_, fscore_


def roc2f_score(ds_dict, fig_name):
    vit_tp = []
    vit_tn = []
    vit_fp = []
    vit_fn = []
    vit_recall = []
    vit_precision = []
    vit_fscore = []
    for threshold in ds_dict['vit']['thresholds']:
        scores = np.where(ds_dict['vit']['results'] > threshold, 1, 0)
        tp, tn, fp, fn, recall, precision, fscore = compute_metrics(scores, ds_dict['gt_results'])

        vit_tp.append(tp)
        vit_tn.append(tn)
        vit_fp.append(fp)
        vit_fn.append(fn)
        vit_recall.append(recall)
        vit_precision.append(precision)
        vit_fscore.append(fscore)
        if threshold == ds_dict['vit']['eer_threshold']:
            vit_tp_eer, vit_tn_eer, vit_fp_eer, vit_fn_eer, vit_recall_eer, vit_precision_eer, vit_fscore_eer = \
                tp, tn, fp, fn, recall, precision, fscore

    resnet_tp = []
    resnet_tn = []
    resnet_fp = []
    resnet_fn = []
    resnet_recall = []
    resnet_precision = []
    resnet_fscore = []
    for threshold in ds_dict['resnet']['thresholds']:
        scores = np.where(ds_dict['resnet']['results'] > threshold, 1, 0)
        tp, tn, fp, fn, recall, precision, fscore = compute_metrics(scores, ds_dict['gt_results'])

        resnet_tp.append(tp)
        resnet_tn.append(tn)
        resnet_fp.append(fp)
        resnet_fn.append(fn)
        resnet_recall.append(recall)
        resnet_precision.append(precision)
        resnet_fscore.append(fscore)
        if threshold == ds_dict['resnet']['eer_threshold']:
            resnet_tp_eer, resnet_tn_eer, resnet_fp_eer, resnet_fn_eer, resnet_recall_eer, resnet_precision_eer, resnet_fscore_eer = \
                tp, tn, fp, fn, recall, precision, fscore

    vgg_tp = []
    vgg_tn = []
    vgg_fp = []
    vgg_fn = []
    vgg_recall = []
    vgg_precision = []
    vgg_fscore = []
    for threshold in ds_dict['vgg']['thresholds']:
        scores = np.where(ds_dict['vgg']['results'] > threshold, 1, 0)
        tp, tn, fp, fn, recall, precision, fscore = compute_metrics(scores, ds_dict['gt_results'])

        vgg_tp.append(tp)
        vgg_tn.append(tn)
        vgg_fp.append(fp)
        vgg_fn.append(fn)
        vgg_recall.append(recall)
        vgg_precision.append(precision)
        vgg_fscore.append(fscore)
        if threshold == ds_dict['vgg']['eer_threshold']:
            vgg_tp_eer, vgg_tn_eer, vgg_fp_eer, vgg_fn_eer, vgg_recall_eer, vgg_precision_eer, vgg_fscore_eer = \
                tp, tn, fp, fn, recall, precision, fscore

    f_scores = ['F-SCORE', 'F-score',
                vit_fscore, resnet_fscore, vgg_fscore,
                vit_fscore_eer, resnet_fscore_eer, vgg_fscore_eer]
    precisions = ['PRECISION', 'Precision',
                vit_precision, resnet_precision, vgg_precision,
                vit_precision_eer, resnet_precision_eer, vgg_precision_eer]
    recalls = ['RECALL', 'Recall',
                vit_recall, resnet_recall, vgg_recall,
                vit_recall_eer, resnet_recall_eer, vgg_recall_eer]
    fns = ['FALSE_NEGATIVES', 'False negatives',
           vit_fn, resnet_fn, vgg_fn,
           vit_fn_eer, resnet_fn_eer, vgg_fn_eer]
    fps = ['FALSE_POSITIVES', 'False positives',
           vit_fp, resnet_fp, vgg_fp,
           vit_fp_eer, resnet_fp_eer, vgg_fp_eer]
    tns = ['TRUE_NEGATIVES', 'True negatives',
           vit_tn, resnet_tn, vgg_tn,
           vit_tn_eer, resnet_tn_eer, vgg_tn_eer]
    tps = ['TRUE_POSITIVES', 'True positives',
           vit_tp, resnet_tp, vgg_tp,
           vit_tp_eer, resnet_tp_eer, vgg_tp_eer]

    all_metrics = [f_scores, precisions, recalls, fns, fps, tns, tps]
    for metric in all_metrics:
        # Figures
        fig, ax = plt.subplots(1, 1, figsize=(10, 9))
        lw = 3

        ax.plot(ds_dict['vit']['thresholds'], metric[2], linestyle='-', lw=lw, color='blue', label=f"ViT_B32")
        ax.scatter(ds_dict['vit']['eer_threshold'], metric[5], color='blue', linewidths=8, zorder=10)
        ax.plot(ds_dict['resnet']['thresholds'], metric[3], linestyle='-', lw=lw, color='orange', label=f"ResNet_50")
        ax.scatter(ds_dict['resnet']['eer_threshold'], metric[6], color='orange', linewidths=8, zorder=10)
        ax.plot(ds_dict['vgg']['thresholds'], metric[4], linestyle='-', lw=lw, color='green', label=f"VGG_16")
        ax.scatter(ds_dict['vgg']['eer_threshold'], metric[7], color='green', linewidths=8, zorder=10)

        ax.set_title(metric[0], fontsize=15)
        ax.set_xlabel('Thresholds', fontsize=15)
        ax.set_ylabel(metric[1], fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.legend(loc='lower right', prop={"size": 11})

        plt.savefig(f"./saved_results/Tests/UPM-GTI-Face/{fig_name}_{metric[0]}.png", bbox_inches='tight')


"""
GTI-FACE DATASET
"""

""" INDOOR """
# Not Masked
GALLERY_I_N = './datasets/UPM-GTI-Face/Indoor/Gallery/Not Masked/'
PROBE_I_N = './datasets/UPM-GTI-Face/Indoor/Camera/Not Masked/Frames/Curated/'
GALLERY_SUBJECTS_I_N = ['Andres', 'Cristina', 'DaniB', 'DaniF', 'Diego', 'Ester', 'German', 'Isa', 'Marcos', 'Narciso', 'Pablo']
PROBE_SUBJECTS_I_N = ['Andres', 'Cristina', 'DaniB', 'DaniF', 'Diego', 'Ester', 'German', 'Isa', 'Marcos_2', 'Narciso', 'Pablo']
# Masked
GALLERY_I_M = './datasets/UPM-GTI-Face/Indoor/Gallery/Masked/'
PROBE_I_M = './datasets/UPM-GTI-Face/Indoor/Camera/Masked/Frames/Curated/'
GALLERY_SUBJECTS_I_M = ['Andres', 'Cristina', 'DaniB', 'DaniF', 'Diego', 'Ester', 'German', 'Isa', 'Marcos', 'Narciso', 'Pablo']
PROBE_SUBJECTS_I_M = ['Andres', 'Cristina', 'DaniB', 'DaniF', 'Diego', 'Ester', 'German', 'Isa', 'Marcos_2', 'Narciso', 'Pablo']

""" OUTDOOR """
# Not Masked
GALLERY_O_N = './datasets/UPM-GTI-Face/Outdoor/Gallery/Not Masked/'
PROBE_O_N = './datasets/UPM-GTI-Face/Outdoor/Camera/Not Masked/Frames/Curated/'
GALLERY_SUBJECTS_O_N = ['Andres', 'Cristina', 'DaniB', 'DaniF_2', 'Diego', 'Ester', 'German', 'Isa_2', 'Marcos', 'Narciso', 'Pablo']
PROBE_SUBJECTS_O_N = ['Andres', 'Cristina', 'DaniB', 'DaniF_2', 'Diego', 'Ester', 'German', 'Isa_2', 'Marcos', 'Narciso', 'Pablo']
# Masked
GALLERY_O_M = './datasets/UPM-GTI-Face/Outdoor/Gallery/Masked/'
PROBE_O_M = './datasets/UPM-GTI-Face/Outdoor/Camera/Masked/Frames/Curated/'
GALLERY_SUBJECTS_O_M = ['Andres', 'Cristina', 'DaniB', 'DaniF_2', 'Diego', 'Ester', 'German', 'Isa_2', 'Marcos', 'Narciso', 'Pablo']
PROBE_SUBJECTS_O_M = ['Andres', 'Cristina', 'DaniB', 'DaniF_2', 'Diego', 'Ester', 'German', 'Isa_2', 'Marcos', 'Narciso', 'Pablo']


"""
EXPERIMENT SETTINGS
"""

# Experiment settings
GALLERY_DIRECTORY = GALLERY_I_M
PROBE_DIRECTORY = PROBE_I_M
SUBJECTS_OF_INTEREST_GALLERY = GALLERY_SUBJECTS_I_M
SUBJECTS_OF_INTEREST_PROBE = PROBE_SUBJECTS_I_M
CASE_OF_STUDY = 'I_N'

"""
CREATE DICTIONARY OF THE DATASET
"""

gallery_dataset = tf.data.Dataset.list_files(GALLERY_DIRECTORY + '*.jpg', shuffle=False)
probe_dataset = tf.data.Dataset.list_files(PROBE_DIRECTORY + '*.png', shuffle=False)

gallery_dataset = gallery_dataset.map(get_label)
probe_dataset = probe_dataset.map(get_label)

gallery_dict = dataset2dict(gallery_dataset)
probe_dict = dataset2dict(probe_dataset)

gallery_dict = only_subjects_of_interest(gallery_dict, SUBJECTS_OF_INTEREST_GALLERY, display=False)
probe_dict = only_subjects_of_interest(probe_dict, SUBJECTS_OF_INTEREST_PROBE, display=False)

gallery_dict = summarize(gallery_dict, mode='gallery', display=False)
probe_dict = summarize(probe_dict, mode='probe', display=False)

gallery_dict = remove_2(gallery_dict, mode='gallery', display=False)
probe_dict = remove_2(probe_dict, mode='probe', display=False)


"""
LOAD MODELS
"""

image_size = 224
num_classes = 8631

""" Vision Transformer """
vit_model = vit.vit_b32(
    image_size=image_size,
    pretrained=True,
    include_top=False,
    pretrained_top=False,
)
y = tf.keras.layers.Dense(num_classes, activation='softmax')(vit_model.output)
vit_model = tf.keras.models.Model(inputs=vit_model.input, outputs=y)

vit_model.load_weights("./saved_results/Models/ViT_B32/checkpoint").expect_partial()   # suppresses warnings
vit_model = tf.keras.models.Model(inputs=vit_model.input, outputs=vit_model.layers[-2].output)
vit_model.summary()

""" ResNet50 """
resnet_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(image_size, image_size, 3),
    pooling=None,
)
Y = tf.keras.layers.GlobalAvgPool2D()(resnet_model.output)
Y = tf.keras.layers.Dense(units=num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform())(Y)
resnet_model = tf.keras.models.Model(inputs=resnet_model.input, outputs=Y, name='ResNet50')

resnet_model.load_weights("./saved_results/Models/ResNet_50/checkpoint").expect_partial()   # suppresses warnings
resnet_model = tf.keras.models.Model(inputs=resnet_model.input, outputs=resnet_model.layers[-2].output)
resnet_model.summary()

""" VGG16 """
vgg_model = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_shape=(image_size, image_size, 3),
    pooling=None,
)
Y = vgg_model.layers[-2].output
Y = tf.keras.layers.Dense(units=num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform)(Y)
vgg_model = tf.keras.models.Model(inputs=vgg_model.input, outputs=Y, name='VGG16')

vgg_model.load_weights("./saved_results/Models/VGG_16/checkpoint").expect_partial()   # suppresses warnings
vgg_model = tf.keras.models.Model(inputs=vgg_model.input, outputs=vgg_model.layers[-2].output)
vgg_model.summary()


"""
COMPUTE EMBEDDINGS (ONLY NEED TO EXECUTE THIS ONCE FOR EVERY SCENARIO IN UPM-GTI-FACE DATASET)
"""

# Uncomment the next line only the first time a particular scenario is tested
# gallery_dict, probe_dict = compute_embeddings(gallery_dict, probe_dict, display=False)


"""
GET DICTIONARY WITH ALL POSSIBLE PAIRS (WITHOUT MASK)
"""

# Load all Non-masked images (both indoor & outdoor). This should amount to:
#   - gallery images:   22
#   - probe images:     220
#   - Number of comparisons: 22 * 220 = 4.840

# distances =       ['1', '3', '6', '9', '12', '15', '18', '21', '24', '27', '30']
distances_to_skip = ['1']
all_pairs_FC = create_all_pairs_no_mask(skip_distances=distances_to_skip)

"""
COMPUTE SCORES AND GROUND TRUTHS
"""

all_pairs_FC = compute_scores_and_ground_truths(all_pairs_FC)


"""
COMPUTE ROC CURVES
"""

results_FC = compute_roc(all_pairs_FC, fig_name='FC_ROC', positive_label=1)


"""
CONVERT ROC CURVES INTO F-SCORE CURVES
"""

roc2f_score(results_FC, fig_name='FC')
