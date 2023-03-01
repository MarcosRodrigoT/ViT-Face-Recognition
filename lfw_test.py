import tensorflow as tf
from vit_keras import vit
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def preprocess(paths):
    path_img1, path_img2 = paths

    img1 = tf.io.read_file(path_img1)
    img1 = tf.image.decode_jpeg(img1, channels=3)
    img1 = tf.image.convert_image_dtype(img1, dtype=tf.float32)
    img1 = tf.image.resize(img1, [224, 224])
    img1 = tf.expand_dims(img1, axis=0)

    img2 = tf.io.read_file(path_img2)
    img2 = tf.image.decode_jpeg(img2, channels=3)
    img2 = tf.image.convert_image_dtype(img2, dtype=tf.float32)
    img2 = tf.image.resize(img2, [224, 224])
    img2 = tf.expand_dims(img2, axis=0)

    return [img1, img2]


def compute_score(embeddings1, embeddings2):
    cosine_distance = cosine(embeddings1, embeddings2)
    score = 1 - cosine_distance

    return score


def compute_roc(results_dictionary, fig_name, positive_label=1):
    vit_results = []
    resnet_results = []
    vgg_results = []
    gt_results = []

    for pair_example in results_dictionary.keys():
        vit_results.append(results_dictionary[pair_example]['vit'])
        resnet_results.append(results_dictionary[pair_example]['resnet'])
        vgg_results.append(results_dictionary[pair_example]['vgg'])
        gt_results.append(results_dictionary[pair_example]['GT'])

    # Figures
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    lw = 3

    """ ViT_B32 """
    # Data
    fpr_vit, tpr_vit, thresholds_vit = roc_curve(gt_results, vit_results, pos_label=positive_label)
    auc_vit = auc(fpr_vit, tpr_vit)
    fnr_vit = 1 - tpr_vit
    eer_vit = fpr_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))]
    eer_vit_threshold = thresholds_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))]

    # Plot
    ax.plot(fpr_vit, tpr_vit, linestyle='-', lw=lw, color='blue', label='ViT_B32 (EER=%s, AUC=%s)' % ('{0:.2f}'.format(eer_vit), '{0:.3f}'.format(auc_vit)))
    ax.scatter(eer_vit, tpr_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))], color='blue', linewidths=8, zorder=10)

    # CSV
    vit_pd = pd.DataFrame({'FPR_ViT': fpr_vit, 'TPR_ViT': tpr_vit})
    vit_pd['EER_ViT'] = pd.DataFrame([eer_vit, tpr_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))]])
    vit_pd.to_csv('./saved_results/Tests/LFW/ViT_B32_ROC.csv', header=True, index=False)

    """ ResNet_B32 """
    # Data
    fpr_resnet, tpr_resnet, thresholds_resnet = roc_curve(gt_results, resnet_results, pos_label=positive_label)
    auc_resnet = auc(fpr_resnet, tpr_resnet)
    fnr_resnet = 1 - tpr_resnet
    eer_resnet = fpr_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))]
    eer_resnet_threshold = thresholds_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))]

    # Plot
    ax.plot(fpr_resnet, tpr_resnet, linestyle='-', lw=lw, color='orange', label='ResNet_50 (EER=%s, AUC=%s)' % ('{0:.2f}'.format(eer_resnet), '{0:.3f}'.format(auc_resnet)))
    ax.scatter(eer_resnet, tpr_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))], color='orange', linewidths=8, zorder=10)

    # CSV
    resnet_pd = pd.DataFrame({'FPR_RESNET': fpr_resnet, 'TPR_RESNET': tpr_resnet})
    resnet_pd['EER_RESNET'] = pd.DataFrame([eer_resnet, tpr_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))]])
    resnet_pd.to_csv('./saved_results/Tests/LFW/ResNet_50_ROC.csv', header=True, index=False)

    """ VGG_16 """
    # Data
    fpr_vgg, tpr_vgg, thresholds_vgg = roc_curve(gt_results, vgg_results, pos_label=positive_label)
    auc_vgg = auc(fpr_vgg, tpr_vgg)
    fnr_vgg = 1 - tpr_vgg
    eer_vgg = fpr_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))]
    eer_vgg_threshold = thresholds_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))]

    # Plot
    ax.plot(fpr_vgg, tpr_vgg, linestyle='-', lw=lw, color='green', label='VGG_16 (EER=%s, AUC=%s)' % ('{0:.2f}'.format(eer_vgg), '{0:.3f}'.format(auc_vgg)))
    ax.scatter(eer_vgg, tpr_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))], color='green', linewidths=8, zorder=10)

    # CSV
    vgg_pd = pd.DataFrame({'FPR_VGG': fpr_vgg, 'TPR_VGG': tpr_vgg})
    vgg_pd['EER_VGG'] = pd.DataFrame([eer_vgg, tpr_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))]])
    vgg_pd.to_csv('./saved_results/Tests/LFW/VGG_16_ROC.csv', header=True, index=False)

    ax.set_title('Receiver Operating Characteristics (ROC)', fontsize=15)
    ax.set_xlabel('FPR', fontsize=15)
    ax.set_ylabel('TPR', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.legend(loc='lower right', prop={"size": 11})

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.savefig(f"./saved_results/Tests/LFW/{fig_name}.png", bbox_inches='tight')

    ax.set_xlim([0.0, 0.3])
    ax.set_ylim([0.7, 1.0])
    plt.savefig(f"./saved_results/Tests/LFW/{fig_name}_zoom.png", bbox_inches='tight')


"""
CREATE DATASET WITH PAIRS
"""

pairs = {}
with open('Z:/Datasets/LFW/pairs.txt', 'r') as file:
    for idx, line in enumerate(file):
        line_aux = line.split('\n')[0].split('\t')
        if len(line_aux) < 4:
            line_aux.insert(2, line_aux[0])

        img_first = f"{line_aux[0]}_{''.join(str(i) for i in [0 for _ in range(4 - len(line_aux[1]))])}{line_aux[1]}.jpg"
        img_second = f"{line_aux[2]}_{''.join(str(i) for i in [0 for _ in range(4 - len(line_aux[3]))])}{line_aux[3]}.jpg"
        pairs[idx] = [
            f".datasets/LFW/lfw/{line_aux[0]}/{img_first}",
            f".datasets/LFW/lfw/{line_aux[2]}/{img_second}",
            1 if line_aux[0] == line_aux[2] else 0,     # 1 if same person, 0 if different
        ]

print(f"[INFO] Pairs:")
for key, val in pairs.items():
    print(f"Pair {key}: {val[0]} \t->\t {val[1]}")


"""
LOAD MODELS
"""

image_size = 224
num_classes = 8631

""" ViT_B32 """
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

""" ResNet_50 """
resnet50_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(image_size, image_size, 3),
    pooling=None,
)
Y = tf.keras.layers.GlobalAvgPool2D()(resnet50_model.output)
Y = tf.keras.layers.Dense(units=num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform())(Y)
resnet50_model = tf.keras.models.Model(inputs=resnet50_model.input, outputs=Y, name='ResNet50')

resnet50_model.load_weights("./saved_results/Models/ResNet_50/checkpoint").expect_partial()   # suppresses warnings
resnet50_model = tf.keras.models.Model(inputs=resnet50_model.input, outputs=resnet50_model.layers[-2].output)
resnet50_model.summary()

""" VGG_16 """
vgg16_model = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_shape=(image_size, image_size, 3),
    pooling=None,
)
Y = vgg16_model.layers[-2].output
Y = tf.keras.layers.Dense(units=num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform)(Y)
vgg16_model = tf.keras.models.Model(inputs=vgg16_model.input, outputs=Y, name='VGG16')

vgg16_model.load_weights("./saved_results/Models/VGG_16/checkpoint").expect_partial()   # suppresses warnings
vgg16_model = tf.keras.models.Model(inputs=vgg16_model.input, outputs=vgg16_model.layers[-2].output)
vgg16_model.summary()


"""
PREPROCESS IMAGE PAIRS AND COMPUTE SCORE MODELS
"""

try:
    with open('./saved_results/Tests/LFW/results.pickle', 'rb') as file:
        results = pickle.load(file)
except FileNotFoundError:
    results = {}
    for key, val in pairs.items():
        img1_, img2_ = preprocess(val[:-1])
        print(f"[INFO] Pair {key}")
        print(
            f"Image 1 shape: {img1_.shape}\n"
            f"Image 1 dtype: {img1_.dtype}\n"
            f"Image 1 Min val: {tf.reduce_min(img1_).numpy()}\n"
            f"Image 1 Max val: {tf.reduce_max(img1_).numpy()}"
        )

        embeddings1_vit = vit_model(img1_).numpy()
        embeddings2_vit = vit_model(img2_).numpy()
        print('Vision Transformer embeddings:')
        print('- Shape:', embeddings1_vit.shape)
        print('- Dtype:', embeddings1_vit.dtype)
        print('- Mean:', tf.reduce_mean(embeddings1_vit))
        print('- Min:', tf.reduce_min(embeddings1_vit))
        print('- Max:', tf.reduce_max(embeddings1_vit))
        score_vit = compute_score(embeddings1_vit, embeddings2_vit)
        print('Vision Transformer score:')
        print('- Score:', score_vit)

        embeddings1_resnet = resnet50_model(img1_).numpy()
        embeddings2_resnet = resnet50_model(img2_).numpy()
        print('ResNet50 embeddings:')
        print('- Shape:', embeddings1_resnet.shape)
        print('- Dtype:', embeddings1_resnet.dtype)
        print('- Mean:', tf.reduce_mean(embeddings1_resnet))
        print('- Min:', tf.reduce_min(embeddings1_resnet))
        print('- Max:', tf.reduce_max(embeddings1_resnet))
        score_resnet = compute_score(embeddings1_resnet, embeddings2_resnet)
        print('ResNet50 score:')
        print('- Score:', score_resnet)

        embeddings1_vgg16 = vgg16_model(img1_).numpy()
        embeddings2_vgg16 = vgg16_model(img2_).numpy()
        print('VGG16 embeddings:')
        print('- Shape:', embeddings1_vgg16.shape)
        print('- Dtype:', embeddings1_vgg16.dtype)
        print('- Mean:', tf.reduce_mean(embeddings1_vgg16))
        print('- Min:', tf.reduce_min(embeddings1_vgg16))
        print('- Max:', tf.reduce_max(embeddings1_vgg16))
        score_vgg16 = compute_score(embeddings1_vgg16, embeddings2_vgg16)
        print('VGG16 score:')
        print('- Score:', score_vgg16)

        results[key] = {'vit': score_vit, 'resnet': score_resnet, 'vgg': score_vgg16, 'GT': val[-1]}

    with open('./saved_results/Tests/LFW/results.pickle', 'wb') as file:
        pickle.dump(results, file)

print(f"[INFO] Results:")
for key, val in results.items():
    print(f"[INFO] Pair {key} -> \tGround truth: {val['GT']}")
    print(f"[INFO] \t ViT: \t\t{round(val['vit'], 2)}")
    print(f"[INFO] \t ResNet: \t{round(val['resnet'], 2)}")
    print(f"[INFO] \t VGG: \t\t{round(val['vgg'], 2)}")

compute_roc(results, fig_name='ROC', positive_label=1)
