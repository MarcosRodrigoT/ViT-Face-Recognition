# Vision Transformers for Face Recognition
![](https://img.shields.io/badge/python-3.9-brightgreen)
![](https://img.shields.io/badge/tensorflow-2.7-orange)
![](https://img.shields.io/badge/CUDA-11.2-blue)
![](https://img.shields.io/badge/cuDNN-8.1-blue)

This is the official implementation of the paper titled "Comprehensive Comparison of Vision Transformers and 
Traditional Convolutional Neural Networks for Face Recognition Tasks".

The whole project, including model weights and extensive results can be found in
https://www.gti.ssr.upm.es/data as a .zip file.

The structure of directories should look like:
```
Project
|-> datasets
    |-> LFW
        |-> lfw
    |-> UPM-GTI-Face
    |-> VGG-Face2
|-> saved_results
    |-> Models
        |-> ResNet_50
        |-> VGG_16
        |->  ViT_B32
    |-> Tests
        |-> LFW
        |-> UPM-GTI-Face
```


## Pre-requisites
1. Download the following datasets and move them to their respective directories:

* [VGG-Face2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
* [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/#download)
* [UPM-GTI-Face](https://www.gti.ssr.upm.es/data/upm-gti-face-dataset)

```sh
cd ~/Downloads/
mv VGG-Face2/* ~/Project/datasets/VGG-Face2/
mv lfw/* ~/Project/datasets/LFW/lfw/
mv UPM-GTI-Face/* ~/Project/datasets/UPM-GTI-Face
```

2. Install the requirements
```bash
pip install -r requirements.txt
```


## Training
The training of the three models can be achieved executing their respective files.
The results of the training will be saved to ``/tmp`` directory.

* ViT_B32
```bash
python vitb32_train.py
```

* ResNet_50
```bash
python resnet50_train.py
```

* VGG_16
```bash
python vgg16_train.py
```

Any of the networks can be trained from scratch by commenting the following line in the respective training file:
```python
"""
LOAD PRE-TRAINED MODEL WEIGHTS
"""

# Load pre-trained model weights before training
best_weights = "./saved_results/Models/ViT_B32/checkpoint"
vit_model.load_weights(best_weights)
```


## LFW Test
The test can be performed by executing the corresponding file. Results will be saved to
``/saved_results/Tests/LFW``.

```bash
python lfw_test.py
```


## UPM-GTI-Face Test
The test can be performed by executing the corresponding file. Results will be saved to
``/saved_results/Tests/UPM-GTI-Face``.

```bash
python upm-gti-face_test.py
```


## Authors
* Marcos Rodrigo - marcos.rodrigo@upm.es
* Carlos Cuevas - carlos.cuevas@upm.es
* Narciso García - narciso.garcia@upm.es


## Citation
@inproceedings{rodrigo2023comprehensive,\
    title={Comprehensive Comparison of Vision Transformers and Traditional Convolutional Neural Networks for Face Recognition Tasks},\
    author={Rodrigo, Marcos and Cuevas, Carlos and García, Narciso},\
    booktitle={Under revision for the 2023 IEEE International Conference on Image Processing},\
    year={2023},\
    organization={IEEE}\
}