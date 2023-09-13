import os
import tensorflow as tf
from vit_keras import vit


"""
CREATE DATASET
"""
BASE_DIR = '/mnt/Data/mrt/SCface_database'
MUGSHOT_DIR = f'{BASE_DIR}/mugshot_frontal_cropped_all'
SURVEILLANCE_DIR = f'{BASE_DIR}/surveillance_cameras_all'

mugshot_data = {}
for file in os.listdir(MUGSHOT_DIR):
    person = file.split('_')[0]
    file_path = os.path.join(MUGSHOT_DIR, file)
    mugshot_data[person] = {'file': file_path, 'embeddings': None}

surveillance_data = {}
for person in mugshot_data.keys():
    surveillance_data[person] = {'files': [], 'embeddings': []}
for file in os.listdir(SURVEILLANCE_DIR):
    person = file.split('_')[0]
    file_path = os.path.join(SURVEILLANCE_DIR, file)
    surveillance_data[person]['files'].append(file_path)
    surveillance_data[person]['embeddings'].append(None)
