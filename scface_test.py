import os

BASE_DIR = '/mnt/Data/mrt/SCface_database'
MUGSHOT_DIR = f'{BASE_DIR}/mugshot_frontal_cropped_all'
SURVEILLANCE_DIR = f'{BASE_DIR}/surveillance_cameras_all'

print(os.listdir(SURVEILLANCE_DIR))
