import tensorflow as tf
import os


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(f"Setting memory growth for gpu: {gpu}")
    tf.config.experimental.set_memory_growth(gpu, True)


vggface2_train = './datasets/VGG-Face2/data/train'    # Total nº files: 3141890    'Z:/Datasets/VGG-Face2/data/train'
vggface2_test = './datasets/VGG-Face2/data/test'      # Total nº files: 169396     'Z:/Datasets/VGG-Face2/data/test'
dataset_path = vggface2_train
labels_list = [tf.constant(x) for x in sorted(os.listdir(dataset_path))]

AUTO = tf.data.AUTOTUNE


def get_label(file_path):
    return tf.strings.split(file_path, os.path.sep)[-2]


def process_image(file_path, target_size, labels=True):
    label = get_label(file_path)
    label = tf.argmax(label == labels_list)     # Sparse labels

    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, [target_size, target_size])

    if labels:
        return img, label
    else:
        return img


def create_data_generators(target_size, batch_size, ret_labels=True, seed=0):
    dataset = tf.data.Dataset.list_files(dataset_path + '/*/*', shuffle=True, seed=seed)

    total_samples = len(dataset)
    train_split = 0.9
    val_split = 0.05
    test_split = 0.05

    train_len = int(total_samples * train_split)
    val_len = int(total_samples * val_split)
    test_len = int(total_samples * test_split)

    train_ds = dataset.take(train_len)
    val_ds = dataset.skip(train_len).take(val_len)
    test_ds = dataset.skip(train_len).skip(val_len)

    print(f"[INFO] Num images for train: {train_len} -> train_ds: {len(train_ds)}")
    print(f"[INFO] Num images for validation: {val_len} -> val_ds: {len(val_ds)}")
    print(f"[INFO] Num images for test: {test_len} -> test_ds: {len(test_ds)}")

    train_ds = (
        train_ds
        .shuffle(buffer_size=2*batch_size, seed=seed)
        .map(lambda x: process_image(x, target_size, labels=ret_labels), num_parallel_calls=AUTO)
        .batch(batch_size=batch_size)
        .prefetch(buffer_size=AUTO)
    )

    val_ds = (
        val_ds
        .map(lambda x: process_image(x, target_size, labels=ret_labels), num_parallel_calls=AUTO)
        .batch(batch_size=batch_size)
        .prefetch(buffer_size=AUTO)
    )

    test_ds = (
        test_ds
        .map(lambda x: process_image(x, target_size, labels=ret_labels), num_parallel_calls=AUTO)
        .batch(batch_size=batch_size)
        .prefetch(buffer_size=AUTO)
    )

    return train_ds, val_ds, test_ds
