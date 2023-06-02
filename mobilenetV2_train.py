import datetime
import tensorflow as tf
from tensorflow.keras.layers import GlobalAvgPool2D, Dense
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from data_generator import create_data_generators


"""
HYPERPARAMETERS
"""

# Distribute training
strategy = tf.distribute.MirroredStrategy()

# Input
image_size = 224

# Hyper-parameters
batch_size = 128 * strategy.num_replicas_in_sync
num_epochs = 25
learning_rate = 0.0001
num_classes = 8631


"""
DATASET
"""

train_gen, val_gen, test_gen = create_data_generators(target_size=image_size, batch_size=batch_size)


"""
MODEL
"""

with strategy.scope():
    mobilenet_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        pooling=None,
    )
    Y = GlobalAvgPool2D()(mobilenet_model.output)
    Y = Dense(units=num_classes, activation='softmax', kernel_initializer=GlorotUniform())(Y)
    mobilenet_model = Model(inputs=mobilenet_model.input, outputs=Y, name='MobileNetV2')
mobilenet_model.summary(line_length=150)


"""
MODEL COMPILE
"""

with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    mobilenet_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top-5-accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=100, name='top-100-accuracy'),
        ]
    )


"""
CALLBACKS
"""

# checkpoint callback
checkpoint_filepath = "./tmp/checkpoint"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    save_freq='epoch',
)

# csv logger callback
csv_filepath = "./tmp/training_log.csv"
csv_logger = tf.keras.callbacks.CSVLogger(
    csv_filepath,
    separator=',',
    append=True,
)

# early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=7,
    verbose=0,
    mode='auto',
)

# tensorboard callback
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir="./tmp/logs" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S"),
    histogram_freq=1,
    write_graph=True,
    update_freq='epoch',
)


"""
LOAD PRE-TRAINED MODEL WEIGHTS
"""

# Load pre-trained model weights before training
best_weights = "./saved_results/Models/MobileNet_V2/checkpoint"
mobilenet_model.load_weights(best_weights)


"""
TRAIN THE MODEL
"""

history = mobilenet_model.fit(
    train_gen,
    epochs=num_epochs,
    validation_data=val_gen,
    callbacks=[
        checkpoint_callback,
        csv_logger,
        early_stopping,
        tb_callback,
    ]
)


"""
EVALUATE THE MODEL
"""

# Load best weights seen during training
mobilenet_model.load_weights(checkpoint_filepath)

# Evaluate the model
loss, accuracy, top_five_accuracy, top_ten_accuracy, top_hundred_accuracy = mobilenet_model.evaluate(test_gen)
accuracy = round(accuracy * 100, 2)
top_five_accuracy = round(top_five_accuracy * 100, 2)
top_ten_accuracy = round(top_ten_accuracy * 100, 2)
top_hundred_accuracy = round(top_hundred_accuracy * 100, 2)
print(f"Accuracy on the test set: {accuracy}%.")
print(f"Top 5 Accuracy on the test set: {top_five_accuracy}%.")
print(f"Top 10 Accuracy on the test set: {top_ten_accuracy}%.")
print(f"Top 100 Accuracy on the test set: {top_hundred_accuracy}%.")


"""
HISTORY FIGURES
"""

# PLOTS
# Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./tmp/model accuracy.png')
plt.close()
# Top 5 accuracy
plt.plot(history.history['top-5-accuracy'])
plt.plot(history.history['val_top-5-accuracy'])
plt.title('model top 5 accuracy')
plt.ylabel('top 5 accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./tmp/model top 5 accuracy.png')
plt.close()
# Top 10 accuracy
plt.plot(history.history['top-10-accuracy'])
plt.plot(history.history['val_top-10-accuracy'])
plt.title('model top 10 accuracy')
plt.ylabel('top 10 accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./tmp/model top 10 accuracy.png')
plt.close()
# Top 100 accuracy
plt.plot(history.history['top-100-accuracy'])
plt.plot(history.history['val_top-100-accuracy'])
plt.title('model top 100 accuracy')
plt.ylabel('top 100 accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./tmp/model top 100 accuracy.png')
plt.close()
# Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./tmp/model loss.png')
plt.close()
