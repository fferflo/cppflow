import tensorflow as tf
import os

model = tf.keras.applications.EfficientNetB0()

# Export the model to a SavedModel
model.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), "model"), save_format='tf')

print("Output shape: " + str(model.layers[-1].output.shape))
