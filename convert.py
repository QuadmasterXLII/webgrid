import keras
import tensorflowjs as tfjs
import os
name = input()


model = keras.models.load_model("light_diverse_linefinder")
tfjs.converters.save_keras_model(model, "../webgrid/static/tfjs_dir")