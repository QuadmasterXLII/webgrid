import keras
import tensorflowjs as tfjs

model = keras.models.load_model("mediocre_linefinder.h5")
tfjs.converters.save_keras_model(model, "tfjs_dir")