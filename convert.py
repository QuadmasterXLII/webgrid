import keras
import tensorflowjs as tfjs

model = keras.models.load_model("web_linefinder")
tfjs.converters.save_keras_model(model, "tfjs_dir")