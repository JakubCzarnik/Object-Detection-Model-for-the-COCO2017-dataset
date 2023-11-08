from datetime import datetime 
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from data_loader import get_generators, extract_annotations
from utils import ImageGeneratorCallback, get_class_weights, set_memory_growth
from model import build_model
from metrics import YoloLoss
from config import Config


cfg = Config()

set_memory_growth()

### Preprocess annotations ###
extract_annotations(cfg)
class_weights = get_class_weights(cfg.train_extracted_annotations)


### Create generators ###
train_gen, val_gen = get_generators(cfg)


### Load/build model ###
if cfg.load_checkpoint:
   model = load_model(f'{cfg.checkpoint_name}.h5', custom_objects={'YoloLoss': YoloLoss})
else:
   model = build_model(cfg)


model.get_layer("EfficientNetV2B0").trainable = cfg.train_base
model.summary()

### Callbacks ###
igc = ImageGeneratorCallback(output_dir="train_output", 
                             model=model, 
                             datagen=val_gen, 
                             config=cfg)

model_checkpoint = ModelCheckpoint(filepath="checkpoints\last.h5", 
                                 monitor="val_loss",
                                 save_best_only=True, 
                                 save_freq='epoch', 
                                 verbose=1)

tensorboard_callback = TensorBoard(log_dir=f"./logs/{datetime.now().strftime('%d_%H-%M')}")
### Fit the model ###
model.compile(optimizer=tf.keras.optimizers.Adam(cfg.lr),
              loss=YoloLoss(image_size=cfg.image_size, 
                            split_size=cfg.split_size,
                            class_weights=class_weights))


model.fit(train_gen, 
          validation_data=val_gen,
          epochs=cfg.epochs, 
          batch_size=cfg.batch_size, 
          callbacks=[model_checkpoint, igc, tensorboard_callback])
