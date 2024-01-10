from config import Config
from utils import set_memory_growth, get_generators, extract_annotations
from models.model import DetectionModel
from metrics import DetectionLoss, IouMetric, F1Score
from callbacks import MapsCompareCallback, SaveCallback
from datetime import datetime 
import tensorflow as tf


cfg = Config()
set_memory_growth()

### Preprocess annotations ###
extract_annotations(cfg)

### Create generators ###
train_gen, val_gen = get_generators(cfg)

### build and compile the model ###
detector = DetectionModel(cfg)

optimizer=tf.keras.optimizers.Adam(cfg.learning_rate)
detector.compile(optimizer=optimizer,
                  loss=DetectionLoss(cfg),
                  metrics=[IouMetric(),
                           F1Score(threshold=0.6)])

### Create callbacks ###
callbacks = []
checkpoint = tf.train.Checkpoint(optimizer=optimizer, 
                                 model=detector.base_model)

manager = tf.train.CheckpointManager(checkpoint, 
                                     cfg.checkpoint_save_path, 
                                     max_to_keep=cfg.checkpoints_to_keep
                                     )
saver = SaveCallback(checkpoint, manager)
callbacks.append(saver)
mpc = MapsCompareCallback(val_gen, cfg)
callbacks.append(mpc)

if cfg.run_tensorboard:
   import datetime
   folder_name = datetime.datetime.now().strftime("%d_%H-%M")
   tb = tf.keras.callbacks.TensorBoard(log_dir=f"logs/{folder_name}")
   callbacks.append(tb)

### load checkpoint ###
if cfg.load_last_checkpoint:
   print("Loading checkpoint...")
   checkpoint.restore(manager.latest_checkpoint) 
elif cfg.load_checkpoint:
   print("Loading checkpoint...")
   checkpoint.restore(cfg.checkpoint_load_path)

### train the model ###
detector.base_model.summary()
detector.fit(train_gen, 
            validation_data=val_gen,
            epochs=cfg.epochs, 
            callbacks=callbacks)
