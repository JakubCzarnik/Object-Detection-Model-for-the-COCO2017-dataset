from utils import set_memory_growth, get_generators, extract_annotations
from models.model import DetectionModel
from metrics import DetectionLoss, IouMetric, F1Score
from config import Config
import tensorflow as tf


cfg = Config()
cfg.val_batches_per_epoch = 4800//cfg.batch_size

set_memory_growth()

### Preprocess annotations ###
extract_annotations(cfg)


### Create generators ###
train_gen, val_gen = get_generators(cfg)


### build model ###
detector = DetectionModel(cfg)

optimizer=tf.keras.optimizers.Adam(learning_rate=0)

detector.compile(optimizer=optimizer,
              loss=DetectionLoss(cfg),
              metrics=[IouMetric(),
                       F1Score(threshold=0.6)]
                       )

### Callbacks ###
checkpoint = tf.train.Checkpoint(optimizer=optimizer, 
                                 model=detector.base_model)
manager = tf.train.CheckpointManager(checkpoint, 
                                     cfg.checkpoint_save_path, 
                                     max_to_keep=cfg.checkpoints_to_keep)



# load checkpoint
detector.base_model.summary()
if cfg.load_last_checkpoint:
   print("Loading checkpoint...")
   checkpoint.restore(manager.latest_checkpoint) 
elif cfg.load_checkpoint:
   print("Loading checkpoint...")
   checkpoint.restore(cfg.checkpoint_load_path)

### train the model ###
detector.train_step = detector.test_step

detector.fit(val_gen,
            epochs=1)
