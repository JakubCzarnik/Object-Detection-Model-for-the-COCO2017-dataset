from metrics import YoloLoss, ClassMatchMetric, AnchorMatchMetric, ConfidenceF1Score, MeanIou
from tensorflow.keras.models import load_model
from data_loader import DataGenerator, extract_annotations
from utils import set_memory_growth, BBoxParser
from config import Config
import numpy as np

cfg = Config()

set_memory_growth()

### Preprocess annotations ###
extract_annotations(cfg, extract_train=False)


## Create generator ###
val_gen = DataGenerator(filepath="val_annotations.json",
                        config=cfg,
                        num_batches=cfg.val_batches,
                        shuffle_keys=False,
                        apply_augmentation=False,
                        data_folder=cfg.val_dataset)



### Load model ### 
model = load_model(f'{cfg.checkpoint_name}.h5', custom_objects={'YoloLoss': YoloLoss})
model.summary()

model.compile(optimizer="adam",
              loss=YoloLoss(image_size=cfg.image_size,
                            split_size=cfg.split_size,
                            class_weights=[1]*len(cfg.classes)),
              metrics=[ConfidenceF1Score(cfg.threshold),
                       ClassMatchMetric(cfg.threshold),
                       AnchorMatchMetric(cfg.threshold)])

model.evaluate(val_gen)

# evaluate mean iou
metric = MeanIou()
parser = BBoxParser(cfg)

indices = np.arange(len(val_gen))
for idx in indices:
   images, y_true = val_gen[idx]

   y_pred = model.predict(images, verbose=0)

   y_true = parser.parse_batch(y_true)
   y_pred = parser.parse_batch(y_pred)

   metric.update_state(y_true, y_pred)
   iou = metric.get_state()
   print("Iou: ", round(iou*100, 2), "%", end="\r")
# lambda_f1: 0.5565 - class_acc: 0.5858 - anchor_acc: 0.9240 IOU 32.03%