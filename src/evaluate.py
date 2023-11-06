from metrics import YoloLoss, MeanIou, GlobalIou, ClassRecoil, AnchorRecoil, ConfidenceAccuracy
from tensorflow.keras.models import load_model
from data_loader import DataGenerator
from utils import set_memory_growth, extract_annotations
from config import Config

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
              metrics=[MeanIou(cfg.anchors, cfg.split_size, cfg.threshold), 
                       GlobalIou(cfg.anchors, cfg.split_size, cfg.threshold),
                       ConfidenceAccuracy(cfg.threshold),
                       ClassRecoil(),
                       AnchorRecoil()])

model.evaluate(val_gen)
# loss: 95.3270 - mean_iou: 0.1375 - global_iou: 0.1334 - lambda_acc: 0.9606 - class_rec: 0.6996 - anchor_rec: 0.7454