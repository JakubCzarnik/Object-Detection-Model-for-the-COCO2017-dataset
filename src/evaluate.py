import os
import tensorflow as tf
from metrics import YoloLoss, MeanIou, GlobalIou, ClassRecoil, AnchorRecoil, ConfidenceAccuracy
from tensorflow.keras.models import load_model
from data_loader import DataGenerator, get_annotations, extract_coco
from utils import get_class_weights
from constants import available_class, anchors


gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
   raise SystemError
for gpu in gpus:
   print(gpu) 
   tf.config.experimental.set_memory_growth(gpu, True)

### Settings ###
image_size = (224, 224)
split_size = 7          # On every image will be generatet grid {split_size}x{split_size},
                        # where every cell has his own label.
batch_size = 16
threshold = 0.3         # Threshold for metrics
 
val_batches = 300         

checkpoint_name = "checkpoints/last"

annotations_folder = "D:/COCO 2017/annotations/"
val_dataset = "D:/COCO 2017/val2017/"

### Preprocess annotations ###
if not os.path.isfile("val_annotations.json"):
   extract_coco(f"{annotations_folder}instances_val2017.json", save_filename="val_annotations.json")


class_weights = get_class_weights("train_annotations.json")
classes = {i: value for i, value in enumerate(available_class.values())}


val_ann = get_annotations("val_annotations.json")
print(f"Validation images: {len(val_ann)}")

## Create generator ###
val_gen = DataGenerator(annotations=val_ann,
                        batch_size=batch_size,
                        split_size=split_size,
                        image_size=image_size,
                        classes=classes,
                        num_batches=val_batches,
                        anchors=anchors,
                        apply_augmentation=False,
                        data_folder=val_dataset)


### Load model ### 
model = load_model(f'{checkpoint_name}.h5', custom_objects={'YoloLoss': YoloLoss})
model.summary()

model.compile(optimizer="adam",
              loss=YoloLoss(image_size=image_size,
                            split_size=split_size,
                            class_weights=class_weights),
              metrics=[MeanIou(anchors, split_size, threshold), 
                       GlobalIou(anchors, split_size, threshold),
                       ConfidenceAccuracy(threshold),
                       ClassRecoil(),
                       AnchorRecoil()])

model.evaluate(val_gen)
