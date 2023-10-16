import os
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from data_loader import DataGenerator, get_annotations, extract_coco
from metrics import YoloLoss
from model import build_model
from utils import ImageGeneratorCallback, get_class_weights
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
epochs = 150
lr = 2e-4
threshold = 0.3         # Threshold for generating images to visualize.
 
num_batches = 1500      # Beacuse COCO daataset is very huge (around 118k images)
                        # on every epoch, {num_batches} batches will be randomly selected from all datasets.
val_batches = 100         

train_base = True       # Unlock ResNet101 for training?
load_checkpoint = True
checkpoint_name = "checkpoints/last"

annotations_folder = "D:/COCO 2017/annotations/"
train_dataset = "D:/COCO 2017/train2017/"
val_dataset = "D:/COCO 2017/val2017/"


### Preprocess annotations ###
if not os.path.isfile("train_annotations.json"):
   extract_coco(f"{annotations_folder}instances_train2017.json", save_filename="train_annotations.json")

if not os.path.isfile("val_annotations.json"):
   extract_coco(f"{annotations_folder}instances_val2017.json", save_filename="val_annotations.json")

train_annotations = get_annotations("train_annotations.json")
val_annotations = get_annotations("val_annotations.json")
print(f"Train images: {len(train_annotations)}")
print(f"Validation images: {len(val_annotations)}")


classes = {i: value for i, value in enumerate(available_class.values())}
class_weights = get_class_weights("train_annotations.json")


### Create generators ###
train_gen = DataGenerator(annotations=train_annotations, 
                          batch_size=batch_size,
                          split_size=split_size,
                          image_size=image_size,
                          num_batches=num_batches,
                          classes=classes,
                          anchors=anchors,
                          data_folder=train_dataset)


val_gen = DataGenerator(annotations=val_annotations,
                        batch_size=batch_size,
                        split_size=split_size,
                        image_size=image_size,
                        classes=classes,
                        num_batches=val_batches,
                        anchors=anchors,
                        shuffle_keys=False,
                        apply_augmentation=False,
                        data_folder=val_dataset)


### Load/build model ###
if load_checkpoint:
   model = load_model(f'{checkpoint_name}.h5', custom_objects={'YoloLoss': YoloLoss})
else:
   model = build_model((*image_size, 3), split_size, len(anchors), len(classes))

if train_base:
   base_model = model.get_layer("resnet101")
   base_model.trainable = True
model.summary()

### Callbacks ###
igc = ImageGeneratorCallback(output_dir="train_output", 
                             model=model, 
                             datagen=val_gen, 
                             split_size=split_size, 
                             threshold=threshold, 
                             image_size=image_size,
                             classes=classes,
                             anchors=anchors)
model_checkpoint = ModelCheckpoint(filepath='checkpoints\last.h5', 
                                   save_best_only=False, 
                                   save_freq='epoch', 
                                   verbose=1)

tensorboard_callback = TensorBoard(log_dir=f"./logs/{datetime.now().strftime('%d_%H-%M')}")
### Fit the model ###
model.compile(optimizer=Adam(lr),
              loss=YoloLoss(image_size=image_size, 
                            split_size=split_size,
                            class_weights=class_weights))


model.fit(train_gen, 
          validation_data=val_gen,
          epochs=epochs, 
          batch_size=batch_size, 
          callbacks=[igc, model_checkpoint, tensorboard_callback])

