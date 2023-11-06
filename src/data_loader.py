import cv2, json, ijson, os
import numpy as np
import tensorflow as tf
import albumentations as A
from utils import sigmoid
from constants import coco_to_ohe_idx


class DataGenerator(tf.keras.utils.Sequence):
   def __init__(self, 
                filepath:dict,   # annotations
                config,
                num_batches,
                data_folder,        
                shuffle_keys=True,
                apply_augmentation=True, # (excluding resize)
                ): 
      self.annotations = DataGenerator.get_annotations(filepath)
      self.batch_size = config.batch_size
      self.split_size = config.split_size
      self.image_size = config.image_size
      self.classes = config.classes
      self.anchors = np.array(config.anchors, np.float32)

      self.num_batches = num_batches
      self.data_folder = data_folder
      self.shuffle_keys = shuffle_keys
      self.apply_augmentation = apply_augmentation
      # normalize anchors
      self.anchors[..., 0] = self.anchors[..., 0] / self.image_size[1]
      self.anchors[..., 1] = self.anchors[..., 1] / self.image_size[0]
      self.n_anchors = self.anchors.shape[0]
      self.n_classes = len(self.classes)
      self.annotations_keys = np.array(list(self.annotations.keys()))
      self.n_annotations = len(self.annotations)
      self.transformer = self.get_transformer()
      self.on_epoch_end(create_indices=True)


   @staticmethod
   def get_annotations(annotations_path="annotations.json"):
      """Returns train/test annotations from annotations file.
      """
      with open(annotations_path) as file:
         annotations = json.load(file)
      return annotations


   def on_epoch_end(self, create_indices=False):
      """Choices new indices from annotations.
         Notice: this method selecting {num_batches} batches randomly from whole dataset."""
      if self.shuffle_keys or create_indices:
         data_size = len(self.annotations)
         samples = int(self.batch_size*self.num_batches)
         self.indices = np.random.choice(range(data_size), size=samples, replace=False)
   

   def __len__(self):
      return self.indices.shape[0] // self.batch_size


   def __getitem__(self, index):
      """Generate one batch of data."""
      if index == len(self):
         raise ValueError(f"Not enough data: {len(self.annotations)} to load one batch of size: {self.batch_size}")
      indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
      keys = [self.annotations_keys[idx] for idx in indices]
      X, y, original_images = self.__data_generation(keys)
      self.previous_batch_info = {"original_images": original_images}
      return X, y


   def __data_generation(self, keys):
      """Generates data containing batch_size samples."""
      X = np.zeros(shape=(self.batch_size, *self.image_size, 3))
      y = np.zeros(shape=(self.batch_size, self.split_size, self.split_size, self.n_anchors, 5+self.n_classes))
      original_images = []
      for i, key in enumerate(keys): 
         image, bboxes, class_labels = self.load_data(key)
         aug_image, bboxes  = self.apply_transformer(image, bboxes, class_labels)
         bboxes, anchors_indices = self.match_anchors(bboxes)
         label = self.preprocess_label(bboxes, anchors_indices)
         X[i] = aug_image
         y[i] = label
         original_images.append(image)
      return X, y, original_images


   def load_data(self, image_name):
      """Returns original image with normalized bboxes to [0, 1].
      """
      image = cv2.imread(f"{self.data_folder}{image_name}")
      height, width, _ = image.shape
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      bboxes = self.annotations[image_name]["bboxes"]
      normalized_bboxes = []
      class_labels = []
      for bbox in bboxes: # bbox = [xmin, ymin, w, h, class_id]
         w = bbox[2] / width
         h = bbox[3] / height
         x_center = (bbox[0] + bbox[2]/2.) / width
         y_center = (bbox[1] + bbox[3]/2.) / height
         # append
         normalized_bboxes.append([x_center, y_center, w, h])
         class_labels.append(coco_to_ohe_idx[int(bbox[4])])
      return image, normalized_bboxes, class_labels


   def match_anchors(self, bboxes):
      """Matches a bbox to the most suitable anchor and
         transforms width and height to correct format.
      """
      anchors_indices = []
      for i, bbox in enumerate(bboxes):
         w, h = bbox[2], bbox[3]
         # match anchor
         anchor_idx = None
         min_diff = float("inf")
         for idx, anchor in enumerate(self.anchors):
            anchor_w = anchor[0]
            anchor_h = anchor[1]
            diff = abs(anchor_w - w) + abs(anchor_h - h)
            if diff < min_diff:
               min_diff = diff
               anchor_idx = idx
         # preprocess w and h
         anchor = self.anchors[anchor_idx]
         w_ratio = w/anchor[0]
         h_ratio = h/anchor[1]
         # convert to model output
         w = sigmoid(np.log(w_ratio))
         h = sigmoid(np.log(h_ratio))
         bboxes[i][2] = w 
         bboxes[i][3] = h
         anchors_indices.append(anchor_idx)
      return bboxes, anchors_indices


   def preprocess_label(self, bboxes, anchors_indices):
      """Creates a {split_size}x{split_size} grid where each square is labeled individually 
         based on the provided split_size.
      """
      label = np.zeros((self.split_size, self.split_size, self.n_anchors, self.n_classes+5))
      for k in range(len(bboxes)):
         x, y, w, h, ohe = bboxes[k]
         a_idx = anchors_indices[k]
         cell_x = x*self.split_size
         cell_y = y*self.split_size
         i, j = int(cell_x), int(cell_y)
         label[j, i, a_idx, 0:5] = [1., cell_x%1, cell_y%1, w, h]
         label[j, i, a_idx, 5:] = ohe
      return label


   def apply_transformer(self, image, bboxes, class_labels):
      """Applays augmentations on images and bbox."""
      bboxes = np.clip(bboxes, 1e-3, 1-(1e-3), dtype=np.float32) 
      transformed = self.transformer(image=image, bboxes=bboxes, class_labels=class_labels)
      image = transformed['image']
      bboxes = transformed['bboxes']
      class_labels = transformed['class_labels']

      for i, bbox in enumerate(bboxes):
         bbox = list(bbox)
         bbox.append(class_labels[i])
         bbox[-1] = tf.one_hot(bbox[-1], depth=self.n_classes)
         bboxes[i] = bbox
      return image/255, bboxes


   def get_transformer(self):
      if self.apply_augmentation:
         return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=1),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.9),
            A.SmallestMaxSize(max_size=384),
            A.RandomCrop(width=self.image_size[1], height=self.image_size[0], p=0.5),
            A.GaussNoise(p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.CLAHE(clip_limit=2),
            A.ImageCompression(quality_lower=60),
            A.Resize(width=self.image_size[1], height=self.image_size[0])
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],
                                       min_visibility=0.50, min_area=100))

      else:
         return A.Compose([
            A.Resize(width=self.image_size[1], height=self.image_size[0])
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],
                                        min_visibility=0.50, min_area=100))


def get_generators(config):
   train_gen = DataGenerator(filepath=config.train_extracted_annotations,
                          config=config, 
                          num_batches=config.train_batches,
                          data_folder=config.train_dataset)


   val_gen = DataGenerator(filepath=config.val_extracted_annotations,
                           config=config,
                           num_batches=config.val_batches,
                           shuffle_keys=False,
                           apply_augmentation=False,
                           data_folder=config.val_dataset)
   return train_gen, val_gen



def extract_coco(filename, save_filename):
   """Extracts the most important information from COCO annotations and saves it to a JSON file. 
   """
   annotations = {}
   uniq_classes = set()
   with open(filename, 'r') as file:
      objects = ijson.items(file, 'annotations.item')
      for obj in objects:
         is_crowd = obj["iscrowd"]
         class_id = obj["category_id"]
         class_id = int(class_id)-1 # class ids starts from 1
         uniq_classes.add(class_id)
         if is_crowd:
            continue
         image_id = obj["image_id"]
         image_id = "0"*(12-len(str(image_id))) + str(image_id) + ".jpg"
         bbox = obj["bbox"]
         bbox = [float(x) for x in bbox]
         bbox.append(class_id)
         if image_id not in annotations:
            annotations[image_id] = {"bboxes": [bbox]}
         else:
            annotations[image_id]["bboxes"].append(bbox)

   with open(filename, 'r') as file:
      objects = ijson.items(file, 'images.item')
      for obj in objects:
         image_id = obj["id"]
         image_id = "0"*(12-len(str(image_id))) + str(image_id) + ".jpg"
         if image_id in annotations:
            annotations[image_id]["height"] = obj["height"]
            annotations[image_id]["width"] = obj["width"]

   # save
   with open(f"{save_filename}", 'w') as f:
      json.dump(annotations, f)
   return uniq_classes


def extract_annotations(config, extract_val=True, extract_train=True):
   if extract_train:
      if not os.path.isfile(config.train_extracted_annotations):
         extract_coco(f"{config.train_annotations}", save_filename=config.train_extracted_annotations)
   if extract_val:
      if not os.path.isfile(config.val_extracted_annotations):
         extract_coco(f"{config.val_annotations}", save_filename=config.val_extracted_annotations)
