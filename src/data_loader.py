import cv2, json
import numpy as np, albumentations as A
from tensorflow.keras.utils import Sequence


class MetaData: 
   @classmethod      
   def return_train_data(cls, image, annotation, max_bboxes):
      image, bboxes = MetaData._preprocess_data(image, annotation)
      label = MetaData.create_label(bboxes, max_bboxes)
      return image, label
   

   @classmethod      
   def return_test_data(cls, original_image, annotation, max_bboxes):
      image, bboxes = MetaData._preprocess_data(image, annotation)
      label = MetaData.create_label(bboxes, max_bboxes)
      return image, label, original_image


   @classmethod
   def update(cls, config):
      cls.target_image_size = config.target_image_size
      cls.n_classes = config.n_classes 

      cls.transform = A.Compose([
         A.HorizontalFlip(p=0.5),
         A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, p=0.75),
         A.SmallestMaxSize(max_size=int(np.max(config.target_image_size))),
         A.RandomCrop(width=config.target_image_size[1], height=config.target_image_size[0], p=0.5),
         A.GaussNoise(var_limit=(10., 30.), p=0.5),
         A.Blur(blur_limit=3, p=0.2),
         A.CLAHE(clip_limit=1),
         A.ImageCompression(quality_lower=80),
         A.Resize(width=config.target_image_size[1], height=config.target_image_size[0]),
         A.Normalize(mean=config.mean, std=config.std)
         ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],
                                          min_visibility=0.40, min_area=16))


   @classmethod
   def _preprocess_data(cls, image, annotation):
      """Transforms image with keypoints"""
      w, h = annotation["width"], annotation["height"]
      bboxes = np.array(annotation["bboxes"], dtype=np.float32)
      classes = np.array(annotation["classes"], dtype=np.int32)

      bboxes[:, [0, 2]] /= w
      bboxes[:, [1, 3]] /= h
      bboxes = np.clip(bboxes, 1e-8, 1.)

      transformed = cls.transform(image=image, bboxes=bboxes, class_labels=classes)
      image = transformed['image']
      bboxes = np.array(transformed['bboxes'])
      classes = np.array(transformed['class_labels'])

      if bboxes.shape[0] == 0:
         return image, bboxes
      
      bboxes = np.concatenate([bboxes, np.expand_dims(classes, axis=-1)], axis=-1)
      return image, bboxes


   @classmethod
   def create_label(cls, annotations, max_bboxes):
      label = np.zeros((max_bboxes, 5 + cls.n_classes), dtype=np.float32)

      for i, bbox in enumerate(annotations):
         xc, yc, w, h, cls_id = bbox

         _xc = xc*cls.target_image_size[1]
         _yc = yc*cls.target_image_size[0]
         _w = w*cls.target_image_size[1]
         _h = h*cls.target_image_size[0]

         # assign
         ohe = np.zeros((cls.n_classes), np.float32)
         ohe[int(cls_id)] = 1

         label[i] = [1, _xc, _yc, _w, _h, *ohe]
      
      return label
      

class DataGenerator(Sequence):
   def __init__(self, 
                annotations_path:str, 
                data_folder:str, 
                batches:int, 
                shuffle:bool=True):
      self.annotations = DataGenerator.load_annotations(annotations_path)
      self.data_folder = data_folder
      self.batches = batches
      self.shuffle = shuffle

      self.on_epoch_end(create_indices=True)


   def __len__(self):
      return len(self.indices) // DataGenerator.batch_size


   @classmethod
   def update(cls, config):
      cls.n_classes = config.n_classes
      cls.batch_size = config.batch_size
      cls.target_image_size = config.target_image_size

 

   @staticmethod
   def load_annotations(annotations_path="annotations.json"):
      """Returns train/test annotations from annotations file.
      """
      with open(annotations_path) as file:
         annotations = json.load(file)

      return annotations


   def on_epoch_end(self, create_indices=False):
      """Choices new indices from annotations.
         Notice: this method selecting {num_batches} batches randomly from whole dataset.
      """
      if self.shuffle or create_indices:
         data_size = len(self.annotations)
         samples = int(DataGenerator.batch_size*self.batches)
         self.indices = np.random.choice(range(data_size), size=samples, replace=False)


   def __getitem__(self, index):
      if index > len(self):
         print(f"Index is greater than size of generator: {index} > {len(self)}")
      start_idx = index * DataGenerator.batch_size
      end_idx = (index + 1) * DataGenerator.batch_size
      batch_indices = self.indices[start_idx:end_idx]

      X = np.zeros((DataGenerator.batch_size, *DataGenerator.target_image_size, 3), 
                    dtype=np.float32)

      _img_name  = []
      max_bboxes = 0
      for i, idx in enumerate(batch_indices):
         image_name = list(self.annotations.keys())[int(idx)]
         annotation = self.annotations[image_name]
         
         _img_name.append(image_name)
         max_bboxes = max(max_bboxes, len(annotation["bboxes"]))
         
      y = np.zeros((DataGenerator.batch_size, max_bboxes, 5 + DataGenerator.n_classes), dtype=np.float32)

      for i, name_id in enumerate(_img_name):
         annotation = self.annotations[name_id]
         image = cv2.imread(f"{self.data_folder}{name_id}")

         image, label = MetaData.return_train_data(image, annotation, max_bboxes)
         
         y[i] = label
         X[i] = image

      return X, y


   def get_test_batch(self, index):
      if index > len(self):
         print(f"Index is greater than size of generator: {index} > {len(self)}")
      start_idx = index * DataGenerator.batch_size
      end_idx = (index + 1) * DataGenerator.batch_size
      batch_indices = self.indices[start_idx:end_idx]

      X = np.zeros((DataGenerator.batch_size, *DataGenerator.target_image_size, 3), 
                    dtype=np.float32)
      
      original_images = []

      _img_name  = []
      max_bboxes = 0
      for i, idx in enumerate(batch_indices):
         image_name = list(self.annotations.keys())[int(idx)]
         annotation = self.annotations[image_name]
         
         _img_name.append(image_name)
         max_bboxes = max(max_bboxes, len(annotation["bboxes"]))
         
      y = np.zeros((DataGenerator.batch_size, max_bboxes, 5 + DataGenerator.n_classes), dtype=np.float32)

      for i, name_id in enumerate(_img_name):
         annotation = self.annotations[name_id]
         image = cv2.imread(f"{self.data_folder}{name_id}")

         original_images.append(image)

         image, label = MetaData.return_train_data(image, annotation, max_bboxes)
         
         y[i] = label
         X[i] = image

      return X, y, original_images


