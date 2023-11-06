import json
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans


class BBoxParser:
   def __init__(self, config):
      self.threshold = config.threshold
      self.split_size = config.split_size
      self.target_image_size = config.image_size
      self.anchors =  config.anchors
      self.threshold_duplicates = config.threshold_iou


   def parse_batch(self, model_output, return_raw=False):
      # (B, split, split, anchors, 5+n_classes) -> (B, split, split, 7)
      data = BBoxParser.extract_anchors(model_output)
      # (B, split, split, 7) -> (B, N, 7) 
      data = self.extract_and_denormalize_bboxes(data.numpy())
      if not return_raw:
         # drop duplicates, (B, N', 7)
         data = self.clear(data)
      return data


   @staticmethod
   def calculate_iou(box1, box2):
      box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
      box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

      x1 = max(box1[0], box2[0])
      y1 = max(box1[1], box2[1])
      x2 = min(box1[2], box2[2])
      y2 = min(box1[3], box2[3])

      intersection = max(0, x2 - x1) * max(0, y2 - y1)

      box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
      box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

      union = box1_area + box2_area - intersection

      return intersection / union if union > 0 else 0


   @staticmethod
   def overlap(rect1, rect2):
      x1, y1, w1, h1 = rect1
      x2, y2, w2, h2 = rect2

      if x1 > x2 + w2 or x2 > x1 + w1:
         return False
      if y1 > y2 + h2 or y2 > y1 + h1:
         return False
      return True


   @staticmethod
   def extract_anchors(bboxes):
      """shape: (B, split, split, anchors, 5+classes) --> shape: (B, split, split, 7) 
         The 7 values are:
         >>> (confidence, xcenter, ycenter, width_ratio, height_ratio, class_idx, anchor_idx)"""
      confidence_scores = bboxes[..., 0]
      indices = tf.argmax(confidence_scores, axis=-1)
      best_anchors = tf.cast(tf.gather(bboxes, indices, batch_dims=3), tf.float32)

      class_ohe = best_anchors[..., 5:]
      class_idx = tf.cast(tf.math.argmax(class_ohe, axis=-1), tf.float32)
      confidence = best_anchors[..., 0]
      xcenter = best_anchors[..., 1]
      ycenter = best_anchors[..., 2]
      width = best_anchors[..., 3]
      height = best_anchors[..., 4]
      anchor_idx = tf.cast(tf.expand_dims(indices, axis=-1), tf.float32)

      epsilon = 1e-4
      height = tf.clip_by_value(height, epsilon, 1 - epsilon)
      width = tf.clip_by_value(width, epsilon, 1 - epsilon)

      width_ratio = tf.cast(tf.math.exp(-tf.math.log(1/width - 1)), tf.float32)
      height_ratio = tf.cast(tf.math.exp(-tf.math.log(1/height - 1)), tf.float32)

      bboxes = tf.concat([tf.expand_dims(confidence, axis=-1), tf.expand_dims(xcenter, axis=-1), 
                              tf.expand_dims(ycenter, axis=-1), tf.expand_dims(width_ratio, axis=-1), 
                              tf.expand_dims(height_ratio, axis=-1),  tf.expand_dims(class_idx, axis=-1),
                              anchor_idx], axis=-1)
      return bboxes
   

   def extract_and_denormalize_bboxes(self, bboxes):
      """Takes extracted bboxes in shape (B, split, split, 7) and
         returns only those bboxes which passed the confidence threshold.
         Output shape: (B, N, 6), where N is number of valid bboxes.
         (v, x, y, w, h, c ). (x, y, w, h) are denormalized.
      """
      scaler_x = np.arange(self.split_size)
      scaler_x = np.tile(scaler_x, reps=[self.split_size, 1])
      scaler_y = scaler_x.T

      scaler_x = np.expand_dims(scaler_x, axis=-1)
      scaler_y = np.expand_dims(scaler_y, axis=-1)

      scaler = np.concatenate([scaler_x, scaler_y], axis=-1)

      # denormalize x, y: [0, split_size] -> [0, 1]
      bboxes[..., 1:3] = (bboxes[..., 1:3] + scaler) / self.split_size
      bboxes[..., 1:3] = bboxes[..., 1:3] * self.target_image_size

      bboxes = bboxes.reshape(-1, self.split_size**2, 7)

      # confidence threshold
      output = []
      for b in bboxes:
         mask = b[..., 0] > self.threshold
         b = b[mask] 
         output.append(b)

      # denormalize w, h
      for i, bboxes in enumerate(output):
         for anchor_id, anchor in enumerate(self.anchors):
            mask = bboxes[:, 6] == anchor_id
            bboxes[mask, 3:5] *= anchor 
         output[i] = bboxes[..., :6]
      return output 
   

   def clear(self, bboxes):
      """Because there are often identical predicted bboxes for the same
         object, this method will filter those bboxes.
      """
      def preprocess_sample(sample):
         removed_bboxes = set()
         for i, bbox1 in enumerate(sample):
            for j, bbox2 in enumerate(sample):
               if i in removed_bboxes or j in removed_bboxes:
                  continue
               if i == j:
                  continue
               if bbox1[5] != bbox2[5]: 
                  continue
               if not BBoxParser.overlap(bbox1[1:5], bbox2[1:5]):
                  continue

               iou = BBoxParser.calculate_iou(bbox1[1:5], bbox2[1:5])
               if iou > self.threshold_duplicates:
                  # pick the one with higher area
                  area1 = bbox1[2]*bbox1[3]
                  area2 = bbox2[2]*bbox2[3]
                  if area1 > area2:
                     removed_bboxes.add(j)
                  else:
                     removed_bboxes.add(i)
         
         filtered_samples = []
         for i, bbox in enumerate(sample):
            if i in removed_bboxes:
               continue
            filtered_samples.append(bbox)
         return filtered_samples

      filtered_batch = []
      for sample in bboxes:
         filtered_batch.append(preprocess_sample(sample))
      return filtered_batch



class ImageGeneratorCallback(tf.keras.callbacks.Callback):
   def __init__(self, output_dir, model, datagen, config):
      super(ImageGeneratorCallback, self).__init__()
      self.output_dir = output_dir
      self.model = model
      self.datagen = datagen
      self.plot_size = (4, 4)

      self.parser = BBoxParser(config)
      self.classes = config.classes
      self.anchors = config.anchors
      self.image_size = config.image_size
      self.split_size = config.split_size
      self.threshold = config.threshold


   def on_epoch_end(self, epoch, logs=None):
      """This function is called at the end of each epoch during training. 
         It visualizes the results of the model on a batch of images.
      """
      rows, cols = self.plot_size

      bboxes, labels, original_images = [], [], []
      while len(bboxes) < rows*cols:
         idx = np.random.randint(0, len(self.datagen), size=1)[0]
         images, _labels = self.datagen[idx]

         out = self.model(images, training=False)
         for p in _labels:
            labels.append(p)
         for p in out:
            bboxes.append(p)
         # grab original images
         for p in self.datagen.previous_batch_info["original_images"]:
            original_images.append(p)

      labels = np.array(labels[:rows*cols])
      bboxes = np.array(bboxes[:rows*cols])
      original_images = original_images[:rows*cols]

      pred_bboxes = self.parser.parse_batch(bboxes, return_raw=True)
      true_bboxes = self.parser.parse_batch(labels, return_raw=True)

      # plot output
      _, axs = plt.subplots(rows, cols, figsize=(40, 40))
      for i in range(rows):
         for j in range(cols):
            original_size = original_images[i * rows + j].shape[:2]
            axs[i, j].imshow(original_images[i * rows + j])
            axs[i, j].axis('off')
            for bbox in pred_bboxes[i * rows + j]:
               confidence, x, y, w, h, class_id = bbox

               xmin, ymin = x-w/2, y-h/2
               # model input size --> original image size
               xmin *= original_size[1]/self.image_size[1]
               ymin *= original_size[0]/self.image_size[0]
               w *= original_size[1]/self.image_size[1]
               h *= original_size[0]/self.image_size[0]
      
               obj_name = self.classes[class_id]

               rect = Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor=(1, 0, 0), facecolor='none')
               axs[i, j].add_patch(rect)
               text = f"{confidence:.2f}  {obj_name}"
               axs[i, j].text(xmin, ymin - 5, text, color=(1, 0, 0), fontsize=6, backgroundcolor='none')
            
            for bbox in true_bboxes[i * rows + j]:
               confidence, x, y, w, h, class_id = bbox
               
               xmin, ymin = x-w/2, y-h/2
               # model input size --> original image size
               xmin *= original_size[1]/self.image_size[1]
               ymin *= original_size[0]/self.image_size[0]
               w *= original_size[1]/self.image_size[1]
               h *= original_size[0]/self.image_size[0]

               obj_name = self.classes[class_id]

               rect = Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor=(0, 1, 0), facecolor='none')
               axs[i, j].add_patch(rect)
               text = f"{confidence:.2f}  {obj_name}"
               axs[i, j].text

      plt.savefig(f'{self.output_dir}/{epoch}.png')
      plt.close()



def sigmoid(x):
   return 1 / (1 + np.exp(-x))


def calculate_anchor_boxes(filepath, target_size, anchors=5):
   """This function calculates anchor boxes for object detection 
      from pre-extracted annotations.
   """
   with open(filepath) as file:
      annotations = json.load(file)
   all_bboxes = []
   for _, value in annotations.items():
      bboxes = value["bboxes"]
      width = value["width"]
      height = value["height"]
      # normalize bboxes to [0, 1]
      norm_bboxes = []
      for bbox in bboxes:
         norm_bboxes.append([bbox[2]/width, bbox[3]/height])
      all_bboxes += norm_bboxes

   bboxes = np.array(all_bboxes, np.float32)
   print(f"Founded bboxes: {bboxes.shape[0]}\nCalculating anchor boxes...")
   # Calculate anchor boxes
   kmeans = KMeans(n_clusters=anchors, n_init=5, random_state=0).fit(bboxes)
   anchor_boxes = kmeans.cluster_centers_
   # denormalize
   anchor_boxes[..., 0] = anchor_boxes[..., 0] * target_size[1]
   anchor_boxes[..., 1] = anchor_boxes[..., 1] * target_size[0]
   anchor_boxes = np.round(anchor_boxes, 2)
   for anchor in anchor_boxes:
      print(anchor, f"Ratio w/h= {anchor[0]/anchor[1]:.2f}")
   return anchor_boxes


def get_class_weights(filename):
   """Returns a dictionary with weights for each class
      from pre-extracted annotations.
   """
   with open(filename) as file:
      annotations = json.load(file)
   weights = {}
   for value in annotations.values():
      for bbox in value["bboxes"]:
         id_class = bbox[-1]
         if id_class in weights:
            weights[id_class] += 1
         else:
            weights[id_class] = 1
   n_bboxes = sum(weights.values())
   n_classes = len(weights.keys())
   for key, value in weights.items():
      weights[key] = round((n_bboxes-value)/n_bboxes * n_classes/(n_classes-1), 3)

   weights = [weights[i] for i in range(max(weights.keys())+1)
                     if i in weights]
   print(f"Class weights:\n", weights)
   return weights


def set_memory_growth():
   gpus = tf.config.list_physical_devices('GPU')
   if len(gpus) == 0:
      raise SystemError
   for gpu in gpus:
      print(gpu) 
      tf.config.experimental.set_memory_growth(gpu, True)