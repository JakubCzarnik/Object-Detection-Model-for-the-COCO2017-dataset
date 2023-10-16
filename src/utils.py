import json
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans

class ImageGeneratorCallback(tf.keras.callbacks.Callback):
   def __init__(self, 
               output_dir,      # path where to save the output
               model,           # model
               datagen,         # data generator
               classes,         # list of class names
               anchors,         # list of anchors
               image_size,      # image size: (height, width)
               split_size,      # creates a {split_size}x{split_size} grid where each square has its own label
               threshold,       # bboxes with less confidence will not be displayed
               plot_size=(4,4)  # size of images grid 
               ):
      super(ImageGeneratorCallback, self).__init__()
      self.output_dir = output_dir
      self.model = model
      self.datagen = datagen
      self.classes = classes
      self.anchors = anchors
      self.image_size = image_size
      self.split_size = split_size
      self.threshold = threshold
      self.plot_size = plot_size


   def on_epoch_end(self, epoch, logs=None):
      """This function is called at the end of each epoch during training. 
         It visualizes the results of the model on a batch of images.
      """
      rows, cols = self.plot_size
      idx = np.random.randint(0, len(self.datagen), size=1)[0]
      images, labels = self.datagen[idx]
      if rows*cols > images.shape[0]:
         raise ValueError(f"Batch size is smaller than plots size: {rows*cols=}")
      info = self.datagen.previous_batch_info
      # preprocess bboxes
      output = self.model(images, training=False)
      pred_bboxes = extract_anchors(output)
      true_bboxes = extract_anchors(labels)
      original_images = info["original_images"]
      # plot output
      _, axs = plt.subplots(rows, cols, figsize=(40, 40))
      for i in range(rows):
         for j in range(cols):
            original_size = original_images[i * rows + j].shape[:2]
            axs[i, j].imshow(original_images[i * rows + j])
            axs[i, j].axis('off') 
            pred_text = self.plot_output(pred_bboxes, original_size, axs, i, j, color='r')
            true_text = self.plot_output(true_bboxes, original_size, axs, i, j, color='g')
            # print sum of bboxes
            axs[i, j].text(5, -10, pred_text, color="r", fontsize=12, backgroundcolor='white')
            axs[i, j].text(5, -30, true_text, color="g", fontsize=12, backgroundcolor='white')

      plt.savefig(f'{self.output_dir}/{epoch}.png')
      plt.close()


   def plot_output(self, bboxes, original_size, axs, i, j, color='g'):
      rows, _ = self.plot_size
      predicted_classes = {}
      # for each grid cell in image 
      for grid_h, row in enumerate(bboxes[i * rows + j]):
         for grid_w, bbox in enumerate(row):
            grid = (grid_h, grid_w)
            bbox = self.denormalize_bbox(bbox, original_size, grid)
            confidence, xmin, ymin, width, height, obj_name = bbox
            if confidence <= self.threshold:
               continue
            if obj_name in predicted_classes:
               predicted_classes[obj_name] += 1
            else:
               predicted_classes[obj_name] = 1
            rect = Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor=color, facecolor='none')
            axs[i, j].add_patch(rect)
            text = f"{confidence:.2f}  {obj_name}"
            axs[i, j].text(xmin, ymin - 5, text, color=color, fontsize=6, backgroundcolor='none')
      text = ""
      for key, value in predicted_classes.items():
         text += f"{key}: {value},  "
      return text[:-3]


   def denormalize_bbox(self, bbox, original_size, grid):
      """Converts normalized bbox to bbox with original values.
      """
      h, w = original_size
      grid_h, grid_w = grid
      confidence, xcenter, ycenter, w_ratio, h_ratio, class_id, anchor_id = bbox[:7]
      obj_name = self.classes[int(class_id)]
      anchor = self.anchors[int(anchor_id)]
      # denormalize
      xcenter = w/self.split_size * (grid_w + xcenter)
      ycenter = h/self.split_size * (grid_h + ycenter)

      width = (anchor[0]/self.image_size[1])*w_ratio * w
      height = (anchor[1]/self.image_size[0])*h_ratio * h
      xmin = (xcenter - (width / 2))
      ymin = (ycenter - (height / 2))
      return confidence, xmin, ymin, width, height, obj_name


def sigmoid(x):
   return 1 / (1 + np.exp(-x))


def extract_anchors(bboxes):
   """Returns bboxes in format:
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
   #
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

