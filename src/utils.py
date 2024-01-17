from data_loader import DataGenerator
from constants import cocoid_to_coco2017id
import json, ijson, os
import tensorflow as tf, numpy as np

class BBoxParser:
   @classmethod
   def update(cls, config):
      """
      Updates the class attribute based on the provided configuration.
      
      Args:
         config (Config): The configuration object.
      """
      cls.bbox_threshold = config.bbox_threshold


   @classmethod
   def parse_batch(cls, model_output):
      output = []

      # v,x,y,w,h,cls
      class_ids = np.argmax(model_output[..., 5:], axis=-1)
      model_output = np.concatenate([model_output[..., :5], np.expand_dims(class_ids, axis=-1)], axis=-1)
      
      for b in model_output:
         mask = b[..., 0] > cls.bbox_threshold
         b = b[mask]
         output.append(b)
      
      return output


def non_max_suppression(batch, max_output_size=200, iou_threshold=0.4):
   """
   This function performs non-maximum suppression (NMS) on the bounding boxes of detected objects.

   Args:
      batch (tensor): A tensor of shape (batch_size, max_boxes, 5+n_classes) representing the bounding boxes for each detected object.
      max_output_size (int, optional): The maximum number of boxes to keep after NMS.
      iou_threshold (float, optional): The IOU threshold for determining duplicate boxes.

   Returns:
      tensor: A tensor of shape (batch_size, max_boxes, 5+n_classes) representing the bounding boxes after NMS.
   """
   scores = batch[..., 0] # (batch_size, n)
   bboxes_xywh = batch[..., 1:5] # (batch_size, n, 4)
   bboxes = xywh2xyxy(bboxes_xywh)

   def single_image_nms(bboxes_scores):
      bboxes, scores = bboxes_scores
      selected_indices = tf.image.non_max_suppression(
         bboxes, scores, max_output_size, iou_threshold)

      # Create a mask of zeros the same size as scores
      mask = tf.zeros_like(scores, dtype=tf.bool)

      # Set the indices of the selected boxes to True
      mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(selected_indices, axis=-1), tf.ones_like(selected_indices, dtype=tf.bool))

      # Zero out the scores and bboxes that are not selected
      scores = tf.where(mask, scores, 0)
      scores = tf.expand_dims(scores, axis=-1)
      
      bboxes = tf.where(tf.tile(tf.expand_dims(mask, axis=-1), [1, 4]), bboxes, 0)
      bboxes = xyxy2xywh(bboxes)

      # Concatenate the scores and bboxes
      return tf.concat([scores, bboxes], axis=-1)

   selected_boxes_scores = tf.map_fn(single_image_nms, (bboxes, scores), dtype=tf.float32)

   selected_boxes_scores = tf.concat([selected_boxes_scores, batch[..., 5:]], axis=-1)
   
   batch = tf.tensor_scatter_nd_update(batch, tf.range(tf.shape(scores)[0])[:, None], selected_boxes_scores)
   return batch


def xywh2xyxy(boxes):
   """
   Convert bbox format from (xc, yc, w, h) to (xmin, ymin, xmax, ymax).

   Args:
      boxes: A tensor of shape (batch_size, n_boxes, 4).

   Returns:
      A tensor of shape (batch_size, n_boxes, 4) representing the boxes in (xmin, ymin, xmax, ymax) format.
   """
   xc, yc, w, h = tf.split(boxes, 4, axis=-1)
   xmin = xc - w / 2
   ymin = yc - h / 2
   xmax = xc + w / 2
   ymax = yc + h / 2
   return tf.concat([xmin, ymin, xmax, ymax], axis=-1)


def xyxy2xywh(boxes):
   """
   Convert bbox format from (xmin, ymin, xmax, ymax) to (xc, yc, w, h).

   Args:
      boxes: A tensor of shape (batch_size, n_boxes, 4).

   Returns:
      A tensor of shape (batch_size, n_boxes, 4) representing the boxes in (xc, yc, w, h) format.
   """
   xmin, ymin, xmax, ymax = tf.split(boxes, 4, axis=-1)
   w = xmax - xmin
   h = ymax - ymin
   xc = xmin + w / 2
   yc = ymin + h / 2
   return tf.concat([xc, yc, w, h], axis=-1)


def get_generators(config):
   train_gen = DataGenerator(config.extracted_train_annotations, 
                           config.train_images, 
                           batches=config.train_batches_per_epoch)
   val_gen = DataGenerator(config.extracted_val_annotations, 
                           config.val_images, 
                           batches=config.val_batches_per_epoch,
                           shuffle=False)
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
         class_id = cocoid_to_coco2017id[int(class_id)]
         uniq_classes.add(class_id)
         if is_crowd:
            continue
         image_id = obj["image_id"]
         image_id = "0"*(12-len(str(image_id))) + str(image_id) + ".jpg"
         bbox = obj["bbox"]
         bbox = [float(x) for x in bbox] # xmin ymin w h
         bbox[0] = bbox[0] + bbox[2]/2 # xmin -> xc
         bbox[1] = bbox[1] + bbox[3]/2 # ymin -> yc

         if image_id not in annotations:
            annotations[image_id] = {"bboxes": [bbox], "classes": [class_id]}
         else:
            annotations[image_id]["bboxes"].append(bbox)
            annotations[image_id]["classes"].append(class_id)

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
      if not os.path.isfile(config.extracted_train_annotations):
         extract_coco(f"{config.train_annotations}", save_filename=config.extracted_train_annotations)
   if extract_val:
      if not os.path.isfile(config.extracted_val_annotations):
         extract_coco(f"{config.val_annotations}", save_filename=config.extracted_val_annotations)


def set_memory_growth():
   gpus = tf.config.list_physical_devices('GPU')
   if len(gpus) == 0:
      raise SystemError
   for gpu in gpus:
      print(gpu) 
      tf.config.experimental.set_memory_growth(gpu, True)



if __name__ == "__main__":
   pass
   #extract_coco("D:/COCO 2017/annotations/instances_train2017.json", save_filename="train_annotations.json")
   #extract_coco("D:/COCO 2017/annotations/instances_val2017.json", save_filename="val_annotations.json")
