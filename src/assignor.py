import tensorflow as tf
from scipy.optimize import linear_sum_assignment


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


def hungarian_method(score_matrix):
   _, col_indices = linear_sum_assignment(-score_matrix)
   return col_indices


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



class DynamicBBoxMatcher:
   def __init__(self, alpha=4, beta=1):
      self.alpha = alpha
      self.beta = beta

 
   def __call__(self, y_true, y_pred):
      """Assigns the predicted bounding boxes to the true bounding boxes.
      
      Args:
         y_true (tf.Tensor): The ground truth bounding boxes and class labels. Shape: (B, max_boxes, 5+n_classes).
         y_pred (tf.Tensor): The predicted bounding boxes and class probabilities. Shape: (B, 2000, 5+n_classes).
      
      Returns:
         tf.Tensor: The assigned ground truth bounding boxes and class labels. Shape: (B, 2000, 5+n_classes).
      """
      alignment_metrics = self.compute_alignment_metric(y_true, y_pred) # -> (B, n_boxes, 2000)

      assignments = tf.map_fn(lambda x: tf.py_function(func=hungarian_method, inp=[x], Tout=tf.float32), alignment_metrics)
      # asign
      y_true_expanded = tf.zeros_like(y_pred)

      batch_indices = tf.range(tf.shape(y_true)[0], dtype=tf.int32)[:, tf.newaxis]
      batch_indices = tf.tile(batch_indices, [1, tf.shape(assignments)[-1]])
      indices = tf.stack([batch_indices, tf.cast(assignments, tf.int32)], axis=-1)

      # asign pred bboxes to true bboxes
      y_true = tf.tensor_scatter_nd_update(y_true_expanded, indices, y_true)
      return y_true


   def compute_alignment_metric(self, y_true, y_pred):
      """Computes the alignment metric between the true and predicted bounding boxes.
      
      Args:
         y_true (tf.Tensor): The ground truth bounding boxes and class labels. Shape: (B, max_boxes, 5+n_classes).
         y_pred (tf.Tensor): The predicted bounding boxes and class probabilities. Shape: (B, 2000, 5+n_classes).
      
      Returns:
         tf.Tensor: The alignment metrics. Shape: (B, max_boxes, 2000).
      """
      xywh_true = y_true[..., 1:5]
      xywh_pred = y_pred[..., 1:5]

      xyxy_true = xywh2xyxy(xywh_true)
      xyxy_pred = xywh2xyxy(xywh_pred)
      ious = self.compute_iou(xyxy_true, xyxy_pred)

      true_class_indices = tf.argmax(y_true[..., 5:], axis=-1) # (B, max_boxes)
      class_probs = tf.gather(y_pred[..., 5:], true_class_indices, axis=-1, batch_dims=1)
      class_probs = tf.transpose(class_probs, perm=[0, 2, 1]) # (B, max_boxes, 2000)

      alignment_metrics = self.alpha * ious + self.beta * class_probs
      return alignment_metrics


   def compute_iou(self, y_true, y_pred):
      """Computes the Intersection over Union (IoU) between the true and predicted bounding boxes.
      
      Args:
         y_true (tf.Tensor): The ground truth bounding boxes. Shape: (B, max_boxes, 4).
         y_pred (tf.Tensor): The predicted bounding boxes. Shape: (B, 2000, 4).
      
      Returns:
         tf.Tensor: The IoU scores. Shape: (B, max_boxes, 2000)
      """
      # (B, max_boxes, 1, 5+n_classes)
      boxes1 = tf.expand_dims(y_true, axis=-2) #xyxy
      # (B, 1, 2000, 5+n_classes)
      boxes2 = tf.expand_dims(y_pred, -3)

      intersection_mins = tf.maximum(boxes1[..., :2], boxes2[..., :2])
      intersection_maxes = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
      intersection_wh = tf.maximum(intersection_maxes - intersection_mins, 0.)
      intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]

      boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
      boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

      union_area = boxes1_area + boxes2_area - intersection_area
      iou_scores = intersection_area / (union_area + 1e-7)
      return iou_scores


