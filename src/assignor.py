import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from utils import xywh2xyxy

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

      assignments = tf.vectorized_map(lambda x: tf.py_function(func=self.hungarian_method, inp=[x], Tout=tf.float32), alignment_metrics)
      # asign
      y_true_expanded = tf.zeros_like(y_pred)

      batch_indices = tf.range(tf.shape(y_true)[0], dtype=tf.int32)[:, tf.newaxis]
      batch_indices = tf.tile(batch_indices, [1, tf.shape(assignments)[-1]])
      indices = tf.stack([batch_indices, tf.cast(assignments, tf.int32)], axis=-1)

      # asign pred bboxes to true bboxes
      y_true = tf.tensor_scatter_nd_update(y_true_expanded, indices, y_true)
      return y_true


   @staticmethod
   def hungarian_method(score_matrix):
      _, col_indices = linear_sum_assignment(-score_matrix)
      return col_indices


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


