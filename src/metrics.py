import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment
from tensorflow.keras.metrics import Metric
from utils import BBoxParser


def dif(value1, value2):
   return tf.reduce_sum(tf.square(tf.subtract(value1, value2)))


class YoloLoss:
   def __init__(self, image_size, split_size, class_weights):
      self.image_size = image_size
      self.split_size = split_size
      self.class_weights = class_weights


   def get_config(self):
      config = {"image_size": self.image_size,
                "split_size": self.split_size,
                "class_weights": self.class_weights}
      return config


   def __call__(self, y_true, y_pred):
      """This function computes the total loss for the YOLO object detection algorithm. 
         It calculates four types of losses: no-object confidence loss, object confidence loss, 
         class loss, and box loss. 
      """
      ### weights ###
      w1 = 2/5  # No-object confidence loss
      w2 = 8/5  # Object confident loss
      w3 = 4/5  # Class loss
      w4 = 6/5  # Box loss (x, y, w, h)

      ### Extract important values ###
      target = y_true[..., 0] 

      y_true_object = tf.gather_nd(y_true, tf.where(target[:]==1))
      y_pred_object = tf.gather_nd(y_pred, tf.where(target[:]==1))
      ### No-object confidence loss ###
      y_pred_noobject = tf.gather_nd(y_pred, tf.where(target[:]==0))
      y_pred_lambdas_noonbj = y_pred_noobject[..., 0] 

      no_obj_loss = dif(tf.zeros_like(y_pred_lambdas_noonbj, tf.float32),
                        tf.cast(y_pred_lambdas_noonbj, tf.float32))
      
      ### Object confidence loss ###
      y_pred_lambdas_obj = y_pred_object[..., 0] 

      obj_loss = dif(tf.ones_like(y_pred_lambdas_obj, tf.float32),
                        tf.cast(y_pred_lambdas_obj, tf.float32))

      ### Class loss ###
      y_true_class = y_true_object[..., 5:]
      y_pred_class = y_pred_object[..., 5:]

      class_loss = tf.square(tf.subtract(tf.cast(y_true_class, tf.float32), tf.cast(y_pred_class, tf.float32)))
      class_loss = tf.reduce_sum(tf.multiply(class_loss, tf.cast(self.class_weights, tf.float32)))
      
      ### Object loss ###
      y_true_pos = y_true_object[..., 1:3]
      y_pred_pos = y_pred_object[..., 1:3]

      center_loss = dif(tf.cast(y_true_pos, tf.float32), 
                        tf.cast(y_pred_pos, tf.float32))
      y_true_pos = y_true_object[..., 3:5]
      y_pred_pos = y_pred_object[..., 3:5]
      
      box_size_loss = dif(tf.cast(y_true_pos, tf.float32),
                     tf.cast(y_pred_pos, tf.float32))
      object_loss = center_loss+box_size_loss
      ### Total loss ###
      total_loss = w1*no_obj_loss + w2*obj_loss + w3*class_loss + w4*object_loss
      return total_loss



class ClassMatchMetric(Metric):
   """This class calculates the classification accuracy for bounding boxes (bboxes) 
      that pass a certain threshold. 
   
      Specifically, it compares the predicted and true labels for bboxes where the 
      predicted value exceeds the threshold (i.e., the bboxes that will be visualized). 
      The accuracy is then calculated as the ratio of correctly classified bboxes 
      to all bboxes that passed the threshold.
      
      The formula used is:
      >>> correctly classified bboxes / all passed bboxes
   """
   def __init__(self, threshold, name='class_acc', **kwargs):
      super().__init__(name=name, **kwargs)
      self.threshold = threshold
      self.trues = self.add_weight(name='trues', initializer='zeros')
      self.denominator = self.add_weight(name='dm', initializer='zeros')


   def update_state(self, y_true, y_pred, sample_weight=None):
      bboxes_true = BBoxParser.extract_anchors(y_true)
      bboxes_pred = BBoxParser.extract_anchors(y_pred)

      target = tf.where(bboxes_pred[..., 0][:] > self.threshold)   
      # take obj cells where y_pred lambda > thrshold
      y_true_class = tf.gather_nd(bboxes_true[..., 5], target)
      y_pred_class = tf.gather_nd(bboxes_pred[..., 5], target)

      # grab TP bboxes and 
      tp = tf.reduce_sum(tf.where(y_true_class==y_pred_class, 1, 0))
      
      bboxes = tf.reduce_sum(tf.where(y_pred_class != -1, 1, 0))

      self.trues.assign_add(tf.cast(tp, tf.float32))
      self.denominator.assign_add(tf.cast(bboxes, tf.float32))


   def result(self):
      if tf.equal(self.denominator, 0):
         return tf.constant(1, dtype=tf.float32)
      return self.trues / self.denominator


   def reset_states(self):
      self.trues.assign(0.)
      self.denominator.assign(0.)



class AnchorMatchMetric(Metric):
   """This class calculates the anchor accuracy for bounding boxes (bboxes) 
      that pass a certain threshold. 
   
      Specifically, it compares the predicted and true labels for bboxes where the 
      predicted value exceeds the threshold (i.e., the bboxes that will be visualized). 
      The accuracy is then calculated as the ratio of correctly choosed anchor 
      to all bboxes that passed the threshold.
      
      The formula used is:
      >>> correctly choosed anchors / all passed bboxes
   """
   def __init__(self, threshold, name='anchor_acc', **kwargs):
      super().__init__(name=name, **kwargs)
      self.threshold = threshold
      self.trues = self.add_weight(name='trues', initializer='zeros')
      self.denominator = self.add_weight(name='dm', initializer='zeros')


   def update_state(self, y_true, y_pred, sample_weight=None):
      bboxes_true = BBoxParser.extract_anchors(y_true)
      bboxes_pred = BBoxParser.extract_anchors(y_pred)

      target = tf.where(bboxes_pred[..., 0][:] > self.threshold)   
      # take obj cells
      y_true_anchor = tf.gather_nd(bboxes_true[..., 6], target)
      y_pred_anchor = tf.gather_nd(bboxes_pred[..., 6], target)
      # grab TP bboxes and 
      tp = tf.reduce_sum(tf.where(y_true_anchor==y_pred_anchor, 1, 0))

      bboxes = tf.reduce_sum(tf.where(y_pred_anchor != 1, 1, 0))

      self.trues.assign_add(tf.cast(tp, tf.float32))
      self.denominator.assign_add(tf.cast(bboxes, tf.float32))


   def result(self):
      if tf.equal(self.trues, 0):
         return tf.constant(1, dtype=tf.float32)
      return self.trues / self.denominator


   def reset_states(self):
      self.trues.assign(0.)
      self.denominator.assign(0.)



class ConfidenceF1Score(Metric):
   """Calculates the F1 score for pred and true bounding boxes.
         >>> 2*precision*reccall/(precision+recall) 
   """
   def __init__(self, threshold, name='lambda_f1', **kwargs):
      super().__init__(name=name, **kwargs)
      self.threshold = threshold
      self.true_positives = self.add_weight(name='tp', initializer='zeros')
      self.false_positives = self.add_weight(name='fp', initializer='zeros')
      self.false_negatives = self.add_weight(name='fn', initializer='zeros')



   def update_state(self, y_true, y_pred, sample_weight=None):
      bboxes_true = BBoxParser.extract_anchors(y_true)
      bboxes_pred = BBoxParser.extract_anchors(y_pred)

      lambda_true = bboxes_true[..., 0] # true
      lambda_pred = bboxes_pred[..., 0] # false

      true_positive = tf.where((tf.cast((lambda_true >= 0.5), tf.int32) + 
                                 tf.cast((lambda_pred >= self.threshold), tf.int32)) == 2,
                                 1, 0)

      false_positive = tf.where((tf.cast((lambda_true < 0.5), tf.int32) + 
                                 tf.cast((lambda_pred > self.threshold), tf.int32)) == 2,
                                 1, 0)

      false_negative = tf.where((tf.cast((lambda_true > 0.5), tf.int32) + 
                                 tf.cast((lambda_pred < self.threshold), tf.int32)) == 2,
                                 1, 0)
      
      tp_count = tf.reduce_sum(true_positive)
      fp_count = tf.reduce_sum(false_positive)
      fn_count = tf.reduce_sum(false_negative)
      self.true_positives.assign_add(tf.cast(tp_count, tf.float32))
      self.false_positives.assign_add(tf.cast(fp_count, tf.float32))
      self.false_negatives.assign_add(tf.cast(fn_count, tf.float32))



   def result(self):
      eps = 1e-8
      precision = self.true_positives/(self.true_positives+self.false_positives+eps)
      recall = self.true_positives/(self.true_positives+self.false_negatives+eps)
      f1 = 2*precision*recall/(precision+recall+eps)

      return f1


   def reset_states(self):
      self.correct_predictions.assign(0.)
      self.total_predictions.assign(0.)


class MeanIou:
   def __init__(self):
      self.ious = 0
      self.count = 0

   def update_state(self, y_true, y_pred):
      """Takes two lists of shapes (B, N, 6) and calculates the IoUs between them.
         Each bounding box is represented as a 6-tuple (v, x, y, w, h, class_id).
      """
      for i in range(len(y_true)):
         # for every image in batch
         true = y_true[i]
         pred = y_pred[i]

         cost_matrix =  np.zeros((len(pred), len(true)))
         for i, pbox in enumerate(pred):
            for j, tbox in enumerate(true):
               # cost = 1 - iou, grab x,y,w,h
               cost_matrix[i, j] = 1 - BBoxParser.calculate_iou(pbox[1:5], tbox[1:5])

         row_ind, col_ind = linear_sum_assignment(cost_matrix)
         for i, j in zip(row_ind, col_ind):
            self.ious += BBoxParser.calculate_iou(pred[i][1:5], true[j][1:5])
         
         self.count += max(len(pred), len(true))

   def get_state(self):
      return self.ious/self.count 