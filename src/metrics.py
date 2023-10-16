import tensorflow as tf
from tensorflow.keras.metrics import Metric
from utils import extract_anchors


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
      w1 = 4/5  # No-object confidence loss
      w2 = 6/5  # Object confident loss
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



class GlobalIou(Metric):
   """Calculates the Intersection over Union (IoU) for predicted and true bounding boxes.

      IoU is calculated as the sum of intersections divided by the sum of unions for all cells.
      If there are no bounding boxes to compare (i.e., union is 0), IoU is set to 1.
      >>> IOU = (sum of intersections) / (sum of unions)
      

      Note: This function calculates intersection and union for each cell independently, 
         which might not be the most accurate metric. An ideal metric would use the Hungarian 
         algorithm to optimally match True and Pred bounding boxes across the entire image.
   """
   def __init__(self, anchors, split_size, threshold, name='global_iou', **kwargs):
      super().__init__(name=name, **kwargs)
      self.anchors = anchors
      self.split_size = split_size 
      self.threshold = threshold

      self.interscetions = self.add_weight(name='is', initializer='zeros')
      self.unions = self.add_weight(name='un', initializer='zeros')


   def update_state(self, y_true, y_pred, sample_weight=None):
      target_size = 224 # size doesn't matter 
      anchors = tf.convert_to_tensor(self.anchors, tf.float32)
      # extract anchors -> (confidence, xcenter, ycenter, width_ratio, height_ratio, class_idx, anchor_idx)
      bboxes_true = extract_anchors(y_true)
      bboxes_pred = extract_anchors(y_pred)
      ### Find TP/FP/FN indices ###
      # get confident score
      lambda_1 = bboxes_true[..., 0]
      lambda_2 = bboxes_pred[..., 0]
      # extract true and false boxes (notice: TN doesn't change anything in the calculations)
      true_positive = tf.where((tf.cast((lambda_1 >= 0.5), tf.int32) + 
                                tf.cast((lambda_2 >= self.threshold), tf.int32)) == 2)
      
      false_positive = tf.where((tf.cast((lambda_1 < 0.5), tf.int32) + 
                                 tf.cast((lambda_2 >= self.threshold), tf.int32)) == 2)
      
      false_negative = tf.where((tf.cast((lambda_1 >= 0.5), tf.int32) + 
                                 tf.cast((lambda_2 < self.threshold), tf.int32)) == 2)

      ### Denormalize x,y,w,h ###
      # gather anchors with highest confidence
      anchor_indices = tf.cast(bboxes_true[..., -1], tf.int32)
      anchors = tf.gather(anchors, anchor_indices)
      # denormalize width, height
      bboxes1_wh = anchors * bboxes_true[..., 2:4]
      bboxes2_wh = anchors * bboxes_pred[..., 2:4]
      # calculate grid scalers
      scaler_y = tf.reshape(tf.repeat(tf.range(self.split_size), self.split_size), (self.split_size, self.split_size))
      scaler_y = tf.cast(scaler_y, tf.float32)
      scaler_x = tf.transpose(scaler_y)
      
      scaler_x = tf.expand_dims(scaler_x, axis=-1)
      scaler_y = tf.expand_dims(scaler_y, axis=-1)
      grid = tf.concat([scaler_x, scaler_y], axis=-1)
      # denormalize x_center, y_center
      bboxes1_xy = (grid + bboxes_true[..., 1:3]) * (target_size/self.split_size)
      bboxes2_xy = (grid + bboxes_pred[..., 1:3]) * (target_size/self.split_size)
      ### Convert format and gather ###
      # (xcenter, ycenter, width, height) -> (xmin, ymin, xmax, ymax)
      bboxes1_min_xy = bboxes1_xy - bboxes1_wh/2
      bboxes1_max_xy = bboxes1_xy + bboxes1_wh/2
      
      bboxes2_min_xy = bboxes2_xy - bboxes2_wh/2
      bboxes2_max_xy = bboxes2_xy + bboxes2_wh/2
   
      bboxes_true = tf.concat([bboxes1_min_xy, bboxes1_max_xy], axis=-1)
      bboxes_pred = tf.concat([bboxes2_min_xy, bboxes2_max_xy], axis=-1)
      # gather TP/FP/FN bbox values  
      bboxes1_tp = tf.gather_nd(bboxes_true, true_positive)
      bboxes2_tp = tf.gather_nd(bboxes_pred, true_positive)
      
      bboxes2_fp = tf.gather_nd(bboxes_pred, false_positive)

      bboxes1_fn = tf.gather_nd(bboxes_true, false_negative)
      ### Calculate IOU ###
      # False positive (intersection = 0)
      areas =  (bboxes2_fp[..., 2] - bboxes2_fp[..., 0]) * (bboxes2_fp[..., 3] - bboxes2_fp[..., 1]) 
      union_fp = tf.reduce_sum(areas)
      # False negative (intersection = 0)
      areas =  (bboxes1_fn[..., 2] - bboxes1_fn[..., 0]) * (bboxes1_fn[..., 3] - bboxes1_fn[..., 1]) 
      union_fn = tf.reduce_sum(areas)
      # True positive
      x1 = tf.maximum(bboxes1_tp[..., 0], bboxes2_tp[..., 0])
      y1 = tf.maximum(bboxes1_tp[..., 1], bboxes2_tp[..., 1])
      x2 = tf.minimum(bboxes1_tp[..., 2], bboxes2_tp[..., 2])
      y2 = tf.minimum(bboxes1_tp[..., 3], bboxes2_tp[..., 3])
      
      intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
      intersection_tp = tf.reduce_sum(intersection)

      bboxes1_area = (bboxes1_tp[..., 2] - bboxes1_tp[..., 0]) * (bboxes1_tp[..., 3] - bboxes1_tp[..., 1])
      bboxes2_area = (bboxes2_tp[..., 2] - bboxes2_tp[..., 0]) * (bboxes2_tp[..., 3] - bboxes2_tp[..., 1])
      union_tp = tf.reduce_sum(bboxes1_area + bboxes2_area - intersection)
      union = union_tp + union_fn + union_fp

      self.interscetions.assign_add(tf.cast(intersection_tp, tf.float32))
      self.unions.assign_add(tf.cast(union, tf.float32))


   def result(self):
      if tf.equal(self.unions, 0):
         return tf.constant(1, dtype=tf.float32)  

      iou = (self.interscetions) / (self.unions + 1e-4)
      return tf.clip_by_value(iou, 0, 1)


   def reset_state(self):
      self.interscetions.assign(0.)
      self.unions.assign(0.)



class MeanIou(Metric):
   """Calculates the Intersection over Union (IoU) for predicted and true bounding boxes.

      IoU is calculated as the mean of IOU for all cells.
      If there are no bounding boxes to compare (i.e., union is 0), IoU is set to 1.
      >>> IOU = mean(IOUS)

      Note: This function calculates intersection and union for each cell independently, 
         which might not be the most accurate metric. An ideal metric would use the Hungarian 
         algorithm to optimally match True and Pred bounding boxes across the entire image.
   """
   def __init__(self, anchors, split_size, threshold, name='mean_iou', **kwargs):
      super().__init__(name=name, **kwargs)
      self.anchors = anchors
      self.split_size = split_size 
      self.threshold = threshold

      self.sum_ious = self.add_weight(name='ious', initializer='zeros')
      self.denominator = self.add_weight(name='dm', initializer='zeros')


   def update_state(self, y_true, y_pred, sample_weight=None):
      target_size = 224 # size doesnt matter, 
      anchors = tf.convert_to_tensor(self.anchors, tf.float32)
      # extract anchors -> (confidence, xcenter, ycenter, width_ratio, height_ratio, class_idx, anchor_idx)
      bboxes_true = extract_anchors(y_true)
      bboxes_pred = extract_anchors(y_pred)
      ### Find TP/FP/FN indices ###
      # get confident score
      lambda_1 = bboxes_true[..., 0]
      lambda_2 = bboxes_pred[..., 0]
      # extract true and false boxes (notice: TN doesn't change anything in the calculations)
      true_positive = tf.where((tf.cast((lambda_1 >= 0.5), tf.int32) + 
                                tf.cast((lambda_2 >= self.threshold), tf.int32)) == 2)
      
      false_positive = tf.where((tf.cast((lambda_1 < 0.5), tf.int32) + 
                                 tf.cast((lambda_2 >= self.threshold), tf.int32)) == 2)
      
      false_negative = tf.where((tf.cast((lambda_1 >= 0.5), tf.int32) + 
                                 tf.cast((lambda_2 < self.threshold), tf.int32)) == 2)
      ### Denormalize x,y,w,h ###
      # gather anchors with highest confidence
      anchor_indices = tf.cast(bboxes_true[..., -1], tf.int32)
      anchors = tf.gather(anchors, anchor_indices)
      # denormalize width, height
      bboxes1_wh = anchors * bboxes_true[..., 2:4]
      bboxes2_wh = anchors * bboxes_pred[..., 2:4]
      # calculate grid scalers
      scaler_y = tf.reshape(tf.repeat(tf.range(self.split_size), self.split_size), (self.split_size, self.split_size))
      scaler_y = tf.cast(scaler_y, tf.float32)
      scaler_x = tf.transpose(scaler_y)
      
      scaler_x = tf.expand_dims(scaler_x, axis=-1)
      scaler_y = tf.expand_dims(scaler_y, axis=-1)
      grid = tf.concat([scaler_x, scaler_y], axis=-1)
      # denormalize x_center, y_center
      bboxes1_xy = (grid + bboxes_true[..., 1:3]) * (target_size/self.split_size)
      bboxes2_xy = (grid + bboxes_pred[..., 1:3]) * (target_size/self.split_size)
      ### Convert format and gather ###
      # (xcenter, ycenter, width, height) -> (xmin, ymin, xmax, ymax)
      bboxes1_min_xy = bboxes1_xy - bboxes1_wh/2
      bboxes1_max_xy = bboxes1_xy + bboxes1_wh/2
      
      bboxes2_min_xy = bboxes2_xy - bboxes2_wh/2
      bboxes2_max_xy = bboxes2_xy + bboxes2_wh/2
   
      bboxes_true = tf.concat([bboxes1_min_xy, bboxes1_max_xy], axis=-1)
      bboxes_pred = tf.concat([bboxes2_min_xy, bboxes2_max_xy], axis=-1)
      # gather TP/FP/FN bbox values  
      bboxes1_tp = tf.gather_nd(bboxes_true, true_positive)
      bboxes2_tp = tf.gather_nd(bboxes_pred, true_positive)
      
      bboxes2_fp = tf.gather_nd(bboxes_pred, false_positive)

      bboxes1_fn = tf.gather_nd(bboxes_true, false_negative)
      ### Calculate IOU ###
      # False positive (intersection = 0)
      areas_fp =  (bboxes2_fp[..., 2] - bboxes2_fp[..., 0]) * (bboxes2_fp[..., 3] - bboxes2_fp[..., 1]) 
      count_fp = tf.cast(tf.size(areas_fp), tf.float32)
      # False negative (intersection = 0)
      areas_fn =  (bboxes1_fn[..., 2] - bboxes1_fn[..., 0]) * (bboxes1_fn[..., 3] - bboxes1_fn[..., 1]) 
      count_fn = tf.cast(tf.size(areas_fn), tf.float32)
      # True positive
      x1 = tf.maximum(bboxes1_tp[..., 0], bboxes2_tp[..., 0])
      y1 = tf.maximum(bboxes1_tp[..., 1], bboxes2_tp[..., 1])
      x2 = tf.minimum(bboxes1_tp[..., 2], bboxes2_tp[..., 2])
      y2 = tf.minimum(bboxes1_tp[..., 3], bboxes2_tp[..., 3])
      
      intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

      bboxes1_area = (bboxes1_tp[..., 2] - bboxes1_tp[..., 0]) * (bboxes1_tp[..., 3] - bboxes1_tp[..., 1])
      bboxes2_area = (bboxes2_tp[..., 2] - bboxes2_tp[..., 0]) * (bboxes2_tp[..., 3] - bboxes2_tp[..., 1])
      union_tp = bboxes1_area + bboxes2_area - intersection

      iou_tp = tf.reduce_sum(intersection / (union_tp + 1e-4))
      count_tp = tf.cast(tf.size(intersection), tf.float32)

      self.sum_ious.assign_add(tf.cast(iou_tp, tf.float32))
      self.denominator.assign_add(tf.cast(count_tp + count_fn + count_fp, tf.float32))


   def result(self):
      if tf.equal(self.denominator, 0):
         return tf.constant(1, dtype=tf.float32)
      mean_iou = self.sum_ious/self.denominator
      return tf.clip_by_value(mean_iou, 0, 1)


   def reset_state(self):
      self.sum_ious.assign(0.)
      self.denominator.assign(0.)



class ClassRecoil(Metric):
   """Calculates the classification recoil for pred and true bounding boxes.
      >>> TP / (TP + FN) 
   """
   def __init__(self, name='class_rec', **kwargs):
      super().__init__(name=name, **kwargs)
      self.trues = self.add_weight(name='trues', initializer='zeros')
      self.denominator = self.add_weight(name='dm', initializer='zeros')


   def update_state(self, y_true, y_pred, sample_weight=None):
      bboxes_true = extract_anchors(y_true)
      bboxes_pred = extract_anchors(y_pred)

      target = bboxes_true[..., 0] 
      # take obj cells
      y_true_class = tf.gather_nd(bboxes_true[..., 5], tf.where(target[:]>0.5))
      y_pred_class = tf.gather_nd(bboxes_pred[..., 5], tf.where(target[:]>0.5))
      # grab TP bboxes and 
      tp = tf.reduce_sum(tf.where(y_true_class==y_pred_class, 1, 0))
      tp_fn = tf.reduce_sum(tf.where(target>0.5, 1, 0))

      self.trues.assign_add(tf.cast(tp, tf.float32))
      self.denominator.assign_add(tf.cast(tp_fn, tf.float32))


   def result(self):
      if tf.equal(self.denominator, 0):
         return tf.constant(1, dtype=tf.float32)
      return self.trues / self.denominator


   def reset_states(self):
      self.trues.assign(0.)
      self.denominator.assign(0.)



class AnchorRecoil(Metric):
   """Calculates the Anchor recoil for pred and true bounding boxes.
      >>> TP / (TP + FN) 
   """
   def __init__(self, name='anchor_rec', **kwargs):
      super().__init__(name=name, **kwargs)
      self.trues = self.add_weight(name='trues', initializer='zeros')
      self.denominator = self.add_weight(name='dm', initializer='zeros')


   def update_state(self, y_true, y_pred, sample_weight=None):
      bboxes_true = extract_anchors(y_true)
      bboxes_pred = extract_anchors(y_pred)

      target = bboxes_true[..., 0] 
      # take obj cells
      y_true_anchor = tf.gather_nd(bboxes_true[..., 6], tf.where(target[:]>0.5))
      y_pred_anchor = tf.gather_nd(bboxes_pred[..., 6], tf.where(target[:]>0.5))
      # grab TP bboxes and 
      tp = tf.reduce_sum(tf.where(y_true_anchor==y_pred_anchor, 1, 0))
      tp_fn = tf.reduce_sum(tf.where(target>0.5, 1, 0))

      self.trues.assign_add(tf.cast(tp, tf.float32))
      self.denominator.assign_add(tf.cast(tp_fn, tf.float32))


   def result(self):
      if tf.equal(self.trues, 0):
         return tf.constant(1, dtype=tf.float32)
      return self.trues / self.denominator


   def reset_states(self):
      self.trues.assign(0.)
      self.denominator.assign(0.)



class ConfidenceAccuracy(Metric):
   """Calculates the confident score accuracy for pred and true bounding boxes.
         >>> (TP + TN) / (TP + TN + FP + FN) 
   """
   def __init__(self, threshold, name='lambda_acc', **kwargs):
      super().__init__(name=name, **kwargs)
      self.threshold = threshold
      self.correct_predictions = self.add_weight(name='cp', initializer='zeros')
      self.total_predictions = self.add_weight(name='tp', initializer='zeros')

   def update_state(self, y_true, y_pred, sample_weight=None):
      bboxes_true = extract_anchors(y_true)
      bboxes_pred = extract_anchors(y_pred)

      lambda_1 = bboxes_true[..., 0]
      lambda_2 = bboxes_pred[..., 0]

      true_positive = tf.where((tf.cast((lambda_1 >= 0.5), tf.int32) + 
                                 tf.cast((lambda_2 >= self.threshold), tf.int32)) == 2,
                                 1, 0)
      true_negative = tf.where((tf.cast((lambda_1 < 0.5), tf.int32) + 
                                 tf.cast((lambda_2 < self.threshold), tf.int32)) == 2,
                                 1, 0)
      
      tp_count = tf.reduce_sum(true_positive)
      tn_count = tf.reduce_sum(true_negative)
      
      all_bboxes = tf.where(lambda_1>=-1., 1, 0)
      all_count = tf.reduce_sum(all_bboxes)

      self.correct_predictions.assign_add(tf.cast(tp_count + tn_count, tf.float32))
      self.total_predictions.assign_add(tf.cast(all_count, tf.float32))


   def result(self):
      return self.correct_predictions / self.total_predictions


   def reset_states(self):
      self.correct_predictions.assign(0.)
      self.total_predictions.assign(0.)