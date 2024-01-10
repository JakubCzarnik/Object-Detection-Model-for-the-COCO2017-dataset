from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from assignor import DynamicBBoxMatcher, non_max_suppression
import tensorflow as tf

def iou(y_true, y_pred):
   # xc, yc, w, h -> xmin, ymin, xmax, ymax
   true_x1 = y_true[..., 0] - y_true[..., 2] / 2
   true_y1 = y_true[..., 1] - y_true[..., 3] / 2
   true_x2 = y_true[..., 0] + y_true[..., 2] / 2
   true_y2 = y_true[..., 1] + y_true[..., 3] / 2

   pred_x1 = y_pred[..., 0] - y_pred[..., 2] / 2
   pred_y1 = y_pred[..., 1] - y_pred[..., 3] / 2
   pred_x2 = y_pred[..., 0] + y_pred[..., 2] / 2
   pred_y2 = y_pred[..., 1] + y_pred[..., 3] / 2

   intersect_x1 = tf.maximum(true_x1, pred_x1)
   intersect_y1 = tf.maximum(true_y1, pred_y1)
   intersect_x2 = tf.minimum(true_x2, pred_x2)
   intersect_y2 = tf.minimum(true_y2, pred_y2)
   intersect_area = tf.maximum(intersect_x2 - intersect_x1, 0) * \
                  tf.maximum(intersect_y2 - intersect_y1, 0)

   true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
   pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

   union_area = true_area + pred_area - intersect_area
   iou = intersect_area / (union_area + 1e-4)
   return iou


class VarifocalLoss(tf.keras.losses.Loss):
   def __init__(self, alpha=0.75, gamma=2.0, name='varifocal_loss'):
      super().__init__(name=name)
      self.alpha = alpha
      self.gamma = gamma


   def call(self, y_true, y_pred):
      pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)

      focal_weight = tf.cast((y_true > 0.5), tf.float32) + self.alpha * tf.abs(y_pred - pt)**self.gamma * tf.cast(y_true <= 0.5, tf.float32)

      bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
      loss = bce * focal_weight
      return tf.reduce_mean(loss)


class BboxLoss(tf.keras.losses.Loss):
   def __init__(self, name='bbox_loss'):
      super().__init__(name=name)


   def call(self, y_true, y_pred):
      ious = iou(y_true, y_pred)
      loss = tf.reduce_mean(1 - ious)

      return loss


class DetectionLoss(tf.keras.losses.Loss):
   def __init__(self, config, name="detection_loss", **kwargs):
      super().__init__(name=name, **kwargs)
      self.w1 = tf.cast(config.w1, tf.float32)
      self.w2 = tf.cast(config.w2, tf.float32)
      self.w3 = tf.cast(config.w3, tf.float32)
      self.w4 = tf.cast(config.w4, tf.float32)

      self.box_loss = BboxLoss()
      self.tasker = DynamicBBoxMatcher()


   def __call__(self, y_true, y_pred, sample_weight=None):
      y_true = self.tasker(y_true, y_pred)

      ##### Extract important values ##### 
      y_true_object = tf.gather_nd(y_true, tf.where(y_true[..., 0]>0.5))
      y_pred_object = tf.gather_nd(y_pred, tf.where(y_true[..., 0]>0.5))

      y_pred_noobject = tf.gather_nd(y_pred, tf.where(y_true[..., 0]<0.5))

      ##### 1. object confidence loss ##### 
      obj_loss = tf.reduce_mean(binary_crossentropy(tf.ones_like(y_pred_object[..., 0:1]), y_pred_object[..., 0:1]))
      noobj_loss = tf.reduce_mean(binary_crossentropy(tf.zeros_like(y_pred_noobject[..., 0:1]), y_pred_noobject[..., 0:1]))


      ##### 2. Bbox loss #####
      y_true_xywh = y_true_object[..., 1:5]
      y_pred_xywh = y_pred_object[..., 1:5]

      bbox_loss = self.box_loss(y_true_xywh, y_pred_xywh)


      ##### 3. Class loss #####
      y_true_object_class = y_true_object[..., 5:]
      y_pred_object_class = y_pred_object[..., 5:]

      class_loss = tf.reduce_mean(categorical_crossentropy(y_true_object_class, y_pred_object_class))
      
      ##### Total loss #####
      total_loss = self.w1*obj_loss + self.w2*noobj_loss + self.w3*bbox_loss + self.w4*class_loss
      return total_loss, {"obj": self.w1*obj_loss, "noobj": self.w2*noobj_loss, "box": self.w3*bbox_loss, "class": self.w4*class_loss}

### Metrics ###

class IouMetric(tf.keras.metrics.Metric):
   def __init__(self, name='iou', **kwargs):
      super(IouMetric, self).__init__(name=name, **kwargs)
      self.total_iou = self.add_weight(name='iou', initializer='zeros')
      self.num_boxes = self.add_weight(name='num_boxes', initializer='zeros')

      self.tasker = DynamicBBoxMatcher()


   def update_state(self, y_true, y_pred, sample_weight=None):
      y_pred = non_max_suppression(y_pred)
      y_true = self.tasker(y_true, y_pred)
      

      ###
      mask_tp = tf.logical_and(y_true[..., 0] > 0.5, y_pred[..., 0] > 0.5)
      y_true_tp = tf.gather_nd(y_true, tf.where(mask_tp))
      y_pred_tp = tf.gather_nd(y_pred, tf.where(mask_tp))

      mask_fn = tf.logical_and(y_true[..., 0] > 0.5, y_pred[..., 0] < 0.5)
      mask_fp = tf.logical_and(y_true[..., 0] < 0.5, y_pred[..., 0] > 0.5)

      y_true_xywh = y_true_tp[..., 1:5] 
      y_pred_xywh = y_pred_tp[..., 1:5] 

      total_iou = tf.reduce_sum(iou(y_true_xywh, y_pred_xywh))
      tp_bboxes = tf.reduce_sum(tf.cast(mask_tp, tf.float32))
      fn_bboxes = tf.reduce_sum(tf.cast(mask_fn, tf.float32))
      fp_bboxes = tf.reduce_sum(tf.cast(mask_fp, tf.float32))
      total_bboxes = tp_bboxes + fn_bboxes + fp_bboxes

      self.total_iou.assign_add(total_iou)
      self.num_boxes.assign_add(total_bboxes)


   def result(self):
      return self.total_iou / self.num_boxes


   def reset_state(self):
      self.total_iou.assign(0.)
      self.num_boxes.assign(0.)


class F1Score(tf.keras.metrics.Metric):
   """Calculates the F1-Score for object detection where:
      True Positive is when:
         >>>  v_t == v_p == 1 && IoU >= th && cls_t == cls_p
      False Positive is when:
         >>>  v_t < 0.5  && v_p > 0.5
      False Negative is when:
         >>> (v_t > 0.5  && v_p < 0.5) || (v_t == v_p == 1 && (IoU < th || cls_t != cls_p))
      
      where:
         >>> IoU - Intersection over Union between true and pred bbox
         >>> th - Threshold
         >>> v_t, v_p - "Confidence" that there is an (true, pred) bbox
         >>> cls_t, cls_p - Class of given Bbox
         >>> ||, && - Logical "or" and "and"
   """
   def __init__(self, threshold=0.6, name='f1_score', **kwargs):
      super(F1Score, self).__init__(name=name, **kwargs)
      self.threshold = threshold
      self.tp = self.add_weight(name='tp', initializer='zeros')
      self.fp = self.add_weight(name='fp', initializer='zeros')  
      self.fn = self.add_weight(name='fn', initializer='zeros')

      self.tasker = DynamicBBoxMatcher()


   def update_state(self, y_true, y_pred, sample_weight=None):
      y_pred = non_max_suppression(y_pred)
      y_true = self.tasker(y_true, y_pred)
      ###
      mask = tf.logical_or(y_true[..., 0] > 0.5, y_pred[..., 0] > 0.5)
      y_true_items = tf.gather_nd(y_true, tf.where(mask))
      y_pred_items = tf.gather_nd(y_pred, tf.where(mask))
      y_true_class_ids = tf.argmax(y_true_items[..., 5:], axis=-1)
      y_pred_class_ids = tf.argmax(y_pred_items[..., 5:], axis=-1)
      
      y_true_xywh = y_true_items[..., 1:5] 
      y_pred_xywh = y_pred_items[..., 1:5] 

      ious = iou(y_true_xywh, y_pred_xywh)

      # true pos
      tp_mask = tf.logical_and(y_true_items[..., 0] > 0.5, y_pred_items[..., 0] > 0.5)
      tp_mask = tf.logical_and(tp_mask, ious >= self.threshold)

      # false pos
      fp_mask = tf.logical_and(y_true_items[..., 0] < 0.5, y_pred_items[..., 0] > 0.5)

      # false neg
      fn_mask_1 = tf.logical_and(y_true_items[..., 0] > 0.5, y_pred_items[..., 0] < 0.5)
      fn_mask_2 = tf.logical_and(y_true_items[..., 0] == 1, y_pred_items[..., 0] == 1)
      fn_mask_2 = tf.logical_and(fn_mask_2, tf.logical_or(ious < self.threshold, y_true_class_ids != y_pred_class_ids))
      fn_mask = tf.logical_or(fn_mask_1, fn_mask_2)

      self.tp.assign_add(tf.reduce_sum(tf.cast(tp_mask, tf.float32)))
      self.fp.assign_add(tf.reduce_sum(tf.cast(fp_mask, tf.float32)))
      self.fn.assign_add(tf.reduce_sum(tf.cast(fn_mask, tf.float32)))


   def result(self):
      precision = self.tp / (self.tp + self.fp + 1e-8)
      recall = self.tp / (self.tp + self.fn + 1e-8)
      return 2 * ((precision * recall) / (precision + recall + 1e-8))


   def reset_state(self):
      self.tp.assign(0.)
      self.fp.assign(0.)
      self.fn.assign(0.)

