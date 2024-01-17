from utils import BBoxParser, non_max_suppression
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
import cv2


class MapsCompareCallback(tf.keras.callbacks.Callback):
   def __init__(self, generator, config):
      self.generator = generator
      self.folder = config.output_folder
      self.mean = config.mean
      self.std = config.std
      self.plot_size = config.plot_size
      self.bbox_threshold = config.bbox_threshold
      self.classes = config.classes


   def on_epoch_end(self, epoch, logs=None):
      images = []
      y_true, y_pred = [], [],

      size = self.plot_size[0] * self.plot_size[1]
   
      while len(images) < size:
         idx = np.random.randint(0, len(self.generator), 1)[0]

         _images, _y = self.generator[idx]
         images.extend(_images)
         y_true.extend(_y)

         _y = self.model(_images)
         y_pred.extend(_y)

      bboxes_true = BBoxParser.parse_batch(np.array(y_true)[:size])
      bboxes_pred = BBoxParser.parse_batch(np.array(y_pred)[:size])

      bboxes_nms = non_max_suppression(np.array(y_pred)[:size])
      bboxes_nms = BBoxParser.parse_batch(bboxes_nms)

      # denomralize
      images = ((np.array(images)*self.std + self.mean)*255).astype(np.int32)

      _plots = [["nms", bboxes_nms], ["pred", bboxes_pred], ["true", bboxes_true]]
      for (_name, _bboxes) in _plots:
         _, axs = plt.subplots(*self.plot_size)
         id = 0

         for i in range(self.plot_size[0]):
            for j in range(self.plot_size[1]):
               image = images[id].copy()

               for bbox in _bboxes[id]:
                  v, x, y, w, h, class_id = bbox

                  class_name = self.classes[class_id]

                  cv2.rectangle(image, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255, 0, 0), 2)
                  cv2.putText(image, f"{v:.2f} {class_name}", (int(x-w/2+10), int(y-h/2-10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
               cv2.putText(image, f"Bboxes: {len(_bboxes[id])}", (int(20), int(20)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
               axs[i, j].imshow(image)
               id += 1
         plt.savefig(f"{self.folder}/{epoch}_{_name}_.png", dpi=400)
         plt.close()



class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint, manager):
        super().__init__()
        self.checkpoint = checkpoint
        self.manager = manager

    def on_epoch_end(self, epoch, logs=None):
        self.manager.save()
