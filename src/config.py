from constants import available_class, anchors

class Config:
   def __init__(self):
      self.checkpoint_name = "checkpoints/best"
      self.train_dataset = "D:/COCO 2017/train2017/"
      self.val_dataset = "D:/COCO 2017/val2017/"

      self.train_annotations = "D:/COCO 2017/annotations/instances_train2017.json"
      self.val_annotations = "D:/COCO 2017/annotations/instances_val2017.json"

      self.train_extracted_annotations = "train_annotations.json"
      self.val_extracted_annotations = "val_annotations.json"


      self.image_size = (384, 384)
      self.split_size = 12          # On every image will be generatet grid {split_size}x{split_size},
                              # where every cell has his own label.
      self.batch_size = 12
      self.epochs = 150
      self.lr = 1e-4

      self.threshold = 0.3       # Threshold for bbox confidence to pass (used in visualisation and metrics).
      self.threshold_iou = 0.60  # Threshold for IOU between duplicates, if iou > th then drop bbox with smaler area.

      self.train_batches = 1500    # on every epoch, {num_batches} batches will be randomly selected from train dataset.
      self.val_batches = 100     # on every epoch, {val_batches} batches will be randomly selected from val dataset.

      self.train_base = True       # Unlock base model for training?
      self.load_checkpoint = True


      self.available_class = available_class
      self.anchors = anchors
      self.classes = {i: value for i, value in enumerate(self.available_class.values())}