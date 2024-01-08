from data_loader import DataGenerator, MetaData
from utils import BBoxParser
import numpy as np
from constants import coco2017id_to_name

class Config:
   def __init__(self):
      ##### Paths #####
      self.train_annotations:str = "D:/COCO 2017/annotations/instances_train2017.json"
      self.val_annotations:str = "D:/COCO 2017/annotations/instances_val2017.json"
      
      self.extracted_train_annotations:str = "train_annotations.json"
      self.extracted_val_annotations:str = "val_annotations.json"

      self.train_images:str = "D:/COCO 2017/train2017/"
      self.val_images:str = "D:/COCO 2017/val2017/"

      self.checkpoint_load_path:str = "checkpoints/last" # Model checkpoint path to load
      self.load_checkpoint:bool = False                  # will load a checkpoint from the {checkpoint_load_path}
      self.load_last_checkpoint:bool = True              # will load last saved checkpoint

      ##### General #####
      self.target_image_size:list = [640, 640]  # (h, w)
      
      self.epochs:int = 300                     # Num of epochs till end the training
      self.batch_size:int = 8                   # Num of samples in single batch
      self.train_batches_per_epoch:int = 2000   # On every epoch will be choosen {} batches from whole train dataset
      self.val_batches_per_epoch:int = 15       # On every epoch will be choosen {} batches from whole val dataset
      self.learning_rate:float = 3e-4           # Optimizer learning rate


      ##### MetaData #####
      self.available_class:dict = coco2017id_to_name
      self.classes:dict = {i: value for i, value in 
                           enumerate(self.available_class.values())}
      self.n_classes:int = len(self.classes)

      # Image augmentation 
      self.mean:tuple = (0.485, 0.456, 0.406)
      self.std:tuple = (0.229, 0.224, 0.225)


      ##### Bbox Parser #####
      self.bbox_threshold:float = 0.40


      ##### Model #####
      self.model_version:str = "s"  # depth, width, ratio
      self.model_settings:dict = {"n": [0.33, 0.25, 2.00],
                                  "s": [0.33, 0.50, 2.00],
                                  "m": [0.67, 0.75, 1.50],
                                  "l": [1.00, 1.00, 1.00],
                                  "x": [1.00, 1.25, 1.00]}
      self.model_params:list(float, float, float) = self.model_settings[self.model_version]

      ##### Model #####
      self.w1:float = 2 # obj loss
      self.w2:float = 40 # noobj loss
      self.w3:float = 1 # bbox loss
      self.w4:float = 1 # class loss


      ##### Callbacks #####
      self.output_folder:str = "training_output/"      # Folder to save results from model on validation data.
      self.checkpoint_save_path:str = "checkpoints/"   # Model Checkpoint save path
      self.checkpoints_to_keep:int = 1                 # how much keep latest checkpoints in folder
      self.run_tensorboard:bool = False                  # metrics and visualizations will be stored in tensorboard
      self.plot_size:tuple(int, int) = (2, 2)

      ##### Others #####
      self.target_image_size = np.array(self.target_image_size)
      

      ##### Update Classes #####
      self.update_settings()


   def update_settings(self):
      DataGenerator.update(self)
      BBoxParser.update(self)
      MetaData.update(self)

      