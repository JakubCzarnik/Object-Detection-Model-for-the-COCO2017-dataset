from constants import coco2017id_to_name
from data_loader import DataGenerator, MetaData
from utils import BBoxParser
import numpy as np

class Config:
   def __init__(self):
      ##### Paths #####
      self.train_annotations:str = "D:/COCO 2017/annotations/instances_train2017.json" # Path to the training annotations
      self.val_annotations:str = "D:/COCO 2017/annotations/instances_val2017.json" # Path to the validation annotations
      
      self.extracted_train_annotations:str = "train_annotations.json" # Path to the extracted training annotations
      self.extracted_val_annotations:str = "val_annotations.json" # Path to the extracted validation annotations

      self.train_images:str = "D:/COCO 2017/train2017/" # Path to the training images
      self.val_images:str = "D:/COCO 2017/val2017/" # Path to the validation images

      self.checkpoint_load_path:str = "checkpoints/last" # Path to load model checkpoint
      self.load_checkpoint:bool = False # If True, will load a checkpoint from the specified path
      self.load_last_checkpoint:bool = True # If True, will load the last saved checkpoint

      ##### General #####
      self.target_image_size:list = np.array([640, 640]) # Target image size (model input size) (height, width)
      
      self.epochs:int = 300 # Number of epochs to train
      self.batch_size:int = 8 # Number of samples in a single batch
      self.train_batches_per_epoch:int = 2000 # Number of batches to choose from the whole training dataset per epoch
      self.val_batches_per_epoch:int = 15 # Number of batches to choose from the whole validation dataset per epoch
      self.learning_rate:float = 3e-4 # Learning rate for the optimizer

      # Image augmentation 
      self.mean:tuple = (0.485, 0.456, 0.406) # Mean for normalization
      self.std:tuple = (0.229, 0.224, 0.225) # Standard deviation for normalization

      ##### Bbox Parser #####
      self.bbox_threshold:float = 0.40 # Threshold for bounding box parser

      ##### Model #####
      self.model_version:str = "s" # Model version ("n", "s", "m", "l", "x")
      #                               (depth, width, ratio)
      self.model_settings:dict = {"n": [0.33,  0.25, 2.00],
                                  "s": [0.33,  0.50, 2.00],
                                  "m": [0.67,  0.75, 1.50],
                                  "l": [1.00,  1.00, 1.00],
                                  "x": [1.00,  1.25, 1.00]}
      self.model_params:list(float, float, float) = self.model_settings[self.model_version] # Model parameters

      ##### Loss Weights #####
      self.w1:float = 2 # Weight for object loss
      self.w2:float = 40 # Weight for no-object loss
      self.w3:float = 1 # Weight for bounding box loss
      self.w4:float = 1 # Weight for class loss

      ##### Callbacks #####
      self.output_folder:str = "training_output/" # Folder to save results from model on validation data
      self.checkpoint_save_path:str = "checkpoints/" # Path to save model checkpoints
      self.checkpoints_to_keep:int = 1 # Number of latest checkpoints to keep in the folder
      self.run_tensorboard:bool = False # If True, metrics and visualizations will be stored in TensorBoard
      self.plot_size:tuple(int, int) = (2, 2) # Size of the plot

      ##### Others #####
      self.available_class:dict = coco2017id_to_name 
      self.classes:dict = {i: value for i, value in 
                           enumerate(self.available_class.values())} 
      self.n_classes:int = len(self.classes) 

      ##### Update Classes #####
      self.update_settings() # Update settings


   def update_settings(self):
      DataGenerator.update(self) # Update DataGenerator settings
      BBoxParser.update(self) # Update BBoxParser settings
      MetaData.update(self) # Update MetaData settings