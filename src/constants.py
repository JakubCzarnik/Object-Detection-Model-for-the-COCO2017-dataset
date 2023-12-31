all_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
               5: 'bus',6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
               10: 'fire hydrant', 11: 'street sign', 12: 'stop sign', 13: 'parking meter', 14: 'bench', 
               15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 
               20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe,', 
               25: 'hat', 26: 'backpack', 27: 'umbrella', 28: 'shoe', 29: 'eye glasses', 
               30: 'handbag', 31: 'tie', 32: 'suitcase', 33: 'frisbee', 34: 'skis', 
               35: 'snowboard', 36: 'sports ball', 37: 'kite', 38: 'baseball bat', 39: 'baseball glove', 
               40: 'skateboard', 41: 'surfboard', 42: 'tennis racket', 43: 'bottle', 44: 'plate', 
               45: 'wine glass', 46: 'cup', 47: 'fork', 48: 'knife', 49: 'spoon', 
               50: 'bowl', 51: 'banana', 52: 'apple', 53: 'sandwich', 54: 'orange', 
               55: 'broccoli', 56: 'carrot', 57: 'hot dog', 58: 'pizza', 59: 'donut', 
               60: 'cake', 61: 'chair', 62: 'couch', 63: 'potted plant', 64: 'bed', 
               65: 'mirror', 66: 'dining table', 67: 'window', 68: 'desk', 69: 'toilet', 
               70: 'door', 71: 'tv', 72: 'laptop', 73:'mouse', 74: 'remote', 75: 'keyboard', 
               76: 'cell phone', 77: 'microwave', 78: 'oven', 79: 'toaster', 80: 'sink', 
               81: 'refrigerator', 82: 'blender', 83: 'book', 84: 'clock', 85: 'vase', 
               86: 'scissors', 87: 'teddy bear', 88: 'hair drier', 89: 'toothbrush', 90: 'hair brush'}

#from data_loader import extract_coco
#annotations_folder = "D:/COCO 2017/annotations/"
#class_indices_2017 = extract_coco(f"{annotations_folder}instances_train2017.json", save_filename="train_annotations.json")
class_indices_2017 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                     21, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                     41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                     59, 60, 61, 62, 63, 64, 66, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                     80, 81, 83, 84, 85, 86, 87, 88, 89}

# clean classes
available_class = {key:value for key, value in all_classes.items() 
                   if key in class_indices_2017}
available_class = {i: value for i, value in enumerate(available_class.values())}

# map original indices to new
coco_to_ohe_idx = {}
for key1, val1 in all_classes.items():
   for key2, val2 in available_class.items():
      if val1==val2:
         coco_to_ohe_idx[key1] = key2


#from utils import calculate_anchor_boxes
#anchors = calculate_anchor_boxes("train_annotations.json", (384, 384), anchors=6)

# 384x384
anchors = [[ 83.6,  166.31],
         [ 18.95,  26.14],
         [240.03, 137.86],
         [329.84, 316.36],
         [ 63.26,  76.53],
         [140.86, 290.39]]

# 224x224
# anchors = [[ 10.88, 15.01 ],
#            [192.08, 182.53],
#            [136.31, 78.8  ],
#            [ 47.65, 95.69 ],
#            [ 81.62, 168.64],
#            [ 36.1 , 43.72 ]]