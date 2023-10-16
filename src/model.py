from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.layers import Conv2D, Input, Reshape
from tensorflow.keras.models import Model


def build_model(input_shape, split_size, num_anchors, num_classes):
   base = ResNet101(weights="imagenet", input_shape=input_shape, include_top=False)
   base.trainable = False

   i = Input(input_shape)
   x = base(i)
   x = Conv2D(num_anchors * (5+num_classes), (1, 1), activation='sigmoid')(x)

   x = Reshape((split_size, split_size, num_anchors, 5+num_classes))(x)

   model = Model(i, x)
   return model
