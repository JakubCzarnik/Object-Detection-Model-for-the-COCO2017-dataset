from tensorflow.keras.layers import Conv2D, Input, Reshape
from tensorflow.keras.models import Model
from keras_efficientnet_v2 import EfficientNetV2B0


def get_base_model(input_shape):
   base_model = EfficientNetV2B0(input_shape=input_shape)
   
   features = base_model.get_layer("post_swish").output
   base_model.trainable = False

   model = Model(inputs=base_model.input, outputs=features, name="EfficientNetV2B0")
   return model


def build_model(config):
   i = Input((*config.image_size, 3))
   
   base_model = get_base_model(input_shape=(*config.image_size, 3))
   x = base_model(i)

   x = Conv2D(len(config.anchors) * (5+len(config.classes)), (1, 1), activation='sigmoid', kernel_initializer="he_normal", padding="same")(x)
   x = Reshape((config.split_size, config.split_size, len(config.anchors), 5+len(config.classes)))(x)

   model = Model(i, x)
   return model

if __name__ == "__main__":
   from config import Config
   cfg = Config()
   model = build_model(cfg)   
   model.summary()