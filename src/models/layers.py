import tensorflow as tf
from tensorflow.keras import layers


class Conv(layers.Layer):
   def __init__(self, filters, kernel, strides=1, bn=True, activation="swish", padding="same", name=None):
      super(Conv, self).__init__(name=name)
      self.conv = layers.Conv2D(filters, kernel, strides=strides, padding=padding, use_bias= not bn)
      self.bn = layers.BatchNormalization() if bn else None
      self.act = layers.Activation(activation)


   def call(self, inputs):
      x = self.conv(inputs)
      if self.bn:
         x = self.bn(x)
      x = self.act(x)
      return x
    

class SPPF(layers.Layer):
   def __init__(self, filters, k=5):
      super(SPPF, self).__init__()
      self.filters = filters
      self.k = k


   def build(self, input_shape):
      ch = input_shape[-1]
      hidden_ch = ch // 2

      self.conv1 = Conv(hidden_ch, kernel=1, strides=1)
      self.conv2 = Conv(ch, kernel=1, strides=1)
      self.m = layers.MaxPool2D(pool_size=self.k, strides=1, padding='same')


   def call(self, inputs):
      x = self.conv1(inputs)
      y1 = self.m(x)
      y2 = self.m(y1)
      y3 = self.m(y2)
      x = tf.concat([x, y1, y2, y3], axis=-1)
      x = self.conv2(x)
      return x


class Bottleneck(layers.Layer):
   def __init__(self, shortcut=False):
      super(Bottleneck, self).__init__()
      self.shortcut = shortcut


   def build(self, input_shape):
      ch = input_shape[-1]
      self.conv1 = Conv(ch, kernel=3, strides=1)
      self.conv2 = Conv(ch, kernel=3, strides=1)
      self.add = layers.Add() if self.shortcut else None


   def call(self, inputs):
      x = self.conv1(inputs)
      x = self.conv2(x)

      if self.shortcut:
         x = self.add([inputs, x])
      return x


class C2F(layers.Layer):
   def __init__(self, filters, shortcut=False, n=2):
      super(C2F, self).__init__()
      self.shortcut = shortcut

      self.conv1 = Conv(filters, kernel=1, strides=1)
      self.split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=3))
      self.bottlenecks = [Bottleneck(self.shortcut) for _ in range(n)]
      self.concat = layers.Concatenate()
      self.conv2 = Conv(filters, kernel=1, strides=1)


   def call(self, inputs):
      x = self.conv1(inputs)
      _, x2 = self.split(x)
      outputs = []

      for bottleneck in self.bottlenecks:
         x2 = bottleneck(x2)
         outputs.append(x2)

      x = self.concat([x, *outputs])
      x = self.conv2(x)
      return x


class Detect(layers.Layer):
   def __init__(self, n_classes):
      super(Detect, self).__init__()
      self.n_classes = n_classes


   def build(self, input_shape):
      filters = input_shape[-1]
      c1, c2, c3 = int(filters*1/4), int(filters*1/4), int(filters*1/2)
      # v: 1
      self.conv1 = Conv(c1, kernel=3, strides=1)
      self.conv2 = Conv(c1, kernel=3, strides=1)
      self.conv3 = Conv(1, 1, bn=False, activation="sigmoid")
      # xywh: 4
      self.conv4 = Conv(c2, kernel=3, strides=1)
      self.conv5 = Conv(c2, kernel=3, strides=1)
      self.conv6 = Conv(4, 1, bn=False, activation="sigmoid")
      # classification: n_classes
      self.conv7 = Conv(c3, kernel=3, strides=1)
      self.conv8 = Conv(c3, kernel=3, strides=1)
      self.conv9 = Conv(self.n_classes, 1, bn=False, activation="softmax")

      self.concat = layers.Concatenate(axis=-1)


   def call(self, inputs):
      x1 = self.conv1(inputs)
      x1 = self.conv2(x1)
      bbox_v = self.conv3(x1)

      x2 = self.conv4(inputs)
      x2 = self.conv5(x2)
      bbox_xywh = self.conv6(x2)

      x3 = self.conv7(inputs)
      x3 = self.conv8(x3)
      class_ohe = self.conv9(x3)

      x = self.concat([bbox_v, bbox_xywh, class_ohe])
      return x


class DenormalizeBboxes(layers.Layer):
   def __init__(self, target_image_size):
      super(DenormalizeBboxes, self).__init__()
      self.target_image_size = tf.cast(target_image_size, tf.float32)


   def build(self, input_shape):
      _, self.G, _, self.V = input_shape

      scaler_x = tf.range(self.G)
      scaler_x = tf.tile(tf.expand_dims(scaler_x, 0), [self.G, 1])
      scaler_y = tf.range(self.G)
      scaler_y = tf.tile(tf.expand_dims(scaler_y, 1), [1, self.G])
      
      scaler_x = tf.expand_dims(scaler_x, axis=-1)
      scaler_y = tf.expand_dims(scaler_y, axis=-1)
      
      self.scaler = tf.cast(tf.concat([scaler_x, scaler_y], axis=-1), tf.float32)


   def call(self, inputs):
      v = inputs[..., 0:1]
      xy = (inputs[..., 1:3] + self.scaler) / self.G * self.target_image_size
      wh = inputs[..., 3:5] * self.target_image_size[::-1]
      clss = inputs[..., 5:]

      bbox = tf.concat([v, xy, wh, clss], axis=-1)
      return tf.reshape(bbox, (-1, self.G*self.G, self.V))