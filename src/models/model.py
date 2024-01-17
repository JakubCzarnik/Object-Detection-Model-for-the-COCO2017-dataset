import tensorflow as tf
from models.layers import *
from tensorflow.keras import Model
from tensorflow.keras import layers


class DetectionModel(tf.keras.Model):
   def __init__(self, config):
      super(DetectionModel, self).__init__()
      self.base_model = build_model(config)


   def compile(self, optimizer, loss, metrics=None, **kwargs):
      super(DetectionModel, self).compile(**kwargs)
      self.optimizer = optimizer
      self.loss = loss
      self.test_metrics = metrics


   def call(self, inputs, training=False):
      return self.base_model(inputs, training=training)


   def train_step(self, data):
      x, y_true = data

      with tf.GradientTape() as tape:
         y_pred = self(x, training=True)
         loss, info = self.loss(y_true, y_pred)

      gradients = tape.gradient(loss, self.base_model.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
      return {'loss': loss, **info}


   def test_step(self, data):
      output_dict = {}
      x, y_true = data

      y_pred = self(x, training=False)
      loss, _ = self.loss(y_true, y_pred)
      output_dict["loss"] = loss

      if self.test_metrics:
         for metric in self.test_metrics:
            metric.update_state(y_true, y_pred)
         output_dict.update({m.name: m.result() for m in self.test_metrics})

      return output_dict


def build_model(config):
   n_classes = config.n_classes
   _in_shape = (*config.target_image_size, 3)
   target_image_size = config.target_image_size
   d, w, r =  config.model_params

   ### Fetures ###
   inputs = layers.Input(_in_shape)
   x = Conv(64*w, kernel=3, strides=2)(inputs)
   x = Conv(128*w, kernel=3, strides=2)(x)
   x = C2F(128*w, shortcut=True, n=int(3*d))(x)

   x = Conv(256*w, kernel=3, strides=2)(x)
   p3 = C2F(256*w, shortcut=True, n=int(6*d))(x)

   x = Conv(512*w, kernel=3, strides=2)(p3)
   p4 = C2F(512*w, shortcut=True, n=int(6*d))(x)

   x = Conv(512*w*r, kernel=3, strides=2)(p4)
   x = C2F(512*w*r, shortcut=True, n=int(3*d))(x)
   p5 = SPPF(512*w*r)(x)

   ### Up ###
   x = layers.UpSampling2D()(p5)
   x = layers.concatenate([x, p4])
   p4 = C2F(512*w, shortcut=False, n=int(3*d))(x) 

   x = layers.UpSampling2D()(p4)
   x = layers.concatenate([x, p3])
   p3 = C2F(256*w, shortcut=False, n=int(3*d))(x)

   ### Down ###
   x = Conv(256*w, kernel=3, strides=2)(p3)   
   x = layers.concatenate([x, p4])
   p4 = C2F(512*w, shortcut=False, n=int(3*d))(x)

   x = Conv(512*w, kernel=3, strides=2)(p4)    
   x = layers.concatenate([x, p5])
   p5 = C2F(512*w*r, shortcut=False, n=int(3*d))(x)

   ### detect ###
   p4 = Detect(n_classes)(p4)
   p5 = Detect(n_classes)(p5)

   ### concat ###
   p4 = DenormalizeBboxes(target_image_size)(p4)
   p5 = DenormalizeBboxes(target_image_size)(p5)

   output = tf.concat([p4, p5], axis=1)
   
   model = Model(inputs=inputs, outputs=output)
   return model




