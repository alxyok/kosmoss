The following material only covers optimizations that happen at the application level and does not cover lower-level optimization schemes. It really is a hands-on approach.

1. TensorFlow + Keras (optional)
* Load the `tf.records` in a TFRecordDataset
* Embed the Model creation in a Strategy scope
* Launch a training with a call on `fit`
* `tf.config.threading` inter-op and intra-op flags

2. PyTorch + Lightning
* Create a Dataset for fast data loading and encapsulate in a datamodule
* Create a LightningModule that contain the training logic
* Create an on-the-fly normalization Module
* Create a Trainer to orchestrate the training and launch a multi-GPU training