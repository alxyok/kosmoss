#!/usr/bin/env python
# coding: utf-8

import climetlab
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import randomname

import shutil
import json

@tf.keras.utils.register_keras_serializable()
class HRLayer(tf.keras.layers.Layer):
    """
    Layer to calculate heating rates given fluxes
    and half-level pressures.
    This could be used to deduce the heating rates
    within the model so that the outputs can be 
    constrained by both fluxes and heating rates
    """
    def __init__(self, name=None, **kwargs):
        super(HRLayer, self).__init__(name=name, **kwargs)
        self.g_cp = tf.constant(9.80665 / 1004 * 24 * 3600)
        
    def build(self, input_shape):
        pass

    def call(self, inputs):
        # fluxes = inputs[0] # shape batch,138,1
        hlpress = inputs[1] # shape batch,138,1
        # netflux = fluxes[..., 0] - fluxes[..., 1]
        netflux = inputs[0]
        flux_diff = netflux[..., 1:] - netflux[..., :-1]
        net_press = hlpress[..., 1:, 0] - hlpress[..., :-1, 0]
        return -self.g_cp * tf.math.divide(flux_diff, net_press)

def _parse_fn(x, y):
    new_x = {}
    new_y = {}
    for key in x.keys():
        if key == 'col_inputs':
            indices = [10, 23, 24, 25, 26]
            new_x[key] = tf.gather(x['col_inputs'], indices, axis=2)
        elif key == 'sca_inputs':
            indices = list(range(14)) + [16]
            new_x[key] = tf.gather(x['sca_inputs'], indices, axis=1)
        else:
            new_x[key] = x[key]
    for key in y.keys():
        if "sw" in key:
            print(key)
            new_y[key] = y[key]
            
    new_y["sw_diff"] = new_y["sw"][..., 0] - new_y["sw"][..., 1]
    new_y["sw_add"] = new_y["sw"][..., 0] + new_y["sw"][..., 1]
    
    print(new_y["sw_diff"].shape)
    
    new_y.pop("sw")
    
    return (new_x, new_y)

def densenet_sw(inp_spec, width=138, depth=2, l1=1e-5, l2=1e-4, dropout=False):
    #Assume inputs have the order
    #scalar, column, hl, inter, pressure_hl
    
    widths = tf.repeat(width, depth)
    
    all_inp = []
    for k in inp_spec.keys():
        all_inp.append(tf.keras.Input(inp_spec[k].shape[1:], name=k))
    
    col_inp = tf.keras.layers.Flatten()(all_inp[1])
    hl_inp = tf.keras.layers.Flatten()(all_inp[2])
    inter_inp = tf.keras.layers.Flatten()(all_inp[3])
    dense = tf.keras.layers.Concatenate(axis=-1)([all_inp[0], hl_inp, col_inp, inter_inp])
    # dense = tf.keras.layers.BatchNormalization()(dense)
    
    for i in range(depth):
        dense = tf.keras.layers.Dense(widths[i],
                                      activation=tf.nn.elu,
                                      kernel_initializer='he_uniform',
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                                     )(dense)
        if dropout:
            dense = tf.keras.layers.Dropout(0.5)(dense)
    
    sw_diff = tf.keras.layers.Dense(138, activation='linear', name="sw_diff")(dense)
    sw_add = tf.keras.layers.Dense(138, activation="linear", name="sw_add")(dense)
    
    hr_sw = HRLayer(name="hr_sw")([sw_diff, all_inp[-1]])
    
    return all_inp, [sw_diff, sw_add, hr_sw]

def create_experiment(path, xp_name):
    xp_dir = os.path.join(path, xp_name)

    if os.path.isdir(xp_dir):
        shutil.rmtree(xp_dir)
    else:
        os.makedirs(xp_dir, exist_ok=True)
        
    log_path = os.path.join(xp_dir, 'logs')
    model_path = os.path.join(xp_dir, 'models')
    plot_path = os.path.join(xp_dir, 'plots')

    paths = [log_path, model_path, plot_path]
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print("Directory {} successfully created !".format(path))
        
    return xp_dir, log_path, model_path, plot_path

def main():
    xp_name = randomname.generate('adj/weather', 'adj/colors', 'n/food')
    print("*"*20)
    print("Starting experiment {}".format(xp_name))
    print("*"*20)
    
    strategy = tf.distribute.MirroredStrategy()

    root_path = os.path.join("/", "home", "jupyter", "ECMWF", "radiation")
    data_path = os.path.join(root_path, "data", "tf-dataset", "train")
    xp_path = os.path.join(root_path, "experiments")
    
    # plot_path = os.path.join(xp_path, "plots")
    # model_path = os.path.join(xp_path, "models")
    # log_path = os.path.join(xp_path, "logs")
    
    model_name = "sw-"+xp_name+".h5"

    xp_dir, log_path, model_path, plot_path = create_experiment(xp_path, xp_name)
    
    climetlab.settings.set("cache-directory", data_path)

    ds = climetlab.load_dataset('maelstrom-radiation-tf',
                                dataset = '3dcorrection',
                                timestep = list(range(0, 3501, 1000)), 
                                filenum = list(range(5)),
                                norm=True,
                                hr_units="K d-1",)
    

    # batch_size = 128
    # initial_learning_rate = 1.e-5
    # epochs = 100
    
    kwargs = {
        "epochs" : 50,
        "batch_size" : 256,
        "learning_rate": 0.00005623413251903502,
        "width": 256,
        "depth": 8,
        "l1": 0.000023713737056616554,
        "l2": 0.0001333521432163324,
        "dropout": False,
        "dropout_val": 0.5,
        
    }
    
    with open(os.path.join(xp_dir, "params.json"), "w") as f:
        json.dump(kwargs, f)

    global_batch_size = kwargs["batch_size"] * strategy.num_replicas_in_sync
    tfds = ds.to_tfdataset(batch_size=kwargs["batch_size"], repeat=False)

    tfds_selected = tfds.map(lambda x, y : _parse_fn(x, y)).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    val_size = (271360 // kwargs["batch_size"])
    val_ds = tfds_selected.take(val_size)
    train_ds = tfds_selected.skip(val_size)

    alpha = 0.1
    dk_steps = 5000
    initial_learning_rate = kwargs["learning_rate"]
    
    with strategy.scope():
    # (inputs, outputs) = local_convnet_sw(train_ds.element_spec[0])
        (inputs, outputs) = densenet_sw(train_ds.element_spec[0],
                                        width=kwargs["width"],
                                        depth=kwargs["depth"],
                                        l1=kwargs["l1"],
                                        l2=kwargs["l2"])
        
        losses = {'sw_diff':'mse', 'sw_add':'mse', 'hr_sw':'mse'}
        loss_weights = {"sw_diff": 1, 'sw_add':1, "hr_sw": 1}
        mod = tf.keras.Model(inputs=inputs, outputs=outputs)
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
                                            initial_learning_rate=initial_learning_rate,
                                            decay_steps=dk_steps,
                                            alpha=alpha)
        opt = tf.keras.optimizers.Adam(learning_rate=kwargs["learning_rate"])
        my_metrics = [tf.keras.metrics.MeanAbsoluteError(name='mae')]

        mod.compile(loss=losses, optimizer=opt, loss_weights=loss_weights, metrics=my_metrics)
        
    mod.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, mode="min")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_path,
                                                 histogram_freq=1,
                                                 write_images=False,
                                                 update_freq=100,)

    history = mod.fit(train_ds,
                      validation_data=val_ds,
                      epochs=kwargs["epochs"],
                      verbose=1,
                      callbacks=[tensorboard])

    mod.save(os.path.join(model_path, model_name))

if __name__ == '__main__':
    main()


