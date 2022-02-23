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

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.integration.keras import TuneReportCallback

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

def densenet_sw(inp_spec, config):
    #Assume inputs have the order
    #scalar, column, hl, inter, pressure_hl
    
    all_inp = []
    for k in inp_spec.keys():
        all_inp.append(tf.keras.Input(inp_spec[k].shape[1:], name=k))
    
    col_inp = tf.keras.layers.Flatten()(all_inp[1])
    hl_inp = tf.keras.layers.Flatten()(all_inp[2])
    inter_inp = tf.keras.layers.Flatten()(all_inp[3])
    dense = tf.keras.layers.Concatenate(axis=-1)([all_inp[0], hl_inp, col_inp, inter_inp])
    # dense = tf.keras.layers.BatchNormalization()(dense)
    
    for i in range(config["depth"]):
        dense = tf.keras.layers.Dense(config["width"],
                                      activation=tf.nn.swish,
                                      kernel_initializer='he_uniform',
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=config["l1"], l2=config["l2"]),
                                     )(dense)
        if config["dropout"]:
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

def train_mlp(config, data=None, num_epochs=50):
    
    
    climetlab.settings.set("cache-directory", data)

    ds = climetlab.load_dataset('maelstrom-radiation-tf',
                                dataset = '3dcorrection',
                                timestep = list(range(0, 3501, 1000)), 
                                filenum = list(range(5)),
                                norm=True,
                                hr_units="K d-1",)
    
    # with open(os.path.join(xp_dir, "params.json"), "w") as f:
    #     json.dump(config, f)

    tfds = ds.to_tfdataset(batch_size=int(config["batch_size"]), repeat=False)

    tfds_selected = tfds.map(lambda x, y : _parse_fn(x, y)).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    val_size = (271360 // int(config["batch_size"]))
    val_ds = tfds_selected.take(val_size)
    train_ds = tfds_selected.skip(val_size)
    
    # with strategy.scope():
    # (inputs, outputs) = local_convnet_sw(train_ds.element_spec[0])
    (inputs, outputs) = densenet_sw(train_ds.element_spec[0], config)

    losses = {'hr_sw':'mse', 'sw_diff':'mse', 'sw_add':'mse'}
    loss_weights = {"hr_sw": 1, "sw_diff": 1, 'sw_add':1}
    mod = tf.keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    my_metrics = [tf.keras.metrics.MeanAbsoluteError(name='mae')]

    mod.compile(loss=losses, optimizer=opt, loss_weights=loss_weights, metrics=my_metrics)
    mod.summary()

    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_path,
    #                                              histogram_freq=1,
    #                                              write_images=False,
    #                                              update_freq=200,)
    tune_reporter = TuneReportCallback({
        "loss" : "val_loss",
        "hr_mae" : "val_hr_sw_mae",
    })

    mod.fit(train_ds,
            validation_data=val_ds,
            epochs=num_epochs,
            verbose=0,
            callbacks=[tune_reporter])
            # callbacks=[tensorboard, tune_reporter])

    # mod.save(os.path.join(model_path, model_name))

    
def main():
    xp_name = randomname.generate('adj/weather', 'adj/colors', 'n/food')
    print("*"*20)
    print("Starting experiment {}".format(xp_name))
    print("*"*20)

    root_path = os.path.join("/", "home", "jupyter", "ECMWF", "radiation")
    data_path = os.path.join(root_path, "data", "tf-dataset", "train")
    xp_path = os.path.join(root_path, "experiments")
    ray_dir = os.path.join("/", "home", "jupyter", "ray_results")
    
    model_name = "sw-"+xp_name+".h5"

    # xp_dir, log_path, model_path, plot_path = create_experiment(xp_path, xp_name)
    
    num_epochs = 50
    
    sched = AsyncHyperBandScheduler(max_t=num_epochs,
                                    grace_period=10,
                                    reduction_factor=4)
    algo = HEBOSearch()
    
    config = {
        "batch_size" : tune.choice([64, 128, 256]),
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "width": tune.choice([128, 256, 512]),
        "depth": tune.randint(1, 10),
        "l1": tune.loguniform(1e-5, 1e-2),
        "l2": tune.loguniform(1e-5, 1e-2),
        "dropout": tune.choice([True, False])
        
    }
    
    train_mlp_param = tune.with_parameters(train_mlp, data=data_path, num_epochs=num_epochs)
    
    resources_per_trial = {"cpu": 2, "gpu": 1}
    
    analysis = tune.run(
        train_mlp_param,
        name=xp_name,
        metric="loss",
        mode="min",
        scheduler=sched,
        search_alg=algo,
        num_samples=10,
        resources_per_trial=resources_per_trial,
        config=config,
    )
    
    print("Best hyperparameters found were: ", analysis.best_config)
    
    with open(os.path.join(ray_dir, xp_name, "best_config.json"), "w") as f:
        json.dump(analysis.best_config, f)
    
if __name__ == '__main__':
    
    main()


