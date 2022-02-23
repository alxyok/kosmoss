import climetlab as cml
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.integration.keras import TuneReportCallback
import tensorflow as tf

from kosmoss import CACHE_DATA_PATH

@tf.keras.utils.register_keras_serializable()
class HRLayer(tf.keras.layers.Layer):
    
    def __init__(self, name=None, **kwargs):
        super(HRLayer, self).__init__(name=name, **kwargs)
        self.g_cp = tf.constant(9.80665 / 1004 * 24 * 3600)
        
    def build(self, input_shape):
        pass

    def call(self, inputs):
        
        hlpress = inputs[1] 
        netflux = inputs[0]
        
        flux_diff = netflux[..., 1:] - netflux[..., :-1]
        net_press = hlpress[..., 1:, 0] - hlpress[..., :-1, 0]
        
        return -self.g_cp * tf.math.divide(flux_diff, net_press)


def create_datasets(config):

    def parse_fn(x, y):
        
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
                new_y[key] = y[key]

        new_y["sw_diff"] = new_y["sw"][..., 0] - new_y["sw"][..., 1]
        new_y["sw_add"] = new_y["sw"][..., 0] + new_y["sw"][..., 1]
        new_y.pop("sw")

        return new_x, new_y
    

    cml.settings.set("cache-directory", CACHE_DATA_PATH)
    cmlds = cml.load_dataset('maelstrom-radiation-tf',
                             dataset = '3dcorrection',
                             timestep = list(range(0, 3501, 1000)), 
                             filenum = list(range(5)),
                             norm=True,
                             hr_units="K d-1",)
    
    tfds = cmlds.to_tfdataset(batch_size=config["batch_size"], repeat=False)
    tfds = tfds.map(parse_fn)
    tfds = tfds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    valsize = 271360 // int(config["batch_size"])
    valds = tfds.take(valsize)
    trainds = tfds.skip(valsize)
    
    return trainds, valds


def create_model(config):

    def build_model():

        # Assuming inputs have the order: scalar, column, hl, inter, pressure_hl
        input_spec = config['input_spec']
        all_inp = [tf.keras.Input(input_spec[k].shape[1:], name=k) 
                   for k in input_spec.keys()]

        col_inp = tf.keras.layers.Flatten()(all_inp[1])
        hl_inp = tf.keras.layers.Flatten()(all_inp[2])
        inter_inp = tf.keras.layers.Flatten()(all_inp[3])
        dense = tf.keras.layers.Concatenate(axis=-1)([all_inp[0], hl_inp, col_inp, inter_inp])

        for _ in range(config["depth"]):

            dense = tf.keras.layers.Dense(
                config["width"],
                activation=config["activation"],
                kernel_initializer='he_uniform',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=config["l1"], l2=config["l2"]),
            )(dense)

            if config["dropout"]:
                dense = tf.keras.layers.Dropout(0.5)(dense)

        sw_diff = tf.keras.layers.Dense(138, activation='linear', name="sw_diff")(dense)
        sw_add = tf.keras.layers.Dense(138, activation="linear", name="sw_add")(dense)

        hr_sw = HRLayer(name="hr_sw")([sw_diff, all_inp[-1]])

        return {
            "inputs": all_inp, 
            "outputs": (sw_diff, sw_add, hr_sw)
        }
    

    o = ['hr_sw', 'sw_diff', 'sw_add']
    losses = {k: 'mse' for k in o}
    loss_weights = {k: 1 for k in o}
    model = tf.keras.Model(**build_model())
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    metrics = [tf.keras.metrics.MeanAbsoluteError(name='mae')]

    model.compile(loss=losses, 
                  optimizer=optimizer, 
                  loss_weights=loss_weights, 
                  metrics=metrics)
    model.summary()
    
    return model

def train_mlp(config, num_epochs):
    
    trainds, valds = create_datasets(config)
    config["input_spec"] = trainds.element_spec[0]
    
    model = create_model(config)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        verbose=0,
        callbacks=[
            TuneReportCallback({
                "loss" : "val_loss",
                "hr_mae" : "val_hr_sw_mae",
            })
        ])


def main():
    
    num_epochs = 50
    config = {
        "batch_size" : 256,
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "width": tune.choice([128, 256, 512]),
        "depth": tune.randint(1, 10),
        "l1": tune.loguniform(1e-5, 1e-2),
        "l2": tune.loguniform(1e-5, 1e-2),
        "dropout": tune.choice([True, False]),
        "activation": "swish"
    }
    
    train_mlp_param = tune.with_parameters(train_mlp, 
                                           num_epochs=num_epochs)
    
    analysis = tune.run(
        train_mlp_param,
        metric="loss",
        mode="min",
        scheduler=AsyncHyperBandScheduler(
            max_t=num_epochs,
            grace_period=10,
            reduction_factor=4
        ),
        search_alg=HEBOSearch(),
        num_samples=10,
        resources_per_trial={
            "cpu": 8, 
            "gpu": 0
        },
        config=config,
    )
    
    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == '__main__':
    
    main()
