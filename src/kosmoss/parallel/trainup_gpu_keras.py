import climetlab as cml
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.integration.keras import TuneReportCallback
import tensorflow as tf
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Layer
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import register_keras_serializable

from kosmoss import CACHE_DATA_PATH, CONFIG

@register_keras_serializable()
class HeatingRateLayer(Layer):
    
    gravitational_cst = 9.80665
    specific_heat_cst = 1004
        
    def build(self, input_shape):
        pass

    def call(self, inputs):
        
        hlpress = inputs[1] 
        netflux = inputs[0]
        
        flux_diff = netflux[..., 1:] - netflux[..., :-1]
        net_press = hlpress[..., 1:, 0] - hlpress[..., :-1, 0]
        
        gcp = -self.gravitational_cst / self.specific_heat_cst * 24 * 3600
        
        return gcp * tf.math.divide(flux_diff, net_press)


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
    

    timestep = int(CONFIG['timestep'])
    cml.settings.set("cache-directory", CACHE_DATA_PATH)
    cmlds = cml.load_dataset('maelstrom-radiation-tf',
                             dataset='3dcorrection',
                             timestep=list(range(0, 3501, timestep)), 
                             filenum=list(range(5)),
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
        inputs = [Input(input_spec[k].shape[1:], name=k) 
                   for k in input_spec.keys()]

        dense = Concatenate(axis=-1)([Flatten()(in_) for in_ in inputs])

        for _ in range(config["depth"]):

            dense = Dense(
                config["width"],
                activation=config["activation"],
                kernel_initializer='he_uniform',
                kernel_regularizer=l1_l2(l1=config["l1"], l2=config["l2"]),
            )(dense)

            if config["dropout"]:
                dense = Dropout(0.5)(dense)

        sw_diff = Dense(138, activation='linear', name="sw_diff")(dense)
        sw_add = Dense(138, activation="linear", name="sw_add")(dense)

        hr_sw = HRLayer(name="hr_sw")([sw_diff, inputs[-1]])

        return {
            "inputs": inputs, 
            "outputs": (sw_diff, sw_add, hr_sw)
        }
    

    o = ['hr_sw', 'sw_diff', 'sw_add']
    losses = {k: 'mse' for k in o}
    loss_weights = {k: 1 for k in o}
    
    # List all GPUs visible in the system
    gpus = tf.config.list_physical_devices('GPU')
    
    # Additionally, select a subset of GPUs
    strategy = MirroredStrategy(gpus)
    with strategy.scope():
        
        model = Model(**build_model())
        
        optimizer = Adam(learning_rate=config["learning_rate"])
        metrics = [MeanAbsoluteError(name='mae')]

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
        trainds,
        validation_data=valds,
        epochs=num_epochs,
        verbose=0,
        callbacks=config["callbacks"])


def main():
    
    num_epochs = 50
    config = {
        "batch_size" : 256,
        "learning_rate": 1e-4,
        "width": 128,
        "depth": 4,
        "l1": 1e-3,
        "l2": 1e-3,
        "dropout": True,
        "activation": "swish"
        "callbacks": []
    }
    
    train_mlp(config, num_epochs)

if __name__ == '__main__':
    
    main()
