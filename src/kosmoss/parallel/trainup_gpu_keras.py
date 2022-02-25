import climetlab as cml
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.integration.keras import TuneReportCallback
import tensorflow as tf
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Layer, Reshape
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import register_keras_serializable

from kosmoss import CACHED_DATA_PATH, CONFIG

CACHED_DATA_PATH = "/home/jupyter/ECMWF/radiation/data/tf-dataset/train"

@register_keras_serializable()
class HeatingRateLayer(Layer):
    
    gravitational_cst = 9.80665
    specific_heat_cst = 1004
        
    def build(self, input_shape):
        pass

    def call(self, inputs):
        
        hlpress = inputs[1] 
        netflux = inputs[0][:, 0] - inputs[0][:, 1]
        
        flux_diff = netflux[..., 1:] - netflux[..., :-1]
        net_press = hlpress[..., 1:, 0] - hlpress[..., :-1, 0]
        
        gcp = -self.gravitational_cst / self.specific_heat_cst * 24 * 3600
        
        return gcp * tf.math.divide(flux_diff, net_press)


def create_datasets(config):

    timestep = int(CONFIG['timestep'])
    cml.settings.set("cache-directory", CACHED_DATA_PATH)
    cmlds = cml.load_dataset('maelstrom-radiation-tf',
                             dataset='3dcorrection',
                             timestep=list(range(0, 3501, timestep)), 
                             filenum=list(range(5)),
                             norm=True,
                             hr_units="K d-1",)
    
    # All data operations are created in a DAG, and delayed to execution time, for optimization purpuses
    tfds = cmlds.to_tfdataset(batch_size=config["batch_size"], repeat=False)
    
    # Prefetch data while GPU is busy with training, so that CPU is never idle
    tfds = tfds.prefetch(buffer_size=tf.data.AUTOTUNE) 
    
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

        sw = Dense(138 * 2, activation='linear')(dense)
        lw = Dense(138 * 2, activation='linear')(dense)
        
        sw = Reshape((138, -1), name='sw')(sw)
        lw = Reshape((138, -1), name='lw')(lw)

        hr_sw = HeatingRateLayer(name="hr_sw")([sw, inputs[-1]])
        hr_lw = HeatingRateLayer(name="hr_lw")([lw, inputs[-1]])

        return {
            "inputs": inputs, 
            "outputs": (sw, hr_sw, lw, hr_lw)
        }
    
    # This function needs to be executed under a strategy scope to enable multi-GPU acceleration
    def compile_model():

        o = ['sw', 'hr_sw', 'lw', 'hr_lw']
        losses = {k: 'mse' for k in o}
        loss_weights = {k: 1 for k in o}
        
        model = Model(**build_model())
        
        optimizer = Adam(learning_rate=config["learning_rate"])
        metrics = [MeanAbsoluteError(name='mae')]

        model.compile(loss=losses, 
                      optimizer=optimizer, 
                      loss_weights=loss_weights, 
                      metrics=metrics)
        return model
    
    
    if os.getenv("enable_gpus"):
        
        # List all GPUs visible in the system
        gpus = tf.config.list_physical_devices('GPU')

        # Additionally, select a subset of GPUs
        strategy = MirroredStrategy(gpus)

        # This is the only departure from a classic model creation standpoint
        # By including model creation in a distributed strategy, the model DAG will be pushed to whatever acceleration device you passed
        with strategy.scope():
            model = compile_model()
            
    else:
        model = compile_model()
        
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
        "activation": "swish",
        "callbacks": []
    }
    
    train_mlp(config, num_epochs)

if __name__ == '__main__':
    
    os.environ['enable_gpus'] = False
    
    tf.config.run_functions_eagerly(False)
    
    # Let it fail if no GPUs available. We want to scale over HW accelerated devices here
    tf.config.set_soft_device_placement(False)
    
    # Potential for optimization with inter and intra op parallelism threads
    # tf.config.threading.set_inter_op_parallelism_threads(num_threads=16)
    # tf.config.threading.set_intra_op_parallelism_threads(num_threads=16)
    
    # Not really specific to distributed training:
    # @tf.function defers the execution of regular code to the DAG, for optimization purposes. Its usage is encouraged whenever possible
    # See https://www.tensorflow.org/guide/intro_to_graphs and https://www.tensorflow.org/api_docs/python/tf/function for more info
    
    # Also, not currently stable as of tensorflow==2.8.0, MLIR optimization will come to TF
    # See https://www.tensorflow.org/mlir for more info
    # tf.config.experimental.enable_mlir_graph_optimization()
    
    main()
