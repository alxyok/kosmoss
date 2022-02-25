from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.integration.keras import TuneReportCallback

from kosmoss.parallel.trainup_gpu_keras import train_mlp

def main() -> None:
    
    num_epochs = 50
    config = {
        "batch_size" : 256,
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "width": tune.choice([128, 256, 512]),
        "depth": tune.randint(1, 10),
        "l1": tune.loguniform(1e-5, 1e-2),
        "l2": tune.loguniform(1e-5, 1e-2),
        "dropout": tune.choice([True, False]),
        "activation": "swish",
        "callbacks": [
            TuneReportCallback({
                "loss" : "val_loss",
                "hr_mae" : "val_hr_sw_mae",
            })
        ]
    }
    
    train_mlp_param = tune.with_parameters(train_mlp, 
                                           num_epochs=num_epochs)
    
    analysis = tune.run(
        train_mlp_param,
        config=config,
        metric="loss",
        mode="min",
        scheduler=AsyncHyperBandScheduler(
            max_t=num_epochs,
            grace_period=10,
            reduction_factor=4
        ),
        search_alg=HEBOSearch(),
        num_samples=1,
        resources_per_trial={
            "cpu": 8, 
            "gpu": 0
        },
        verbose=0,
    )

if __name__ == '__main__':
    
    main()
