* Create a Ray.Tune scheduler

Hyper-parameters (HP) are a set of parameters that will remain fixed throughout the convergence of the estimator. They are not learned during training, they are searched over their hyperspace (HS), one experience at a time.

Because training DL models is usually expansive, we need to build strategies to search the HS efficiently, and to kill unpromising HP sets quickly. To achieve this, we need to expose parameters and orchestrate the entire optimization process. We will use the Ray.Tune framwork, part of the Ray.io framework dedicated to distributed Python applications.