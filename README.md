# santa_cruz_wave_ml

This repo contains resources to design and run a machine learning model
that we developed for Santa Cruz

In simple terms our objective was to create a machine learning surrogate model
of the popular SWAN wave model.

The folder `build_design_matrices` provides resources to create the ML
design matrices (X and Y or features or labels) from standard SWAN model inputs
and outputs.

The folder `ml_model_forecast` contains resources to use these design matrices
to train and run a machine learning model

More details regarding the implementation available from our [paper](https://arxiv.org/pdf/1709.08725.pdf)
