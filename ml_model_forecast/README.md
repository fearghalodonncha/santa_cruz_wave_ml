# Training ML model to predict ocean waves

This uses an MLP model to learn the relationship between features and labels
for a wave forecasting problem.
Features include external wave data, wind speeds and ocean current speeds

## Running the model


The model requires input data of features (boundary conditions) and labels (SWAN model predictions).   

The data can be downloaded here:   

[X design matrix](https://baylor.box.com/s/uqbu8o7jzn9df2j774peklt4ffimm4nq)   
[Y design matrix](https://baylor.box.com/s/h5a7bhnwwnay5l8g5jorca2oemjjxcuf)    

Place the xdt.txt and ydt1.txt files in the ./data folder   

The model can then be run as above:   
python mlp_wave_model_tf.py (TensorFlow) or   
python mlp_wave_model_scikit.py (Sklearn)

For simplicity a docker file is provided for the tensorflow setup

