# Training ML model to predict ocean waves

This uses an MLP model to learn the relationship between features and labels
for a wave forecasting problem.
Features include external wave data, wind speeds and ocean current speeds

## Running the model


The model requires input data of features (boundary conditions) and labels (SWAN model predictions).   

The data can be downloaded here:   

[X design matrix](https://ibm.box.com/s/vqco26m3c3rifx7hivlbjdr5kwnuksm8)   
[Y design matrix](https://ibm.box.com/s/bto1i01stq2a6glq39mk5vanm74ftr6f)    

Place the xdt.txt and ydt1.txt files in the ./data folder   

The model can then be run as follows:
 - To train the MLP model we run as    
   .. - `python mlp_wave_model_scikit.py fit`
 - To use the MLP model to make forecast, we run as
   .. - `python mlp_wave_model_scikit.py pred`


Note that in order to make a prediction, first the model must have been trained
using the first `fit` command.



For simplicity a docker file is provided for the tensorflow setup
