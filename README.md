# Transformer-for-time-series-forecasting-
This code is a realisations of the transformer model from 
Wu, N., Green, B., Ben, X., & O'Banion, S. (2020). Deep transformer models for time series forecasting: The influenza prevalence case. arXiv preprint arXiv:2001.08317.

The ILI data we use is from https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html 

The model is a standard transformer modified to take in time series data where a fully connected layer is added before the input of the endocer.

A greedy decoding method is created when doing long series predictions, as the output of the previous decoder calculation(t-1) will become the input for the calculation for t. 


There are 6 `.py` file in this folder,
utils.py and Network.py files contain descriptions about the model
data_clean.py is used to break down the ILI data and transform it to the correct input type for the transformer.
training.py is used for training
test_validation.py contains functions that we use to validate the model
Testing.py is used for testing the model accuracy. 


The three files we need to run in order:
1. data_clean.py
2. training.py
3. Testing.py
