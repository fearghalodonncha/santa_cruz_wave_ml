# Generating design matrices to train ML model

This folder contains sample data and script to create
X and Y design matrices from SWAN input data.
The basic idea is to read from the standard SWAN input
files the corresponding input data to serve as _features_
to the ML model and from the SWAN output files the _label_
data that we wish the machine learning model to learn.
I.e. given SWAN boundary data and corresponding forecast
at each time point, we wish to train the ML model to learn
the mapping.

## Running the script.
Simply run as
python XYBuilder.py which creates as outputs design matrices
xdt1.txt (features) and ydt1.txt (labels).

The design matrices are generated from SWAN standard input files
(in our case for a stationary model), SWAN current boundary data
(from the ROMS [Californa model](http://thredds.cencoos.org/thredds/catalog.html?dataset=CENCOOS_CA_ROMS_DAS)), and Wind data (that we
  downloaded from IBM The Weatuer Company)
