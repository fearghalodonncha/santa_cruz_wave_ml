import numpy as np
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, ShuffleSplit
import sys
from datetime import datetime
import pickle
import time
import joblib


'''
Run this code as:
    python mlp_wave_model_scikit.py fit/pred (fit: to generate the model, pred: with the model, predict future data)

    NOTE to configure:
        1. line 45, thisdirectory='' to where all the files stored in
        2. line 46 'slash = 'back' ' to slash = forward/back (pick one if Linux: forward, if windows: back)
        3. line 62 kfold_estimate(#, X_train_minmax, y[:,0:-1], neuron), # indicates how many folds

Files required:
    1. xdt1.txt -> input x, 11078*741
    2. ydt1.txt -> input y, to compare predicted with this, 11078*3105 (3104 hsig + 1 Tper)
    3. Optional - Pest_input.txt -> indicates the input nodes and neuron
        (for example: 10, 20, 30, 10, 0 means 4 layers with 10, 20, 30, 10 nodes in each layer)

This code generates:
    1. python scikit_hsig_save.py fit
        scaler_x.pkl: the scaler information for X as preprocessing, required for prediction mode
        rmse.txt: RMSE for the neuron sets: 1 row: train RMSE, 2 row: test RMSE
    You can comnent certain lines out to not save the following two files especially when run Sensen
        hsig-xplot-pred#.dat: 11078*3104 predicted DL hsig output, # for each kfold
        model#_kfold.pkl # for each kfold, can pick the best
    2. python scikit_hsig_save.py pred
        hsig-future-pred.dat: date to be predicted * 3104
'''



def main(mode):

    if mode == 'fit':
     #   neuron = readin() #read in the neuron info from Pest_input.txt
        neuron = [20, 20, 20]
        X = np.genfromtxt(thisdirectory+str(backforward)+'xdt1.txt', delimiter= "\t")
        y = np.loadtxt(thisdirectory+str(backforward)+'ydt1.txt', delimiter= "\t")[:,0:-1]
        print('Total Data Sets available: \t' + str(len(X)))
        scaler = preprocessing.StandardScaler()
        train = scaler.fit(X)
        joblib.dump(train, thisdirectory+str(backforward)+'scaler_x.pkl')
        X_train_minmax = scaler.fit_transform(X) #preprocessing the data and save the scaler info
        pred_hsig = kfold_estimate(1, X_train_minmax, y, neuron)

    elif mode == 'pred':
        X_pred = np.genfromtxt(thisdirectory+str(backforward)+'xdt1.txt', delimiter= "\t")[0:16,:]
        y_pred = np.loadtxt(thisdirectory+str(backforward)+'ydt1.txt', delimiter= "\t")[0:16,0:-1]
        print(X_pred.shape)
        print(y_pred.shape)
        startTime = datetime.now()
        train_out = joblib.load(thisdirectory+str(backforward)+'scaler_x.pkl')
        X_train_minmax = train_out.transform(X_pred)
        pred_hsig = predmodel(X_train_minmax, y_pred)
        print('Making a model forecast took{}'. format(datetime.now() - startTime))


def readin():
    neuron = []
    with open(thisdirectory+str(backforward)+'Pest_input.txt', 'r') as f:
        filelist1 = f.readlines()
        for line1 in filelist1:
            this_line1 = line1.split(',')
            for i in range(0, len(this_line1)):
                if float(this_line1[i]) != 0:
                    neuron.append(int(float(this_line1[i])))
    print('nodes in each layer \t' + str(neuron))
    return neuron


def kfold_estimate(n_num, X_train_minmax, Y_hsig, neuron):
    cv = ShuffleSplit(n_splits = n_num, random_state=0, test_size=0.1, train_size=0.9)
    train_score = []
    test_score = []
    mlp_hsig = MLPRegressor(solver='adam',  hidden_layer_sizes=neuron, max_iter=5000, batch_size=100, shuffle = True, random_state = 0, activation='relu', early_stopping = False, tol=1e-10 )
    f = open(thisdirectory+str(backforward)+'rmse.txt','w')
    i = 0
    for train_index, test_index in cv.split(X_train_minmax):  #https://stackoverflow.com/questions/41458834/how-is-scikit-learn-cross-val-predict-accuracy-score-calculated
        i += 1
        train_x, test_x = X_train_minmax[train_index], X_train_minmax[test_index]
        train_y, test_y = Y_hsig[train_index], Y_hsig[test_index]
        mlp_hsig.fit(train_x, train_y)
        test_score.append(mlp_hsig.fit(train_x, train_y).score(test_x, test_y))
        train_score.append(mlp_hsig.fit(train_x, train_y).score(train_x, train_y))
        train_rmse = (np.sqrt(metrics.mean_squared_error(train_y, mlp_hsig.predict(train_x))))
        test_rmse = (np.sqrt(metrics.mean_squared_error(test_y, mlp_hsig.predict(test_x))))
        f.write('{:2.5f} \n {:2.5f} \n'.format(train_rmse, test_rmse))

        #if do not wish to save the data, comment the next two lines out
        joblib.dump(mlp_hsig, thisdirectory+str(backforward)+'model'+str(i)+'_kfold_hsig.pkl')
        np.savetxt(thisdirectory+str(backforward)+'hsig-xplot-pred'+str(i)+'.dat', mlp_hsig.predict(X_train_minmax),  fmt = '%2.5f', delimiter = '\t')
    return Y_hsig


def predmodel(X_train_minmax, Y_hsig):
    mlp_hsig = joblib.load(thisdirectory+str(backforward)+'model1_kfold_hsig.pkl')
    pred_hsig = mlp_hsig.predict(X_train_minmax)
    print(mlp_hsig.score(X_train_minmax, Y_hsig))
    print(np.sqrt(metrics.mean_squared_error(Y_hsig, pred_hsig)))
    np.savetxt(thisdirectory+str(backforward)+'hsig-future-pred.dat', pred_hsig,  fmt = '%2.5f', delimiter = '\t')
    return pred_hsig


if __name__ == '__main__':
    #read from X, Y txt
    global thisdirectory, backforward
    thisdirectory = '../BuildDesignMatrices/'
    slash = 'forward'
    if slash == 'back':
        backforward = '\\'
    elif slash == 'forward':
        backforward = '/'
    try:
        mode = sys.argv[1]
    except:
        print('please specify fit/pred in order to make a forecast')
        print('e.g. python mlp_wave_model_scikit.py fit')
        sys.exit(1) # exiing with a non zero value is better for returning from an error
    if (mode == 'fit'):
        print('we are going to train the ML model')
    elif (mode == 'pred'):
        print('we are going to use stored model to make a forecast')
    else:
        print('please specify fit/pred')
    main(mode)
