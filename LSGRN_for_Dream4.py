"""
******************************

This code is for Dream4 size100

******************************
"""
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import time
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from numpy import nan as NA
from sklearn.model_selection import train_test_split	
from sklearn.preprocessing import StandardScaler	
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def show_picture(key_x,key_y,idx_picture):
    plt.subplot(211)
    plt.plot( key_x, c='blue',label = 'blue-key_x'+str(idx_picture))
    plt.plot(key_y, c='red',label = 'red-key_y'+'0')
    plt.legend(loc='best')
    plt.show()
def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')  

def normalize(x):
	return (x - np.min(x))/(np.max(x) - np.min(x))
def get_importances_single(TS_data, time_points, time_lag, alpha, input_idx, output_idx, SS_data, param_xgb):
    # time lag used for the finite approximation of the derivative of the target gene expression
    h = 1
    ngenes = TS_data[0].shape[1] #100
    nexp = len(TS_data) 
    nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data]) #210
    ninputs = len(input_idx)  #10
    # Construct learning sample 
    # Time-series data
    input_matrix_time = np.zeros((nsamples_time - h * nexp, ninputs)) 
    output_vect_time = np.zeros(nsamples_time - h * nexp) 
    nsamples_count = 0
    for (i,current_timeseries) in enumerate(TS_data): 
        current_time_points = time_points[i] 
        npoints = current_timeseries.shape[0] 
        time_diff_current = current_time_points[h:] - current_time_points[:npoints-h] 
        current_timeseries_input = current_timeseries[:npoints-h,input_idx] 
        current_timeseries_output = (current_timeseries[h:,output_idx] - current_timeseries[:npoints-h,output_idx]) / time_diff_current + alpha*current_timeseries[:npoints-h,output_idx]
        npoints = current_timeseries_input.shape[0] 
        nsamples_current = current_timeseries_input.shape[0] 
        input_matrix_time[nsamples_count:nsamples_count + nsamples_current,:] = current_timeseries_input
        output_vect_time[nsamples_count:nsamples_count + nsamples_current] = current_timeseries_output
        nsamples_count += nsamples_current
    """ print(input_matrix_time)
    print(input_matrix_time.shape)
    print(output_vect_time)
    print(output_vect_time.shape) """

    # Steady-state data
    if SS_data is not None:
        input_matrix_steady = SS_data[:, input_idx]
        output_vect_steady = SS_data[:, output_idx] * alpha
        # Concatenation
        input_all = np.vstack([input_matrix_steady, input_matrix_time])
        output_all = np.concatenate((output_vect_steady, output_vect_time))
    else:
        input_all = input_matrix_time
        output_all = output_vect_time
    
    input_all[np.isinf(input_all)] = 0
    output_all[np.isinf(output_all)] = 0
    x_train =input_all
    y_train=output_all

    #RF
    treeEstimator1=RandomForestRegressor(n_estimators=500,max_features="auto",max_depth=4,n_jobs=-1,random_state=0)
    treeEstimator1.fit(x_train,y_train)
    RF_feature_importances=treeEstimator1.feature_importances_
    
    #XGB
    treeEstimator = XGBRegressor(**param_xgb)
    treeEstimator.fit(x_train, y_train)
    xgb_feature_importances = treeEstimator.feature_importances_

    fim = np.zeros(ngenes)
    #fim[input_idx] = np.log(RF_feature_importances+1)*np.log(xgb_feature_importances+1)
    fim[input_idx]=RF_feature_importances*xgb_feature_importances
    return fim


def get_importances(TS_data, time_points, time_lag, gene_names, regulators, alpha, SS_data=None, param_xgb={}):
    #time_start = time.time()
    ngenes = TS_data[0].shape[1]
    alphas = [alpha] * ngenes
    # Get the indices of the candidate regulators
    input_idx = [gene_names.index(j) for j in regulators] 
    # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators
    FIM = np.zeros((ngenes, ngenes))
    
    for i in range(ngenes):
        idx=[]
        idx2=[]
        keyword1=('G'+str(i+1))
        for k in range(ngenes):
            keyword2=('G'+str(k+1))
            if keyword1!=keyword2 and cor_index.loc[keyword1,keyword2]>threshold:
                idx.append(k)
                idx2.append(keyword2)
        if idx==[] :
            continue
        else:
            fim = get_importances_single(TS_data, time_points, time_lag, alphas[i], idx, i, SS_data, param_xgb)
            FIM[i, :] = fim
    return FIM

def get_scores(VIM, gold_edges, gene_names, regulators):
    idx = [gene_names.index(j) for j in regulators]
    pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
    pred_edges = pd.DataFrame(pred_edges)
    edges = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
    print(edges)
    print(list(edges['2_y']).count(1))
    edges['2_y'].fillna(0,inplace=True)
    edges['2_x'].fillna(0,inplace=True)
    auroc = roc_auc_score(edges['2_y'], edges['2_x'])
    aupr = average_precision_score(edges['2_y'], edges['2_x'])
    return auroc, aupr




for data_index in range(5):
    print(data_index+1)
    data_index = data_index+1
    SS_data_1 = pd.read_csv("data\Dream4\size100\insilico_size100_{}_knockouts.tsv".format(data_index), sep='\t').values
    SS_data_2 = pd.read_csv("data\Dream4\size100\insilico_size100_{}_knockdowns.tsv".format(data_index), sep='\t').values
    SS_data = np.vstack([SS_data_1, SS_data_2])
    TS_data = pd.read_csv("data\Dream4\size100\insilico_size100_{}_timeseries.tsv".format(data_index), sep='\t').values
    cor_index=pd.read_csv("data\Dream4\size100\cor{}.csv".format(data_index),index_col=0)
    i = np.arange(0, 190, 21)
    j = np.arange(21, 211, 21)
    TS_data = [TS_data[i:j] for (i, j) in zip(i, j)] 
    time_points = [np.arange(0, 1001, 50)] * 10 
    ngenes = TS_data[0].shape[1] 
    gene_names = ['G' + str(i + 1) for i in range(ngenes)]
    regulators = gene_names.copy() 
    gold_edges = pd.read_csv("data\Dream4\size100\insilico_size100_{}_goldstandard.tsv".format(data_index), '\t', header=None)
    overall_scores = []
    scores = []
    args = []

    threshold_all=[0.15,0.177,0.168,0.177,0.173]

    threshold=threshold_all[data_index-1]
    alpha=0.011
    xgb_learning_rate=0.012
    time_lag=0
    xgb_kwargs = dict(learning_rate=xgb_learning_rate, importance_type="weight", n_estimators=500,
                        max_depth=4, objective ='reg:squarederror', n_jobs=-1)
    
    FIM = get_importances(TS_data, time_points,time_lag, gene_names, regulators, 
                alpha, SS_data=SS_data, param_xgb=xgb_kwargs)
    auroc, aupr = get_scores(FIM, gold_edges, gene_names, regulators)
    scores.append((auroc, aupr))
    overall_scores.append((auroc + aupr)/2)
    args.append((xgb_kwargs, alpha, time_lag,threshold))
    
                
    idx = np.argmax(overall_scores)
    print("Best score:", scores[idx][0],scores[idx][1], overall_scores[idx])
    print('threshold:',args[idx][3])
    print('xgb_learning_rate:', args[idx][0]['learning_rate'],'alpha:',args[idx][1])
    #******************************************************************************************
    output_file='Dream4_results.csv'
    print('I am writting!')
    with open(output_file,'a')as file_object: 
            file_object.write(str(data_index))
            file_object.write(",") 
            file_object.write(str(scores[idx][0]))
            file_object.write(",") 
            file_object.write(str(scores[idx][1]))
            file_object.write(",") 
            file_object.write(str(overall_scores[idx]))
            file_object.write(",") 
            file_object.write('alpha:'+str(args[idx][1]))
            file_object.write(",") 
            file_object.write('threshold:'+str(args[idx][3]))
            file_object.write(",") 
            file_object.write('xgb_learning_rate:'+str(args[idx][0]['learning_rate']))
            file_object.write(",") 
            file_object.write(str(args[idx]))
            file_object.write(",") 
            file_object.write("\n") 
    print('Writting finish!')

