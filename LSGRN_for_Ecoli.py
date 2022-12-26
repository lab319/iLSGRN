"""
******************************

This code is for Ecoli

******************************
"""

from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler	
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
import os

def get_importances(expr_data, gene_names, regulators, param={}):
    time_start = time.time()
    ngenes = expr_data.shape[1]
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    VIM_XGB = np.zeros((ngenes,ngenes))
    VIM_RF = np.zeros((ngenes,ngenes))

    test_file="RF_MIC_data\\"+symbol_word+"\\RF_MIC_{}.CSV".format(symbol_word+'_'+str(threshold))
    if (os.path.isfile(test_file)):
        print("RF_MIC_data exist")
        VIM_RF = np.loadtxt(test_file,  delimiter=',')
    else:
        with open(test_file,'a')as file_object:print("File created successfully")
        if (os.path.isfile(test_file)):
            for i in range(ngenes):
                input_idx = idx.copy()
                if i in input_idx:
                    input_idx.remove(i)   
                vi = get_importances_single_RF(expr_data,i,input_idx,gene_names)
                VIM_RF[i,:] = vi
            np.savetxt(test_file, VIM_RF,  delimiter=',')
            VIM_RF = np.loadtxt(test_file,  delimiter=',')
        else:
            print("RF_ MIC file does not exist, file creation failed")
    for i in range(ngenes):
        input_idx = idx.copy()
        if i in input_idx:
            input_idx.remove(i)   
        vi = get_importances_single_XGB(expr_data,i,input_idx, param,gene_names)
        VIM_XGB[i,:] = vi
    
    VIM=VIM_XGB*VIM_RF
    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return VIM
    


def get_importances_single_XGB(expr_data,output_idx,input_idx, param,gene_names):
    ngenes = expr_data.shape[1]
    # Expression of target gene
    output = expr_data[:,output_idx]
    # Normalize output data
    output = output / np.std(output)
    #output=output*alpha
    result_index=[]
    for k in input_idx:
        if cor_index.loc[str(gene_names[output_idx]),str(gene_names[k])] > threshold :
            result_index.append(k)
    if len(result_index)<1:
        vi = np.zeros(ngenes)
        return vi
    expr_data_input = expr_data[:,result_index]
    treeEstimator = XGBRegressor(**param)
    # Learn ensemble of trees
    treeEstimator.fit(expr_data_input,output)
    # Compute importance scores
    XGB_feature_importances = treeEstimator.feature_importances_
    vi = np.zeros(ngenes)
    vi[result_index] = XGB_feature_importances
    return vi

def get_importances_single_RF(expr_data,output_idx,input_idx,gene_names):
    ngenes = expr_data.shape[1]
    # Expression of target gene
    output = expr_data[:,output_idx]
    # Normalize output data
    output = output / np.std(output) 
    #output=output*alpha
    result_index=[]
    for k in input_idx:
        if cor_index.loc[str(gene_names[output_idx]),str(gene_names[k])] > threshold :
            result_index.append(k)
    if len(result_index)<1:
        vi = np.zeros(ngenes)
        return vi
    expr_data_input = expr_data[:,result_index]
    treeEstimator1=RandomForestRegressor(n_estimators=treen_estimators,max_features="auto",max_depth=4,n_jobs=-1,random_state=0)
    treeEstimator1.fit(expr_data_input,output)
    feature_importances_random_tree=treeEstimator1.feature_importances_
    vi = np.zeros(ngenes)
    vi[result_index] = feature_importances_random_tree
    return vi

	
def get_scores(VIM, gold_edges, gene_names, regulators):
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
    pred_edges = pd.DataFrame(pred_edges)
    pred_edges.sort_values(2, ascending=False, inplace=True)
    # Take the top 100000 predicted results
    pred_edges = pred_edges.iloc[:100000]
    final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
    auroc = roc_auc_score(final['2_y'], final['2_x'])
    aupr = average_precision_score(final['2_y'], final['2_x'])
    return auroc, aupr



coldstress_data = pd.read_csv('data\heat_1.csv', sep=',')
symbol_word='heat_1'
cor_index=pd.read_csv("MIC_cor\MIC_heat_1.csv",index_col=0)
output_file='LSGRN_heat_1_new_results.csv'
gold_edges = pd.read_csv("Ecoli\DREAM5_NetworkInference_GoldStandard_Network3.tsv", sep='\t', header=None)
gene_names = list(coldstress_data.columns[:])
coldstress_data = coldstress_data.values[:, :]
regulators = [i for i in set(gold_edges[0]) if i in gene_names]
regulators=gene_names


treen_estimators=1200
threshold=0.56
xgb_learning_rate=0.01
print('threshold'+str(threshold))
print('xgb_learning_rate'+str(xgb_learning_rate))

xgb_param = dict(learning_rate=xgb_learning_rate,max_depth=5, subsample=0.86, n_jobs=-1, n_estimators=120)
VIM = get_importances(coldstress_data, gene_names=gene_names,regulators=regulators, param=xgb_param)
auroc, aupr = get_scores(VIM, gold_edges, gene_names, regulators)
print("AUROC:", auroc, "AUPR:", aupr)
print('I am writting!')
with open(output_file,'a')as file_object: 
        file_object.write('AUROC:'+str(auroc))
        file_object.write(",") 
        file_object.write('AUPR:'+str(aupr))
        file_object.write(",") 
        file_object.write('threshold:'+str(threshold))
        file_object.write(",") 
        file_object.write('xgb_learning_rate:'+str(xgb_learning_rate))
        file_object.write(",")
        file_object.write('treen_estimators:'+str(treen_estimators))
        file_object.write(",")
        file_object.write("\n") 
print('Writting finish!')