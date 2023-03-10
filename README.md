# iLSGRN: Inference of large-scale gene regulatory networks based on multi-model fusion


1 College of Information Science and Technology, Dalian Maritime University, Dalian 116039, China



**LSGRN is a large-scale gene regulatory network inference method based on multi model fusion, which includes dimension reduction using maximum mutual information coefficient and feature fusion of XGBoost and RF machine learning models.** 

If you find our method is useful, please cite our paper:




### The version of Python and packages
Python version       3.8.5
minepy                  1.2.5
numpy                   1.20.3
pandas                  1.2.4
scikit-learn            0.24.2
scipy                   1.6.3
xgboost                 1.4.2



### Parameters Description
    
        alpha:a constant of gene decay rate
        param: a dict of parameters of xgboost
        threshold: Threshold of maximum mutual information coefficient dimension reduction
        xgb_learning_rate: Learning rate of xgboost
    	
    case_size100:
        TS_data: a matrix of time-series data
        time_points: a list of time points
        SS_data: a matrix of time-series data, the default is "none"
        gene_names: a list of gene names
        regulators: a list of names of regulatory genes, the default is "all", 
        param: a dict of parameters of xgboost and RF


