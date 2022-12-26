import numpy as np
from minepy import MINE
import matplotlib.pyplot as plt
import pandas as pd

file1="data\heat_1.csv"
output_file='MIC_cor\MIC_heat_1_test.csv'
TS_data = pd.read_csv(file1)
head_row=list(TS_data.columns[:])
headrow=head_row.copy()
headrow.insert(0,'cor')
df = pd.DataFrame(columns=headrow)

for i in head_row:
    add_data =[i]
    for j in head_row:
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(TS_data[i],TS_data[j])
        add_data.append(mine.mic())
    add=pd.Series(add_data,index=headrow)
    df=df.append(add, ignore_index=True)
    add_data=[]

df.to_csv(output_file, index=False, sep=",",mode='a')