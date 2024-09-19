#%%
import numpy as np
from CB_PLS import CB_PLS as cpls
from kmeansPro import kmeansPro as kmp
# Model further settings
np.set_printoptions(precision=4)

# data creation
Num_sam=100
input_var=5
output_var=4
num_clusters=5
N_testing=5

X=np.random.rand(Num_sam,input_var)
X_new=np.random.rand(N_testing,input_var)
true_beta=np.random.rand(output_var,input_var)*2-1
Y=(X @ true_beta.T).reshape(Num_sam,output_var)
y_new=X_new @ true_beta.T

# %%
# Training the model No clustering 
cluster_pls_no=cpls()
cluster_pls_no.Train(X,Y,1,K=1,ploting_clusters=True)
Y_new_pre_no,pre_accuracy_no=cluster_pls_no.evaluator(X_new,None,y_new)
# Training the model X clustering
cluster_pls_x=cpls()
cluster_pls_x.Train(X,Y,1,K=num_clusters,ploting_clusters=True)
Y_new_pre_x,pre_accuracy_x=cluster_pls_x.evaluator(X_new,None,y_new)

# Training the model Y clustering
cluster_pls_y=cpls()
cluster_pls_y.Train(X,Y,2,K=num_clusters,ploting_clusters=True)
Y_new_pre_y,pre_accuracy_y=cluster_pls_y.evaluator(X_new,None,y_new,y_hosting_method=1)

# Training the model both Clustering
cluster_pls_xy=cpls()
cluster_pls_xy.Train(X,Y,3,K=num_clusters,ploting_clusters=True)
Y_new_pre_xy,pre_accuracy_xy=cluster_pls_xy.evaluator(X_new,None,y_new)

# pringitng the results 
print(f'prediction Accuracy No Clustering (AVERAGE={np.mean(pre_accuracy_no)})')#,all = ({pre_accuracy_no})')#
print(f'prediction Accuracy X Clustering (AVERAGE={np.mean(pre_accuracy_x)})')#,all = ({pre_accuracy_x})')
print(f'prediction Accuracy Y Clustering (AVERAGE={np.mean(pre_accuracy_y)})')#,all = ({pre_accuracy_y})')
print(f'prediction Accuracy XY Clustering (AVERAGE={np.mean(pre_accuracy_xy)})')#,all = ({pre_accuracy_xy})')
# Result Ploting

# %% which y_clustering method is better
# Training the model Y clustering
cluster_pls_y=cpls()
cluster_pls_y.Train(X,Y,2,K=num_clusters,ploting_clusters=True)

Y_new_pre_y1,pre_accuracy_y1=cluster_pls_y.evaluator(X_new,None,y_new,y_hosting_method=1) # spe normalized
Y_new_pre_y2,pre_accuracy_y2=cluster_pls_y.evaluator(X_new,None,y_new,y_hosting_method=2) # hoteling normalized
Y_new_pre_y3,pre_accuracy_y3=cluster_pls_y.evaluator(X_new,None,y_new,y_hosting_method=3) # both normalized

print(f'prediction Accuracy Y Clustering method 1 (AVERAGE={np.mean(pre_accuracy_y1)})')#,all = ({pre_accuracy_no})')#
print(f'prediction Accuracy Y Clustering method 2 (AVERAGE={np.mean(pre_accuracy_y2)})')#,all = ({pre_accuracy_x})')
print(f'prediction Accuracy Y Clustering method 3 (AVERAGE={np.mean(pre_accuracy_y3)})')#,all = ({pre_accuracy_y})')
#input()
# %%
