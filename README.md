# PLS Modeling with K-Means Pro Clustering Approach

This project implements Partial Least Squares (PLS) modeling combined with a **K-Means Pro clustering approach**. The algorithm first clusters the data and then builds separate PLS models for each cluster. When a new observation is provided, the algorithm determines which cluster it belongs to and uses the appropriate PLS model to estimate the output (Y). This approach improves the consistency of the dataset and creates more accurate models by training on clusters of similar data points.

**This project is based on our publication:**
**[Predicting the Volume Phase Transition Temperature of Multi-Responsive Poly(N-isopropylacrylamide)-Based Microgels Using a Cluster-Based Partial Least Squares Modeling Approach](https://doi/abs/10.1021/acsapm.2c01487)**  
If you use this code in your research or project, please cite the above publication.

## Features

- **Cluster-Based PLS Modeling**: Utilizes **K-Means Pro clustering** to group data before developing PLS models. 
- **Number of Clusters**: The number of clusters can either be determined by the user or automatically chosen such that no cluster holds less than 20% of the original dataset.
- **Clustering in Latent Space**: Clustering is performed using the first two latent space scores (or the first score if only one component is selected).
- **Handling New Observations**: For new observations, the algorithm assigns the observation to a cluster and applies the relevant PLS model.
- **Improved Consistency**: Cluster-specific models trained on more consistent subsets of the data are expected to have better predictive power.

## Algorithm Overview

1. **Clustering**: The data is clustered using **K-Means Pro** based on:
   - **Number of clusters**: Can either be defined by the user or automatically determined such that no cluster holds less than 20% of the dataset.
   - **Latent Space Clustering**: Clustering is performed using the first two latent scores (or just the first score if only one component is used).
   
2. **PLS Model Development**: For each cluster, a separate PLS model is developed.

3. **Prediction for New Observations**: When a new observation is introduced, the algorithm assigns it to a cluster and applies the corresponding PLS model to estimate the output (Y).

## Usage

```python
import numpy as np
from CB_PLS import CB_PLS as cpls

# Model further settings
np.set_printoptions(precision=4)

# Data creation
Num_sam = 40
input_var = 5
output_var = 4
num_clusters = 5
N_testing = 50

X = np.random.rand(Num_sam, input_var)
X_new = np.random.rand(N_testing, input_var)
true_beta = np.random.rand(output_var, input_var) * 2 - 1
Y = (X @ true_beta.T).reshape(Num_sam, output_var)
y_new = X_new @ true_beta.T

# Training the model without clustering
cluster_pls_no = cpls()
cluster_pls_no.Train(X, Y, 1, K=1, ploting_clusters=False)
Y_new_pre_no, pre_accuracy_no = cluster_pls_no.evaluator(X_new, None, y_new)

# Training the model with X clustering
cluster_pls_x = cpls()
cluster_pls_x.Train(X, Y, 1, K=num_clusters, ploting_clusters=True)
Y_new_pre_x, pre_accuracy_x = cluster_pls_x.evaluator(X_new, None, y_new)

# Training the model with Y clustering
cluster_pls_y = cpls()
cluster_pls_y.Train(X, Y, 2, K=num_clusters, ploting_clusters=True)
Y_new_pre_y, pre_accuracy_y = cluster_pls_y.evaluator(X_new, None, y_new, y_hosting_method=1)

# Training the model with both X and Y clustering
cluster_pls_xy = cpls()
cluster_pls_xy.Train(X, Y, 3, K=num_clusters, ploting_clusters=True)
Y_new_pre_xy, pre_accuracy_xy = cluster_pls_xy.evaluator(X_new, None, y_new)

# pringitng the results 
print(f'prediction Accuracy No Clustering (AVERAGE={np.mean(pre_accuracy_no)})')
print(f'prediction Accuracy X Clustering (AVERAGE={np.mean(pre_accuracy_x)})')
print(f'prediction Accuracy Y Clustering (AVERAGE={np.mean(pre_accuracy_y)})')
print(f'prediction Accuracy XY Clustering (AVERAGE={np.mean(pre_accuracy_xy)})')
```

## Reference

This algorithm is described in detail in the following publication:

**[Predicting the Volume Phase Transition Temperature of Multi-Responsive Poly(N-isopropylacrylamide)-Based Microgels Using a Cluster-Based Partial Least Squares Modeling Approach](https://doi/abs/10.1021/acsapm.2c01487)**  
Published in *ACS APPLIED POLYMER MATERIALS*. Please cite this paper if you use this algorithm in your work.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
