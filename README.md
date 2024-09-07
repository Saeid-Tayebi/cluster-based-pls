# PLS Modeling with K-Means Clustering Approach

This project implements Partial Least Squares (PLS) modeling combined with a **K-Means clustering approach**. The algorithm first clusters the data and then builds separate PLS models for each cluster. When a new observation is provided, the algorithm determines which cluster it belongs to and uses the appropriate PLS model to estimate the output (Y). This approach improves the consistency of the dataset and creates more accurate models by training on clusters of similar data points.

**This project is based on our publication:**
**[Predicting the Volume Phase Transition Temperature of Multi-Responsive Poly(N-isopropylacrylamide)-Based Microgels Using a Cluster-Based Partial Least Squares Modeling Approach](https://doi/abs/10.1021/acsapm.2c01487)**  
If you use this code in your research or project, please cite the above publication.

### Note: This project is currently under preparation. The full code and documentation will be added soon. Stay tuned for updates!
## Features

- **Cluster-Based PLS Modeling**: Utilizes **K-Means clustering** to group data before developing PLS models. 
- **Number of Clusters**: The number of clusters can either be determined by the user or automatically chosen such that no cluster holds less than 20% of the original dataset.
- **Clustering in Latent Space**: Clustering is performed using the first two latent space scores (or the first score if only one component is selected).
- **Handling New Observations**: For new observations, the algorithm assigns the observation to a cluster and applies the relevant PLS model.
- **Improved Consistency**: Cluster-specific models trained on more consistent subsets of the data are expected to have better predictive power.

## Algorithm Overview

1. **Clustering**: The data is clustered using **K-Means** based on:
   - **Number of clusters**: Can be either defined by the user or automatically determined such that no cluster holds less than 20% of the dataset.
   - **Latent Space Clustering**: Clustering is performed using the first two latent scores (or just the first score if only one component is used).
   
2. **PLS Model Development**: For each cluster, a separate PLS model is developed.

3. **Prediction for New Observations**: When a new observation is introduced, the algorithm assigns it to a cluster and applies the corresponding PLS model to estimate the output (Y).

## Usage

```python
# Example of usage
# Assuming X_data and Y_data are numpy arrays

from pls_clustering_model import PLSClusteringModel

# Create PLS-Cluster model (automatic number of clusters)
pls_cluster_model = PLSClusteringModel(X_data, Y_data)

# Create PLS-Cluster model with user-defined number of clusters
pls_cluster_model = PLSClusteringModel(X_data, Y_data, num_clusters=5)

# Predicting Y for a new observation
new_observation = X_new[0, :]
y_predicted, cluster = pls_cluster_model.predict(new_observation)
```

## Reference

This algorithm is described in detail in the following publication:

**[Predicting the Volume Phase Transition Temperature of Multi-Responsive Poly(N-isopropylacrylamide)-Based Microgels Using a Cluster-Based Partial Least Squares Modeling Approach](https://doi/abs/10.1021/acsapm.2c01487)**  
Published in *ACS APPLIED POLYMER MATERIAL*. Please cite this paper if you use this algorithm in your work.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---
