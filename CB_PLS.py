#%%
import numpy as np
from MyPcaClass import MyPca as pca
from MyPlsClass import MyPls as pls
from kmeansPro import kmeansPro as kmp
import matplotlib.pyplot as plt
class CB_PLS:
    def __init__(self):
        self.X=None
        self.Y=None
        self.clustering_mode=None #1:based on X, 2:based on Y, 3:combined X and Y
        self.clustere_str=None #it would be Kmeans outcome
        self.pls_models=None
        self.other=None  # it would be second cb_pls if the mode is 3
        
    def Train(self,X,Y,clustering_mode=1,A_pca=None,A_pls=None,ploting_clusters=False,K=None,min_cluster_threshold=0.2):
        '''
        receive the data and clustering_mode (either based on X,Y or both)
        if it is based on Y it clusters data based on Y and then find the hosting cluster
        for future X based on X that are in the same cluster and find the best cluster that way
        if clustering Mode is both the output is the mean of both type of clustering
        '''
        if clustering_mode==1 or clustering_mode==2:
            Z=X
            if clustering_mode==2:
                Z=Y
            self.Classifier_Fcn(Z,A_pca,ploting_clusters,K,min_cluster_threshold)
            self.Pls_developer(X,Y,A_pls)
            self.X=X
            self.Y=Y
        elif clustering_mode==3:
            self.Train(X,Y,1,A_pca,A_pls,ploting_clusters,K)
            self.other=CB_PLS()
            self.other.Train(X,Y,2,A_pca,A_pls,ploting_clusters,K)
        self.clustering_mode=clustering_mode

    def Classifier_Fcn(self,Z,A=None,ploting_clusters=False,K=None,min_cluster_threshold=0.2):
        '''
        should receive either X or Y (Z can be either) for which it want to apply the clustering
        '''
        # determine the number of components (using Num_components_sugg) if it is not given by the used
        if A is None:
            A=self.Num_components_sugg(Z)
        # developing the pca
        pca_model=pca()
        pca_model.train(Z,A)
        to_be_clustered_data=pca_model.T
        # sending the corresponding scores for clustering
        Kmean_structure=kmp()
        if K==1:
            Num_repeat=1
        else:
            Num_repeat=1000
        while True:
            
            Kmean_structure.fit(to_be_clustered_data,K,Num_repeat)
            min_count=np.min(Kmean_structure.clustr_counts)
            if min_count<=3:
                K=K-1
                K=Kmean_structure.max_possible_num_cluster(to_be_clustered_data,min_cluster_threshold,K0=K)
            elif min_count>3:
                break
        Kmean_structure.pca_model=pca_model
        if ploting_clusters is True:
            Kmean_structure.visual_ploting()
        
        #returning the clustering structure
        self.clustere_str = Kmean_structure
    
    def Pls_developer(self,X,Y,A_pls=None):
        '''
        developing PLS models for each cluster (for now use Num_components_sugg) 
        '''
        kmean_structure=self.clustere_str
        idx=kmean_structure.idx.reshape(-1)
        K=kmean_structure.NumCluster
        pls_models=np.empty(K,dtype=pls)
        for i in range(K):
            x_data=X[idx==i,:]
            y_data=Y[idx==i,:]
            pls_model=pls()
            A_pls=self.Num_components_sugg(x_data)
            pls_model.train(x_data,y_data,A_pls)
            pls_models[i]=pls_model
        self.pls_models=pls_models
        
    def Num_components_sugg(self,Z,ploting=False):
        '''
        Z can be either X or Y
        determines the number of components that describe the data quite good using the eigenvalue_greater_than_one_rule
        '''
        pca_model=pca()
        pca_model.train(Z)
        eig_val=pca_model.covered_var
        Num_com_sugg=np.sum(eig_val>1)
        if ploting==True:
            plt.figure()
            plt.bar(range(1,eig_val.shape[1]+1),eig_val.reshape(-1),label='Covered Variance')
            plt.xlabel('Components')
            plt.ylabel('Variance Covered')
            plt.plot([0,eig_val.shape[1]+1],[1,1],'k--',label='Threshold Line')
            plt.legend()
            plt.show()
        # plot the eig_Val using barchart to show if suggested A is rational or not

        return Num_com_sugg

    def Cluster_determination(self,clustering_mode,x_new,y_hosting_method=1,x_hosting_method=1,k_nn=3,A_pca=None,ploting=False):
        '''
        if the samples are clustered based on X it is easier as we can directly calculate the score of this new sample and find its hosting cluster 
        if samples are clustered based on y, so now we have have some X data in different clusters
        and we also have a x_new for which we need to define the hosting cluster,using different optional methods
        we develop pca over data in each group and then assess the similarity of the new data to each of these pca models
        method 1 : based on the lowest normalized PCA spe
        method 2 : based on the lowest normalized PCA hotelingT 
        method 3 : based on both (the one with the lowest summation of the two above )
        '''
        if clustering_mode==1: # clustered based on X
            __,corre_scores,__,__=self.clustere_str.pca_model.evaluation(x_new)            
            hosting_cluster=self.clustere_str.hosting_cluster(corre_scores,x_hosting_method,k_nn)
        elif clustering_mode==2: # clustered based on Y
            Xtr=self.X
            idx_y=self.clustere_str.idx.reshape(-1)
            K=self.clustere_str.NumCluster
            spe_normalized=np.zeros((K,x_new.shape[0]))
            hoteling_normalized=np.zeros_like(spe_normalized)
            spe_i=np.zeros((1,x_new.shape[0]))
            hotelingt2_i=np.zeros_like(spe_i)
            for i in range(K):
                x_pca=Xtr[idx_y==i,:]
                A_pca=self.Num_components_sugg(x_pca,ploting)
                ith_pca_model=pca()
                ith_pca_model.train(x_pca,A_pca)
                __,__,hotelingt2_i,spe_i=ith_pca_model.evaluation(x_new)
                spe_normalized[i,:]=spe_i/(ith_pca_model.SPE_lim_x[-1]+0.00001)
                hoteling_normalized[i,:]=hotelingt2_i/ith_pca_model.T2_lim[-1]
            if y_hosting_method==1:  # spe normalized
                hosting_cluster=np.argmin(spe_normalized,axis=0)
            elif y_hosting_method==2: # hoteling normalized
                hosting_cluster=np.argmin(hoteling_normalized,axis=0)
            elif y_hosting_method==3: # both nortmalized
                sum_spe_hoteling=spe_normalized+hoteling_normalized
                hosting_cluster=np.argmin(sum_spe_hoteling,axis=0)
        return hosting_cluster
            
    def evaluator(self,x_new,clustering_mode=None,y_new=None,y_hosting_method=1,x_hosting_method=1,k_nn=3):
        '''
        first find the new data hosting cluster and using its corresponding pls model make the predictions
        '''
        if clustering_mode is None:
            clustering_mode=self.clustering_mode
        # if mode is 1 or 2
        if clustering_mode==1 or clustering_mode==2:
            y_pre=np.zeros((x_new.shape[0],self.Y.shape[1]))
            hosting_cluster=self.Cluster_determination(clustering_mode,x_new,y_hosting_method,x_hosting_method,k_nn)
            for ii in range(x_new.shape[0]):   
                y_pre[ii,:],__,__,__,__=self.pls_models[int(hosting_cluster[ii])].evaluation(x_new[ii,:].reshape(1,-1))
    
        elif clustering_mode==3: #
            y_pre1,__=self.evaluator(x_new,1,y_new,y_hosting_method,x_hosting_method,k_nn)
            y_pre2,__=self.other.evaluator(x_new,2,y_new,y_hosting_method,x_hosting_method,k_nn)
            y_pre=(y_pre1+y_pre2)/2     
        
        # Y_actual implementation for error calculation
        Prediction_accuracy=None
        if y_new is not None:
            Prediction_accuracy=self.Single_obs_error_calculation(y_new,y_pre,self.Y)            
        return y_pre , Prediction_accuracy
    
    def Single_obs_error_calculation(self,y_act,y_pre,Y_act=None):
        '''
        it receives y and y_pre and calculte the single prediction accuracy
        it need Y (the entire Y block to make sure there is not bias caused by the magnitude of th ecolomns)
        '''
        if Y_act is None:
            Y_act=y_act
        pa=np.zeros_like(y_act)
        for i in range(y_act.shape[1]):
            base_value=np.min(Y_act[:,i])
            scaled_Y=Y_act[:,i]-base_value
            Y_avr=np.mean(scaled_Y)
            error=np.abs(y_act[:,i]-y_pre[:,i])
            pa[:,i]=1-(error/Y_avr)
        Prediction_accuracy=np.mean(pa,axis=1)
        return Prediction_accuracy