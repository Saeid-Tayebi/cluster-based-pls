import numpy as np
import plotly.graph_objects as go
import plotly.colors as colors


class kmeansPro:
    def __init__(self):
        kmeansPro.NumCluster=None
        kmeansPro.centers=None
        kmeansPro.points=None
        kmeansPro.idx=None
        kmeansPro.goodness=None
        kmeansPro.clustr_counts=None

    def replace_with(self, other):
        # Replace attributes of self with those of other
        self.__dict__.update(other.__dict__)

    def fit(self,data:np.ndarray,k=None,Num_repeat=50):
        '''
        if k is not given it calculate the K using max possible method with 20% sample in each cluster
        Num_repeat is the number of times that it goes through the fiting the clusters to find the best data clustering with the most uniform distance between samples and centers

        '''
        def inner_fit(clustered_i,data:np.ndarray,k:int):
            Num_sam,Num_var=data.shape
            idx=np.zeros((Num_sam,1))
            dist_to_center=np.zeros_like(idx)
            C=np.zeros((k,Num_var))

            #Center initializing (Prevent any empty cluster)
            selected_samples = np.random.choice(data.shape[0], size=k, replace=False)
            C=data[selected_samples,:]
            #Assignment
            stop_condition=0
            while True:
                idx_old=idx.copy()
                for i in range(Num_sam):
                    xi=data[i,:]
                    d=np.zeros((k,1))
                    for j in range(k):
                        d[j]=np.linalg.norm(C[j,:]-xi)
                    dist_to_center[i]=np.min(d)
                    idx[i]=np.argmin(d)
                # Center updating
                for i in range(k):
                    C[i,:]=np.mean(data[idx.reshape(-1)==i,:],axis=0)
                
                if np.all(idx==idx_old):
                    stop_condition=stop_condition+1
                if stop_condition==10:
                    break
            goodness=np.std(dist_to_center,ddof=1)
            __, clustr_counts = np.unique(idx,return_counts=True)

            clustered_i.NumCluster=k
            clustered_i.centers=C
            clustered_i.points=data
            clustered_i.idx=idx
            clustered_i.clustr_counts=clustr_counts
            clustered_i.goodness=goodness
            return clustered_i
        
        if k is None:
            k=self.max_possible_num_cluster(data)
        clusters_list=[None]*Num_repeat
        goodness_list=np.zeros((Num_repeat,1))
        for i in range(Num_repeat):
            clustered_i=kmeansPro()
            clustered_i=inner_fit(clustered_i,data,k)
            clusters_list[i]=clustered_i
            goodness_list[i]=clustered_i.goodness
        best_one_idx=np.argmin(goodness_list)
        best_cluster=clusters_list[best_one_idx]
        self.replace_with(best_cluster)
    
    def max_possible_num_cluster(self,data:np.ndarray,threshold=0.2,K0=None):
        '''
        find the max possible number of cluster to make sure each cluster contains at least
        threshold portion of the entire samples
        if K0 is a number in would check its possibility and if not if would decrease the k0 from this point down
        '''
        Num_sam=data.shape[0]
        threshold=np.round(threshold*Num_sam)
        if K0 is None:
            K0=int(np.round(Num_sam/threshold))
        kmean_model=kmeansPro()  
        while True:
            kmean_model.fit(data,K0)
            unique, counts = np.unique(kmean_model.idx,return_counts=True)
            if np.min(counts)>threshold:
                break
            K0=K0-1
        return K0

    def visual_ploting(self,axis_plot=None,new_candiates=None,method=1,K_nn=3):
        '''
        it plot the data distribution among the clusters
        it also show the new data with its assign cluster
        '''
        fig=go.Figure()
        Num_sam,Num_var=self.points.shape
        data=self.points
        centers=self.centers
        idx=self.idx.reshape(-1)
        K=self.NumCluster
        if axis_plot is None:
            axis_plot=np.array([1, min(2, Num_var)])

        axis_plot=axis_plot-1
        color_scale = colors.get_colorscale('Jet')  # Get the 'Jet' colorscale
        color_map = [colors.sample_colorscale(color_scale, i / (K-1)) for i in range(K)]  # Generate k distinct colors
        for i in range(K):
            fig.add_trace(go.Scatter(x=data[idx==i,axis_plot[0]], y=data[idx==i,axis_plot[1]],name='Cluster'+str(i+1),mode='markers',marker=dict(size=10,color=color_map[i]*Num_sam), showlegend=True))
            fig.add_trace(go.Scatter(x=[centers[i,axis_plot[0]]],y=[centers[i,axis_plot[1]]],name='Centers cluster'+str(i+1),mode='markers',marker=dict(size=20,color=color_map[i],symbol='hexagon'),showlegend=True))

        if new_candiates is not None:
            candidates_idx=self.hosting_cluster(new_candiates,method,K_nn).reshape(-1)
            Num_candidates=new_candiates.shape[0]
            candiates_color=[None]*Num_candidates
            for i in range(candidates_idx.shape[0]):
                candiates_color[i]=color_map[int(candidates_idx[i])]
            for j in range(Num_candidates):
                fig.add_trace(go.Scatter(x=[new_candiates[j,axis_plot[0]]], y=[new_candiates[j,axis_plot[1]]],name='New Candidates'+str(j+1),mode='markers',marker=dict(size=20,color=candiates_color[j],symbol='star'), showlegend=True))
            
        fig.update_layout(width=600,height=600,xaxis=dict(scaleanchor="y", scaleratio=1))
        fig.show()
    def hosting_cluster(self,new_candiates,method=1,K_nn=3):
        """
        method1 determin the hosting cluster based on nearest center
        method2 determin the hosting cluster based on N-nearest neighbor
        """
        K=self.NumCluster
        C=self.centers
        points=self.points
        idx=self.idx
        Num_sam=self.points.shape[0]
        Num_new_points=new_candiates.shape[0]
        hosting_cluster=np.zeros((Num_new_points,1))
        for j in range(Num_new_points):
            if method==1:  
                d=np.zeros((K,1))
                for i in range(K):
                    d[i]=np.linalg.norm(C[i,:]-new_candiates[j,:])
                hosting_cluster[j]=np.argmin(d)
            else:
                distance=np.zeros((Num_sam,))
                for i in range(Num_sam):
                    distance[i]=np.linalg.norm(points[i,:]-new_candiates[j,:])
                idx_of_knn=idx[np.argsort(distance)[:K_nn]]   
                unique, counts = np.unique(idx_of_knn, return_counts=True)
                hosting_cluster[j] = unique[np.argmax(counts)]
        return hosting_cluster