import numpy as np
from scipy.stats import chi2, f
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MyPca:
    def __init__(self):
        # Initialize attributes if needed
        self.T = None
        self.P = None  
        self.x_hat =None 
        self.tsquared =None
        self.T2_lim=None
        self.ellipse_radius =None
        self.SPE_x =None
        self.SPE_lim_x =None
        self.Rsquared =None
        self.covered_var =None
        self.x_scaling =None
        self.Xtrain_normal =None
        self.Xtrain_scaled =None
        self.alpha =None
        self.Num_com =None

    def train(self,X, Num_com=None, alpha=0.95, to_be_scaled=1):
        
        if Num_com is None: 
            Num_com=X.shape[1]
        if Num_com>(X.shape[0]-1):
            Num_com=X.shape[0]-1
        # Data Preparation
        X_orining = X
        Cx = np.mean(X, axis=0)
        Sx = np.std(X, axis=0,ddof=1) + 1e-16

        if to_be_scaled==1:
            X = (X - Cx) / Sx
        
        Num_obs = X.shape[0]
        K = X.shape[1]  # Num of X Variables
        X_0 = X

        # Blocks initialization
        T = np.zeros((Num_obs, Num_com))
        P = np.zeros((K,Num_com))
        covered_var=np.zeros((1,Num_com))
        SPE_x = np.zeros_like(T)
        SPE_lim_x = np.zeros(Num_com)
        tsquared = np.zeros_like(T)
        T2_lim = np.zeros(Num_com)
        ellipse_radius = np.zeros(Num_com)
        Rx = np.zeros(Num_com)

        # NIPALS Algorithm
        for i in range(Num_com):
            t1 = X[:, np.argmax(np.var(X_orining, axis=0,ddof=1))]
            while True:
                P1=(t1.T@X)/(t1.T@t1)
                P1=P1/np.linalg.norm(P1)
                t_new=((P1@X.T)/(P1.T@P1)).T

                Error = np.sum((t_new - t1) ** 2)
                t1=t_new
                if Error < 1e-16:
                    break
            x_hat=t1.reshape(-1,1)@P1.reshape(1,-1)
            X=X-x_hat
            P[:,i]=P1
            T[:,i]=t1

            covered_var[:,i]=np.var(t1,axis=0,ddof=1)
            # SPE_X
            SPE_x[:, i], SPE_lim_x[i], Rx[i] = self.SPE_calculation(T, P, X_0, alpha)

            # Hotelling T2 Related Calculations
            tsquared[:, i], T2_lim[i], ellipse_radius[i] = self.T2_calculations(T[:, :i+1], i+1, Num_obs, alpha)

        # Function Output
        self.T=T
        self.P=P
        self.x_hat=((T @ P.T)*Sx)+Cx
        self.tsquared=tsquared
        self.T2_lim=T2_lim
        self.ellipse_radius=ellipse_radius
        self.SPE_x=SPE_x
        self.SPE_lim_x=SPE_lim_x
        self.Rsquared=Rx.T*100
        self.covered_var=covered_var
        self.x_scaling=np.vstack((Cx, Sx))
        self.Xtrain_normal=X_orining
        self.Xtrain_scaled=X_0
        self.alpha=alpha
        self.Num_com=Num_com
        
        return self

    def evaluation(self,X_new): 
        """
        receive pca model and new observation and calculate its
        x_hat,T_score,Hotelin_T2,SPE_X
        """
        x_new_scaled=self.scaler(X_new)

        T_score=x_new_scaled @ self.P
        x_hat_new_scaled=T_score @ self.P.T

        x_hat_new=self.unscaler(x_hat_new_scaled)
        Hotelin_T2=np.sum((T_score/np.std(self.T,axis=0,ddof=1))**2,axis=1)
        SPE_X,_,_ = self.SPE_calculation(T_score, self.P, x_new_scaled, self.alpha)

        return x_hat_new,T_score,Hotelin_T2,SPE_X

    def SPE_calculation(self,score, loading, Original_block, alpha):
        # Calculation of SPE and limits
        X_hat = score @ loading.T
        Error = Original_block - X_hat
        #Error.reshape(-1,loading.shape[1])
        spe = np.sum(Error**2, axis=1)
        spe_lim, Rsquare=None,None
        if Original_block.shape[0]>1:
            m = np.mean(spe)
            v = np.var(spe,ddof=1)
            spe_lim = v / (2 * m) * chi2.ppf(alpha, 2 * m**2 / (v+1e-15))
            Rsquare = 1 - np.var(Error,ddof=1) / np.var(Original_block,ddof=1) # not applicaple for pls vali
        return spe, spe_lim, Rsquare

    def T2_calculations(self,T, Num_com, Num_obs, alpha):
        # Calculation of Hotelling T2 statistics
        tsquared = np.sum((T / np.std(T, axis=0,ddof=1))**2, axis=1)
        T2_lim = (Num_com * (Num_obs**2 - 1)) / (Num_obs * (Num_obs - Num_com)) * f.ppf(alpha, Num_com, Num_obs - Num_com)
        ellipse_radius = np.sqrt(T2_lim * np.std(T[:, Num_com - 1],ddof=1)**2)
        return tsquared, T2_lim, ellipse_radius


    def scaler(self,X_new):

        Cx=self.x_scaling[0,:]
        Sx=self.x_scaling[1,:]
        X_new=(X_new-Cx)/Sx
        
        return X_new
    
    def unscaler(self,X_new):
        Cx=self.x_scaling[0,:]
        Sx=self.x_scaling[1,:]
        X_new=(X_new * Sx) + Cx
        return X_new


    def visual_plot(self, score_axis=None, X_test=None, color_code_data=None, data_labeling=False, testing_labeling=False):

        # inner Functions
        def confidenceline(r1, r2, center):
            t = np.linspace(0, 2 * np.pi, 100)  # Increase the number of points for a smoother ellipse
            x = center[0] + r1 * np.cos(t)
            y = center[1] + r2 * np.sin(t)
            return x, y
        
        def inner_ploter(y_data,position,legend_str,X_test=None,y_data_add=None,legend_str2=None):       
            X_data = np.arange(1, len(y_data) + 1)
            fig2.add_trace(go.Scatter(x=X_data, y=y_data, mode='markers', marker=dict(color='blue', size=10), name=legend_str,showlegend=True),
                    row=position[0], col=position[1])
            if X_test is not None:
                y_data = np.concatenate((y_data, y_data_add))
                X_data = np.arange(1, len(y_data) + 1)
                fig2.add_trace(go.Scatter(x=X_data[Num_obs:], y=y_data[Num_obs:], mode='markers', marker=dict(color='red', symbol='star', size=12), name=legend_str2,showlegend=True),
                        row=position[0], col=position[1])
            fig2.add_trace(go.Scatter(x=[1, X_data[-1] + 1], y=[self.T2_lim[-1]] * 2, mode='lines', line=dict(color='black', dash='dash'), name='Hoteling T^2 Lim',showlegend=False),
                    row=position[0], col=position[1])
            fig2.update_xaxes(
            tickmode='linear',  # Ensures all ticks are shown linearly
            tick0=2,            # Starting tick (adjust if needed)
            dtick=1,            # Interval between ticks
            range=[0.5, len(X_data)+0.5],
            row=position[0], col=position[1] ) # Apply only to the specific subplot

        # Ploting Parameters
        Num_obs, Num_com = self.T.shape
        if score_axis is None:
            score_axis = np.array([1, min(2, Num_com)])

        # Create subplots
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'PLS Score Plot Distribution','Loading of'+str(score_axis[0])+'Component','Loading of'+str(score_axis[1])+'Component'),          
            specs=[[{"colspan": 2}, None],   
                [{}, {}]],row_heights=[0.5, 0.5])

        fig2 = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'SPE_X','Hoteling T^2 Plot'),          
            specs=[[{}],[{}]],row_heights=[0.5, 0.5]
        )
        # axis labeling
        fig1.update_xaxes(title_text='T '+str(score_axis[0])+'score',row=1,col=1)
        fig1.update_yaxes(title_text='T '+str(score_axis[1])+'score',row=1,col=1)
        fig2.update_xaxes(title_text='Observations',row=4,col=1)
        #score plot
        tscore_x = self.T[:, score_axis[0] - 1]
        tscore_y = self.T[:, score_axis[1] - 1]

        r1 = self.ellipse_radius[score_axis[0] - 1]
        r2 = self.ellipse_radius[score_axis[1] - 1]
        xr, yr = confidenceline(r1, r2, np.array([0, 0]))
        label_str = f'Confidence Limit ({self.alpha * 100}%)'

        fig1.add_trace(go.Scatter(x=xr, y=yr, mode='lines', line=dict(color='black', dash='dash'), name=label_str,showlegend=True),row=1, col=1)
        if color_code_data is None:
            fig1.add_trace(go.Scatter(x=tscore_x, y=tscore_y, mode='markers', marker=dict(color='blue', size=10), name='Score(Training Dataset)',showlegend=True),row=1, col=1)
        else:
            fig1.add_trace(go.Scatter(x=tscore_x, y=tscore_y,mode='markers',marker=dict(size=10,color=color_code_data,colorscale='Viridis', colorbar=dict(orientation='h',y=1.3,yanchor='top'),showscale=True,),name='Score(Training Dataset)', showlegend=True,),row=1,col=1)
        
        
        if data_labeling:  
            for i in range(Num_obs):   
                fig1.add_trace(go.Scatter(x=[tscore_x[i]], y=[tscore_y[i]], text=str(i + 1), mode='text', textposition='top center',showlegend=False),
                        row=1, col=1)
        # Testing Data
        tscore_testing, hoteling_t2_testing, spe_x_testing=None,None,None
        if X_test is not None:
            Num_new = X_test.shape[0]
            _, tscore_testing, hoteling_t2_testing, spe_x_testing = self.evaluation(X_test)

            t_score_x_new = tscore_testing[:, score_axis[0] - 1]
            t_score_y_new = tscore_testing[:, score_axis[1] - 1]

            fig1.add_trace(go.Scatter(x=t_score_x_new, y=t_score_y_new, mode='markers', marker=dict(color='red', symbol='star', size=12), name='Score(New Data)',showlegend=True),
                        row=1, col=1)
            if testing_labeling:
                for i in range(Num_new):
                    fig1.add_trace(go.Scatter(x=[t_score_x_new[i]], y=[t_score_y_new[i]], text=str(i + 1), mode='text', textposition='top center',showlegend=False),
                                row=1, col=1)

        # Loading bar plots
        for k in range(2):
            Num_var_X=self.Xtrain_normal.shape[1]
            x_data = np.empty(Num_var_X, dtype=object)
            y_data=self.P[:,k]
            for j in range(Num_var_X):
                x_data[j]='variable '+str(j+1)
            fig1.add_trace(go.Bar(x=x_data,y=y_data,name='Loding'+str(score_axis[k]),marker=dict(color='blue')),row=2,col=k+1)
        
        # SPE_X Plot
        y_data = self.SPE_x[:, -1]
        inner_ploter(y_data,[1,1],'SPE_X(Training Data)',X_test,spe_x_testing,'SPE_X(New Data)')
        # Hoteling T^2 Plot
        y_data = self.tsquared[:, -1]
        inner_ploter(y_data,[2,1],'Hoteling T2(Training Data)',X_test,hoteling_t2_testing,'Hoteling T2(New Data)')
    
    
        # Update layout for font sizes and other customization
        fig1.update_layout(title_text='PLS Model Visual Plotting(scores)',title_x=0.5,font=dict(size=15),legend=dict(x=1, y=1, traceorder='normal'),showlegend=True )
        fig2.update_layout(title_text='PLS Model Visual Plotting(Statistics)',title_x=0.5,font=dict(size=15),legend=dict(x=1, y=1, traceorder='normal'),showlegend=True )
        fig1.show()
        fig2.show()

