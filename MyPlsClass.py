import numpy as np
from scipy.stats import chi2, f
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MyPls:
    def __init__(self):
        # Initialize attributes if needed
        self.T = None
        self.S = None
        self.P = None  
        self.u =None
        self.U =None
        self.Q =None
        self.Wstar =None
        self.B_pls =None
        self.x_hat_scaled = None
        self.y_fit_scaled = None  
        self.tsquared =None
        self.T2_lim=None
        self.ellipse_radius =None
        self.SPE_x =None
        self.SPE_lim_x =None
        self.SPE_y =None
        self.SPE_lim_y =None
        self.Rsquared =None
        self.covered_var =None
        self.x_scaling =None
        self.y_scaling =None
        self.Xtrain_normal =None
        self.Ytrain_normal =None
        self.Xtrain_scaled =None
        self.Ytrain_scaled =None
        self.alpha =None
        self.Null_Space =None
        self.Num_com =None

    def train(self,X, Y, Num_com=None, alpha=0.95, to_be_scaled=1):

        if Num_com is None: 
            Num_com=X.shape[1]
        if Num_com>(X.shape[0]-1):
            Num_com=X.shape[0]-1
        # Data Preparation
        X_orining = X
        Y_orining = Y
        Cx = np.mean(X, axis=0)
        Cy = np.mean(Y, axis=0)
        Sx = np.std(X, axis=0,ddof=1) + 1e-16
        Sy = np.std(Y, axis=0,ddof=1) + 1e-16

        if to_be_scaled==1:
            X = (X - Cx) / Sx
            Y = (Y - Cy) / Sy


        Num_obs = X.shape[0]
        K = X.shape[1]  # Num of X Variables
        M = Y.shape[1]  # Num of Y Variables
        X_0 = X
        Y_0 = Y

        # Blocks initialization
        W = np.zeros((K, Num_com))
        U = np.zeros((Num_obs, Num_com))
        Q = np.zeros((M, Num_com))
        T = np.zeros((Num_obs, Num_com))
        P = np.zeros_like(W)
        SPE_x = np.zeros_like(T)
        SPE_y = np.zeros_like(T)
        SPE_lim_x = np.zeros(Num_com)
        SPE_lim_y = np.zeros(Num_com)
        tsquared = np.zeros_like(T)
        T2_lim = np.zeros(Num_com)
        ellipse_radius = np.zeros(Num_com)
        Rx = np.zeros(Num_com)
        Ry = np.zeros(Num_com)

        # NIPALS Algorithm
        for i in range(Num_com):
            u = Y[:, np.argmax(np.var(Y_orining, axis=0,ddof=1))]
            while True:
                w = X.T @ u / (u.T @ u)
                w = w / np.linalg.norm(w)
                t1 = X @ w / (w.T @ w)
                q1 = Y.T @ t1 / (t1.T @ t1)
                unew = Y @ q1 / (q1.T @ q1)
                Error_x = np.sum((unew - u) ** 2)
                u = unew
                if Error_x < 1e-16:
                    break

            P1 = X.T @ t1 / (t1.T @ t1)
            X = X - t1[:, None] @ P1[None, :]
            Y = Y - t1[:, None] @ q1[None, :]
            W[:, i] = w
            P[:, i] = P1
            T[:, i] = t1
            U[:, i] = unew
            Q[:, i] = q1
            # SPE_X
            SPE_x[:, i], SPE_lim_x[i], Rx[i] = self.SPE_calculation(T, P, X_0, alpha)

            # SPE_Y
            SPE_y[:, i], SPE_lim_y[i], Ry[i] = self.SPE_calculation(T, Q, Y_0, alpha)

            # Hotelling T2 Related Calculations
            tsquared[:, i], T2_lim[i], ellipse_radius[i] = self.T2_calculations(T[:, :i+1], i+1, Num_obs, alpha)

        Wstar = W @ np.linalg.pinv(P.T @ W)
        B_pls = Wstar @ Q.T
        S = np.linalg.svd(T.T @ T)[1]**0.5
        u = T / S

        # Null space
        A = Num_com
        KK = Y_orining.shape[1]
        if KK > A:
            Null_Space = 0
        elif KK == A:
            Null_Space = 1
        else:
            Null_Space = 2

        self.T=T
        self.S=S
        self.u=u
        self.P=P
        self.U=U
        self.Q=Q
        self.Wstar=Wstar
        self.B_pls=B_pls
        self.x_hat_scaled=T @ P.T

        self.y_fit_scaled=T @ Q.T,
        self.tsquared=tsquared
        self.T2_lim=T2_lim
        self.ellipse_radius=ellipse_radius
        self.SPE_x=SPE_x
        self.SPE_lim_x=SPE_lim_x
        self.SPE_y=SPE_y
        self.SPE_lim_y=SPE_lim_y
        self.Rsquared=np.array([Rx.T,Ry.T])*100
        self.covered_var=np.array([Rx, Ry]).T * 100
        self.x_scaling=np.vstack((Cx, Sx))
        self.y_scaling=np.vstack((Cy, Sy))
        self.Xtrain_normal=X_orining
        self.Ytrain_normal=Y_orining
        self.Xtrain_scaled=X_0
        self.Ytrain_scaled=Y_0
        self.alpha=alpha
        self.Null_Space=Null_Space
        self.Num_com=Num_com

        return self

    def evaluation(self,X_new):
        """
        receive pls model and new observation and calculate its
        y_pre,T_score,Hotelin_T2,SPE_X,SPE_Y
        """
        #if X_new.ndim==1:
        #      X_new=X_new.reshape(1,X_new.size)  
        y_pre,T_score=self.Y_fit_Calculation(X_new)
        X_new_scaled,Y_new_scaled=self.scaler(X_new,y_pre)
        
        Hotelin_T2=np.sum((T_score/np.std(self.T,axis=0,ddof=1))**2,axis=1)
        SPE_X,_,_ = self.SPE_calculation(T_score, self.P, X_new_scaled, self.alpha)
        SPE_Y,_,_ = self.SPE_calculation(T_score, self.Q, Y_new_scaled, self.alpha)

        return y_pre,T_score,Hotelin_T2,SPE_X,SPE_Y


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

    def Y_fit_Calculation(self, X_new):
        x_new_scaled,_ = self.scaler(X_new,0)
        y_fit_scaled = x_new_scaled @ self.B_pls
        T_score=x_new_scaled @ self.Wstar
        _,y_fit = self.unscaler(0,y_fit_scaled)
        return y_fit,T_score

    def scaler(self,X_new,Y_new):

        Cx=self.x_scaling[0,:]
        Sx=self.x_scaling[1,:]
        X_new=(X_new-Cx)/Sx
        #if not Y_new==0:
        Cy=self.y_scaling[0,:]
        Sy=self.y_scaling[1,:]
        Y_new=(Y_new-Cy)/Sy
        return X_new,Y_new
    
    def unscaler(self,X_new,Y_new):
        Cx=self.x_scaling[0,:]
        Sx=self.x_scaling[1,:]
        X_new=(X_new * Sx) + Cx
        #if not Y_new==0:
        Cy=self.y_scaling[0,:]
        Sy=self.y_scaling[1,:]
        Y_new=(Y_new * Sy) + Cy
        return X_new,Y_new


    def visual_plot(self, score_axis=None, X_test=None, data_labeling=False, testing_labeling=False):

        # inner Functions
        def confidenceline(r1, r2, center):
            t = np.linspace(0, 2 * np.pi, 100)  # Increase the number of points for a smoother ellipse
            x = center[0] + r1 * np.cos(t)
            y = center[1] + r2 * np.sin(t)
            return x, y
        
        def inner_ploter(y_data,position,legend_str,X_test=None,y_data_add=None,legend_str2=None):       
            X_data = np.arange(1, len(y_data) + 1)
            fig.add_trace(go.Scatter(x=X_data, y=y_data, mode='markers', marker=dict(color='blue', size=10), name=legend_str,showlegend=True),
                    row=position[0], col=position[1])
            if X_test is not None:
                y_data = np.concatenate((y_data, y_data_add))
                X_data = np.arange(1, len(y_data) + 1)
                fig.add_trace(go.Scatter(x=X_data[Num_obs:], y=y_data[Num_obs:], mode='markers', marker=dict(color='red', symbol='star', size=12), name=legend_str2,showlegend=True),
                        row=position[0], col=position[1])
            fig.add_trace(go.Scatter(x=[1, X_data[-1] + 1], y=[self.T2_lim[-1]] * 2, mode='lines', line=dict(color='black', dash='dash'), name='Hoteling T^2 Lim',showlegend=False),
                    row=position[0], col=position[1])
            fig.update_xaxes(
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
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'PLS Score Plot Distribution','SPE_X', 'SPE_Y Plot','Hoteling T^2 Plot'),
            specs=[[{"colspan": 2}, None],   # Row 1: Plot 1 spans columns 1 and 2
                [{}, {}],                        # Row 2: Normal 2-column layout
                [{"colspan":2}, {}]]             # Row 3: Normal 2-column layout
        ,row_heights=[0.5, 0.25, 0.25],
        )
        # axis labeling
        fig.update_xaxes(title_text='T '+str(score_axis[0])+'score',row=1,col=1)
        fig.update_yaxes(title_text='T '+str(score_axis[1])+'score',row=1,col=1)
        fig.update_xaxes(title_text='Observations',row=3,col=1)
        #score plot
        tscore_x = self.T[:, score_axis[0] - 1]
        tscore_y = self.T[:, score_axis[1] - 1]

        r1 = self.ellipse_radius[score_axis[0] - 1]
        r2 = self.ellipse_radius[score_axis[1] - 1]
        xr, yr = confidenceline(r1, r2, np.array([0, 0]))
        label_str = f'Confidence Limit ({self.alpha * 100}%)'

        fig.add_trace(go.Scatter(x=xr, y=yr, mode='lines', line=dict(color='black', dash='dash'), name=label_str,showlegend=True),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=tscore_x, y=tscore_y, mode='markers', marker=dict(color='blue', size=10), name='Score(Training Dataset)',showlegend=True),
                    row=1, col=1)
        
        
        if data_labeling:
            for i in range(Num_obs):
                fig.add_trace(go.Scatter(x=[tscore_x[i]], y=[tscore_y[i]], text=str(i + 1), mode='text', textposition='top center',showlegend=False),
                            row=1, col=1)
        # Testing Data
        if X_test is not None:
            Num_new = X_test.shape[0]
            _, tscore_testing, hoteling_t2_testing, spe_x_testing, spe_y_testing = self.evaluation(X_test)

            t_score_x_new = tscore_testing[:, score_axis[0] - 1]
            t_score_y_new = tscore_testing[:, score_axis[1] - 1]

            fig.add_trace(go.Scatter(x=t_score_x_new, y=t_score_y_new, mode='markers', marker=dict(color='red', symbol='star', size=12), name='Score(New Data)',showlegend=True),
                        row=1, col=1)
            if testing_labeling:
                for i in range(Num_new):
                    fig.add_trace(go.Scatter(x=[t_score_x_new[i]], y=[t_score_y_new[i]], text=str(i + 1), mode='text', textposition='top center',showlegend=False),
                                row=1, col=1)

        
        if X_test is not None:
            # SPE_X Plot
            y_data = self.SPE_x[:, -1]
            inner_ploter(y_data,[2,1],'SPE_X(Training Data)',X_test,spe_x_testing,'SPE_X(New Data)')
            # SPE_Y Plot
            y_data = self.SPE_y[:, -1]
            inner_ploter(y_data,[2,2],'SPE_Y(Training Data)',X_test,spe_y_testing,'SPE_Y(New Data)')
            # Hoteling T^2 Plot
            y_data = self.tsquared[:, -1]
            inner_ploter(y_data,[3,1],'Hoteling T2(Training Data)',X_test,hoteling_t2_testing,'Hoteling T2(New Data)')
        else:
            # SPE_X Plot
            y_data = self.SPE_x[:, -1]
            inner_ploter(y_data,[2,1],'SPE_X(Training Data)')
            # SPE_Y Plot
            y_data = self.SPE_y[:, -1]
            inner_ploter(y_data,[2,2],'SPE_Y(Training Data)')
            # Hoteling T^2 Plot
            y_data = self.tsquared[:, -1]
            inner_ploter(y_data,[3,1],'Hoteling T2(Training Data)')
    
        # Update layout for font sizes and other customization
        fig.update_layout(
        title_text='PLS Model Visual Plotting',
        title_x=0.5,
        font=dict(size=15),
        legend=dict(x=1, y=1, traceorder='normal'),
        showlegend=True,
        # Use annotations for X and Y labels for the entire figure
        )
        fig.show()

