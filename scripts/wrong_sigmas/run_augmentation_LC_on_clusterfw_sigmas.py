# -*- coding: utf-8 -*-
"""
Created on Tue May 24 08:08:44 2022

@author: maout
"""
if __name__ == '__main__':
    
    
    

    
    
    

    import numpy as np
    
    from scipy.spatial.distance import pdist,squareform
    import os
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    import gpflow as gpf
    import time
    import pickle
    import ot
    import numba
    
    
    import math
    from pyemd import emd_with_flow
    
    from core import utils, manifolds, geodesics
    from sklearn.cluster import KMeans
    #from sklearn.metrics.pairwise import pairwise_kernels
    #from numpy.linalg import pinv
    from functools import reduce
    import torch
    from torch import autograd
    from torch.autograd import grad
    import torch.functional 
    
    from scipy.spatial.distance import cdist
    import sys


    dsystem = 'LC'
    
    if dsystem == 'LC':
        def f(x, t=0,mu=1):#Van der Pol oscillator
            x0 = mu*(x[0] - (1/3)*x[0]**3-x[1])
            x1 = (1/mu)*x[0]
            return np.array([x0,x1])
        x0 = np.array([1.81, -1.41])
        dim = 2
    elif dsystem == 'Lorentz':
        rho = 28
        sigm = 10
        beta = 8/3.
        def f(x,t=0):
          return  np.array([sigm* (x[1]-x[0]) , x[0]*(rho-x[2])-x[1],x[0]*x[1] - (beta)* x[2]]) # Lorentz 63 rh0=28 :chaotic, rho=0.5 : fixed point
    
        x0 = np.array([-0,-0,5])
        dim = 3
    
    #%%
    def simulate_path(f_drift, timegrid, the_g):
        
      """
      Simulates a path on timegrid with drift `f_dtift`.
    
      Parameters
      -----------
      f_drift: function for the drift
      timegrid: ndarray, (n_times)
    
      Returns
      --------
      F : ndarray, (dim, n_times)
    
      """
      F = np.zeros((dim,timegrid.size))
      for ti,t in enumerate(timegrid):
          if ti==0:
              F[:,0] = x0
          else:
              F[:,ti] = F[:,ti-1]+ h* f_drift(F[:,ti-1].reshape(-1,1)).T+(the_g)*np.random.normal(loc = 0.0,
                                                                        scale = np.sqrt(h),
                                                                        size=(dim,))
          
      return F
    
    
    #%% 
    
    def GP_drift_estimation( X, dt, nsparse=200, lengthscales=0.1, variance=0.1, multirep=False, rep_keep=1):
        """
        Gaussian process drift estimation for observations in X (ndim, n_timegrid)
        employing `nsparse` inducing points.
    
        Parameters
        ----------
        X            : ndarray, (n_dim, n_timegrid) or (n_dim, n_reps, n_timegrid  )
                      Array with the observation of the diffusion process.
        dt           : float,
                      Temporal distance between two successive observations.
        nsparse      : int, 
                      Number of inducing point to be employed for the GP regression.
        lengthscales : float,
                      Initial estimation of lengthscales (will be optimised during the gp optimisation)
        variance     : float,
                      Initial likelihood variance 
        multirep     : bool,
                      Indicates whether observation array contains multiple realisations/repetitions
        rep_keep     : int,
                      Determines how many from the multiple realisations will be employed for the drift estimation.
    
        Returns
        ----------
    
        mf0   : gpflow model object
    
        f_est : function of drift estimate
    
    
        """
    
        if not multirep:
            ### x shape should be N x ndim 
            x = X[:,:-1].T   
            y = (np.diff(X)/dt ).T  
        elif multirep:
            x = X[:,:rep_keep,:-1].reshape(dim,-1).T   
            y = (np.diff(X[:,:rep_keep,:])/dt ).reshape(dim,-1).T  
    
        noise0 = g**2/dt
    
        
        Zi = np.random.uniform(low=[np.min(x[0]),np.min(x[1])], 
                              high=[np.max(x[0]),np.max(x[1])], 
                              size=(nsparse,dim) ) 
    
        k0 = gpf.kernels.RBF( variance=0.1, lengthscales=[0.1,0.1])
        mf0 = gpf.models.sgpr.SGPR(data=(x, y), kernel=k0,noise_variance=noise0, inducing_variable=Zi)
    
    
        gpf.utilities.print_summary(mf0)
        gpf.utilities.set_trainable(k0.variance, True)
        gpf.utilities.set_trainable(mf0.likelihood.variance, False)
    
    
        opt =gpf.optimizers.Scipy()
        opt_logs = opt.minimize(mf0.training_loss, mf0.trainable_variables, options=dict(maxiter=200))
    
    
        gpf.utilities.print_summary(mf0)
    
        
    
        return mf0
    
    
    #%%
    
    def evaluate_vector_field( f_given, etype= 'est', n=20, bounds=[[-2.5, 2.5], [-2.5, 2.5]]):
      """
      Evaluates vector field for given f on a grid extending within the `bounds` 
      consisting of `n` grid points across each dimension.
    
      Parameters
      -----------
      f      : Function to be evaluated on the grid points.
      etype   : Type of function:
                - 'pure' : user defined function (true function of the underlying dynamical system)
                - 'est'  : GPflow estimated function (returns mean and variance)
      n      : int, Number of points on the grid per dimension.
      bounds : ndarray-like with min-max bounds per dimension. (n_dims, 2)
    
      Returns
      ---------
      xt      : grid points
      mu, var : If `type` is `est`. ndarrays with evaluation of the mean and variance of the GP at each.
                grid point.
      ztrue   : If `type is `pure` ndarray with evaluation of the provided fucntion on the grid.
    
      """
    
    
      if dim ==3:
          #n=10
          #xt = np.mgrid[-25:25:complex(0,n),-20:30:complex(0,n),-0:45:complex(0,n)]
          xt = np.mgrid[bounds[0,0]:bounds[0,1]:complex(0,n),
                        bounds[1,0]:bounds[1,1]:complex(0,n),
                        bounds[2,0]:bounds[2,1]:complex(0,n)]
    
          if etype=='est':
              mu = np.zeros(xt.shape)
              var = np.zeros(xt.shape)
              for i in range(n):
                for j in range(n):
                    for ki in range(n):                      
                        mu[:,i,j,ki], var[:,i,j,ki] = f_given(xt[:,i,j,ki])
              return mu,var
          elif etype=='pure':
              ztrue = np.zeros(xt.shape)
              for i in range(n):
                for j in range(n):
                    for ki in range(n):                      
                        ztrue[:,i,j,ki] = f_given(xt[:,i,j,ki])
              return ztrue
          else: 
              print('Please select a proper type for the provided function.')
              return -1
                    
          
              
              
                    
      elif dim==2:
          #n=20
          #xt = np.mgrid[-2.5:2.5:complex(0,n),-2.5:2.5:complex(0,n)]
          xt = np.mgrid[bounds[0][0]:bounds[0][1]:complex(0,n),
                        bounds[1][0]:bounds[1][1]:complex(0,n)]
    
          mu = np.zeros(xt.shape)
          
          ztrue = np.zeros(xt.shape)
          var = np.zeros(xt.shape)
          
          if etype=='est':
              for i in range(n):
                for j in range(n):
                  mu[:,i,j], var[:,i,j] = f_given(xt[:,i,j])
              return xt, mu,var
          elif etype=='pure':   
              for i in range(n):
                #for j in range(n):
                ztrue[:,i,:] = f_given(xt[:,i,:])
              return xt, ztrue
    
          else: 
              print('Please select a proper type for the provided function.')
              return -1    

#%%
    
    def compute_Wasserstein(Fest, Ftrue):
        """
        Computes the Wasserstein distance between the two provided densities.
        For efficient computation we approximate the we compute 20 independent computations of 
        the sliced Wasserstein distance with 10**3 projections and return the mean and standart deviation
        over the 20 trials
    
        Parameters
        -----------
        Fest : ndarray
        Ftrue : ndarray
    
    
        Returns
        ---------
    
        res_mean : Mean Wasserstein distance over the 20 trials
        res_std  : Std of Wasserstein distance over the 20 trials
    
        """
        # uniform distribution on samples
        a, b = np.ones((Fest.shape[1],)) / Fest.shape[1], np.ones((Ftrue.shape[1],)) / Ftrue.shape[1]  
    
        n_seed = 20
        n_projections = 10**3
        res = np.empty((n_seed))
    
        for seed in range(n_seed):        
            res[seed] = ot.sliced_wasserstein_distance(Fest.T,Ftrue.T, a, b, n_projections, seed=seed)
    
        res_mean = np.mean(res, axis=0)
        res_std = np.std(res, axis=0)
        return res_mean, res_std          
      
    #%%
    def compute_mean_squared_error(A, B):
        mse = np.mean((A-B)**2)
        return np.sqrt(mse)
        
    
    def evaluation_for_given_drift(f_given, f_givenvar, ztrue, the_timegrid,
                                   obs_den, F_obs, bounds, g, etype='pure'):
        ## etype: 'pure' is for drift estimates without uncertainty
        ##        'est' is for drift estimates with uncertainty
        ##Evaluate vector field 
        if etype=='est':
            _, mu, var = evaluate_vector_field( f_givenvar, etype= etype, n=n_grid, bounds=bounds)
        else:
            var = 0
            _, mu = evaluate_vector_field( f_given, etype= etype, n=n_grid, bounds=bounds)
        ##Compute Mean squared errorbetween evaluated vector field and true vector field
        RMSE = compute_mean_squared_error(ztrue, mu)   
        was_mean = np.zeros(2)
        was_std = np.zeros(2)
        for i in range(2): #simulate 10 paths - return only one full and statistics for all
            ##Simulate a full continous path with estimated drift
            F_sim = simulate_path(f_given, the_timegrid, g)
            ##Compute Wasserstein distance between simulated path (after subsampling) and Observations
            was_mean[i], was_std[i] = compute_Wasserstein(F_sim[:,::obs_den], F_obs)
        return mu, var, RMSE, F_sim, was_mean, was_std
    
    #%%
        
    
    def score_function_multid_seperate_all_dims(X,Z,func_out=False, C=0.001,kern ='RBF',l=1,which=1):
        """
        returns function psi(z)
        Input: X: N observations
               Z: sparse points
               func_out : Boolean, True returns function if False return grad-log-p on data points                    
               l: lengthscale of rbf kernel
               C: weighting constant           
               which: return 1: grad log p(x) 
               
        Output: psi: array with density along the given dimension N or N_s x 1
        
        """
        
        if kern=='RBF':
            #l = 1 # lengthscale of RBF kernel
            #@numba.njit(parallel=True,fastmath=True)
            def Knumba(x,y,l,res,multil=False): #version of kernel in the numba form when the call already includes the output matrix
                if multil:         
                    #print('here')
                    #res = np.ones((x.shape[0],y.shape[0]))                
                    for ii in range(len(l)): 
                        tempi = np.zeros((x[:,ii].size, y[:,ii].size ), dtype=np.float64)
                        ##puts into tempi the cdist result
                        #print(x[:,ii:ii+1].shape)
                        my_cdist(x[:,ii:ii+1], y[:,ii:ii+1],tempi,'sqeuclidean')
                        
                        res = np.multiply(res,np.exp(-tempi/(2*l[ii]*l[ii])))                    
                        ##res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
                    #return res
                else:
                    tempi = np.zeros((x.shape[0], y.shape[0] ), dtype=np.float64)
                    #return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
                    my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
                    res = np.exp(-tempi/(2*l*l))
                return 0
            
            def K(x,y,l,multil=False):
                if multil:         
                    #print('here')
                    res = np.ones((x.shape[0],y.shape[0]))                
                    for ii in range(len(l)): 
                        tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
                        ##puts into tempi the cdist result
                        my_cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),tempi,'sqeuclidean')
                        res = np.multiply(res,np.exp(-tempi/(2*l[ii]*l[ii])))                    
                        ##res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
                    return res
                else:
                    tempi = np.zeros((x.shape[0], y.shape[0] ))
                    #return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
                    my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
                    return np.exp(-tempi/(2*l*l))
                #return np.exp(-(x-y.T)**2/(2*l*l))
                #return np.exp(np.linalg.norm(x-y.T, 2)**2)/(2*l*l) 
            #@njit
            def grdx_K_all(x,y,l,multil=False): #gradient with respect to the 1st argument - only which_dim
                N,dim = x.shape    
                M,_ = y.shape
                diffs = x[:,None]-y                         
                redifs = np.zeros((1*N,M,dim))
                for ii in range(dim):          
                
                    if multil:
                        redifs[:,:,ii] = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])   
                    else:
                        redifs[:,:,ii] = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
                return redifs
                #return -(1./(l*l))*(x-y.T)*K(x,y)
            
            def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
                N,dim = x.shape 
                M,_ = y.shape
                diffs = x[:,None]-y                         
                redifs = np.zeros((1*N,M))
                ii = which_dim -1
                #print('diffs:',diffs)
                if multil:
                    redifs = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])  
                    #print(redifs.shape)
                else:
                    redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
                return redifs
         
            
                #############################################################################
        elif kern=='periodic': ###############################################################################################
          ###periodic kernel
            ## K(x,y) = exp(  -2 * sin^2( pi*| x-y  |/ (2*pi)  )   /l^2)
            
            ## Kx(x,y) = (K(x,y)* (x - y) cos(abs(x - y)/2) sin(abs(x - y)/2))/(l^2 abs(x - y))
            ## -(2 K(x,y) π (x - y) sin((2 π abs(x - y))/per))/(l^2 s abs(x - y))
          #per = 2*np.pi ##period of the kernel
          #l = 0.5
          def K(x,y,l,multil=False):
            
            if multil:          
              #print('here')
              res = np.ones((x.shape[0],y.shape[0]))                
              for ii in range(len(l)): 
                  #tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
                  ##puts into tempi the cdist result
                  #my_cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),tempi, 'l1')              
                  #res = np.multiply(res, np.exp(- 2* (np.sin(tempi/ 2 )**2) /(l[ii]*l[ii])) )
                  res = np.multiply(res, np.exp(- 2* (np.sin(cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'minkowski', p=1)/ 2 )**2) /(l[ii]*l[ii])) )
              return -res
            else:
                #tempi = np.zeros((x.shape[0], y.shape[0] ))
                ##puts into tempi the cdist result
                #my_cdist(x, y, tempi,'l1')
                #res = np.exp(-2* ( np.sin( tempi / 2 )**2 ) /(l*l) )
                res = np.exp(-2* ( np.sin( cdist(x, y,'minkowski', p=1) / 2 )**2 ) /(l*l) )
                return res
            
          def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
              N,dim = x.shape            
              diffs = x[:,None]-y   
              #print('diffs:',diffs)
              redifs = np.zeros((1*N,N))
              ii = which_dim -1
              #print(ii)
              if multil:
                  redifs = np.divide( np.multiply( np.multiply( np.multiply( -2*K(x,y,l,True),diffs[:,:,ii] ),np.sin( np.abs(diffs[:,:,ii]) / 2) ) ,np.cos( np.abs(diffs[:,:,ii])  / 2) ) , (l[ii]*l[ii]* np.abs(diffs[:,:,ii]))  ) 
              else:
                  redifs = np.divide( np.multiply( np.multiply( np.multiply( -2*diffs[:,:,ii],np.sin( np.abs(diffs[:,:,ii]) / 2) ) ,K(x,y,l) ),np.cos( np.abs(diffs[:,:,ii]) / 2) ) ,(l*l* np.abs(diffs[:,:,ii])) )           
              return -redifs
    
        dim = X.shape[1]
    
        if isinstance(l, (list, tuple, np.ndarray)):
           multil = True
           ### for different lengthscales for each dimension 
           #K_xz =  np.ones((X.shape[0],Z.shape[0]), dtype=np.float64) 
           #Knumba(X,Z,l,K_xz,multil=True) 
           K_xz = K(X,Z,l,multil=True) 
           #Ks =  np.ones((Z.shape[0],Z.shape[0]), dtype=np.float64) 
           #Knumba(Z,Z,l,Ks,multil=True) 
           Ks = K(Z,Z,l,multil=True)    
           
           #print(Z.shape)
           Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
           A = K_xz.T @ K_xz    
                  
           gradx_K = -grdx_K_all(X,Z,l,multil=True) #-
           gradxK = np.zeros((X.shape[0],Z.shape[0],dim))
           for ii in range(dim):
               gradxK[:,:,ii] = -grdx_K(X,Z,l,multil=True,which_dim=ii+1)
           # if not(Test_p == 'None'):
           #     K_sz = K(Test_p,Z,l,multil=True)
           np.testing.assert_allclose(gradxK, gradx_K) 
        else:
            multil = False
            
            K_xz = K(X,Z,l,multil=False) 
            
            Ks = K(Z,Z,l,multil=False)    
            
            Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
            A = K_xz.T @ K_xz    
            
            gradx_K = -grdx_K_all(X,Z,l,multil=False)   #shape: (N,M,dim)
        sumgradx_K = np.sum(gradx_K ,axis=0) ##last axis will have the gradient for each dimension ### shape (M, dim)
        #print( sumgradx_K.shape )
        if func_out==False: #if output wanted is evaluation at data points
            
            # res1 = np.zeros((N, dim))    
            # ### evaluatiion at data points
            # for di in range(dim):
            #     res1[:,di] = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K[:,di]
            
            
            res1 = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K
            
            
            #res1 = np.einsum('ik,kj->ij', -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv, sumgradx_K)
            
            
        else:           
            #### for function output 
            if multil:                
                #res = np.ones((x.shape[0],y.shape[0]))                
                #for ii in range(len(l)): 
                if kern=='RBF':      
                    K_sz = lambda x: reduce(np.multiply, [ np.exp(-cdist(x[:,iii].reshape(-1,1), Z[:,iii].reshape(-1,1),'sqeuclidean')/(2*l[iii]*l[iii])) for iii in range(x.shape[1]) ])
            
                    
                elif kern=='periodic':
                    K_sz = lambda x: np.multiply(np.exp(-2*(np.sin( cdist(x[:,0].reshape(-1,1), Z[:,0].reshape(-1,1), 'minkowski', p=2)/(l[0]*l[0])))),np.exp(-2*(np.sin( cdist(x[:,1].reshape(-1,1), Z[:,1].reshape(-1,1),'sqeuclidean')/(l[1]*l[1])))))
                
                
                #return K_sz
            else:
                if kern=='RBF':
                    K_sz = lambda x: np.exp(-cdist(x, Z,'sqeuclidean')/(2*l*l))
                elif kern=='periodic':
                    K_sz = lambda x: np.exp(-2* ( np.sin( cdist(x, Z,'minkowski', p=1) / 2 )**2 ) /(l*l) )
                #return K_sz
    
            res1 = lambda x: K_sz(x) @ ( -np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0])) ) @ Ksinv@sumgradx_K
            # res1 = np.zeros((N, dim))
            # for di in range(dim):
            #     res1 = lambda x: K_sz(x) @ ( -np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0])) ) @ Ksinv@sumgradx_K[:,di]
            
                
            #np.testing.assert_allclose(res2, res1)
        
        return res1   ### shape out N x dim
    
    def my_cdist(r,y, output,dist='euclidean'):
        """   
        Fast computation of pairwise distances between data points in r and y matrices.
        Stores the distances in the output array.
        Available distances: 'euclidean' and 'seucledian'
        Parameters
        ----------
        r : NxM array
            First set of N points of dimension M.
        y : N2xM array
            Second set of N2 points of dimension M.
        output : NxN2 array
            Placeholder for storing the output of the computed distances.
        dist : type of distance, optional
            Select 'euclidian' or 'sqeuclidian' for Euclidian or squared Euclidian
            distances. The default is 'euclidean'.
    
        Returns
        -------
        None. (The result is stored in place in the input array output).
    
        """
        N, M = r.shape
        N2, M2 = y.shape
        #assert( M == M2, 'The two inpus have different second dimention! Input should be N1xM and N2xM')
        if dist == 'euclidean':
            for i in numba.prange(N):
                for j in numba.prange(N2):
                    tmp = 0.0
                    for k in range(M):
                        tmp += (r[i, k] - y[j, k])**2            
                    output[i,j] = math.sqrt(tmp)
        elif dist == 'sqeuclidean':
            for i in numba.prange(N):
                for j in numba.prange(N2):
                    tmp = 0.0
                    for k in range(M):
                        tmp += (r[i, k] - y[j, k])**2            
                    output[i,j] = tmp   
        elif dist == 'l1':
            for i in numba.prange(N):
                for j in numba.prange(N2):
                    tmp = 0.0
                    for k in range(M):
                        tmp += (r[i, k] - y[j, k])**2          
                    output[i,j] = math.sqrt(tmp)   
        return 0
    
    #%%
    
    
    def score_function_multid_seperate(X,Z,func_out=False, C=0.001,kern ='RBF',l=1,which=1,which_dim=1):
        
        """
        returns function psi(z)
        Input: X: N observations
               Z: sparse points
               func_out : Boolean, True returns function if False return grad-log-p on data points                    
               l: lengthscale of rbf kernel
               C: weighting constant           
               which: return 1: grad log p(x) 
               which_dim: which gradient of log density we want to compute (starts from 1 for the 0-th dimension)
        Output: psi: array with density along the given dimension N or N_s x 1
        
        """
        if kern=='RBF':
            #l = 1 # lengthscale of RBF kernel
            
            def K(x,y,l,multil=False):
                if multil:                
                    res = np.ones((x.shape[0],y.shape[0]))                
                    for ii in range(len(l)): 
                        res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
                    return res
                else:
                    return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
                #return np.exp(-(x-y.T)**2/(2*l*l))
                #return np.exp(np.linalg.norm(x-y.T, 2)**2)/(2*l*l) 
            
            def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
                N,dim = x.shape            
                diffs = x[:,None]-y   
                #print(diffs.shape)
                redifs = np.zeros((1*N,N))
                ii = which_dim -1
                #print(ii)
                if multil:
                    redifs = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])   
                else:
                    redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
                return redifs
                #return -(1./(l*l))*(x-y.T)*K(x,y)
         
            def grdy_K(x,y): # gradient with respect to the second argument
                N,dim = x.shape
                diffs = x[:,None]-y            
                redifs = np.zeros((N,N))
                ii = which_dim -1              
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)         
                return -redifs
                #return (1./(l*l))*(x-y.T)*K(x,y)
                    
            def ggrdxy_K(x,y):
                N,dim = Z.shape
                diffs = x[:,None]-y            
                redifs = np.zeros((N,N))
                for ii in range(which_dim-1,which_dim):  
                    for jj in range(which_dim-1,which_dim):
                        redifs[ii, jj ] = np.multiply(np.multiply(diffs[:,:,ii],diffs[:,:,jj])+(l*l)*(ii==jj),K(x,y))/(l**4) 
                return -redifs
                #return np.multiply((K(x,y)),(np.power(x[:,None]-y,2)-l**2))/l**4
         
        if isinstance(l, (list, tuple, np.ndarray)):
           ### for different lengthscales for each dimension 
           K_xz = K(X,Z,l,multil=True) 
           Ks = K(Z,Z,l,multil=True)    
           multil = True
           #print(Z.shape)
           Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
           A = K_xz.T @ K_xz           
           gradx_K = -grdx_K(X,Z,l,which_dim=which_dim,multil=True) 
           # if not(Test_p == 'None'):
           #     K_sz = K(Test_p,Z,l,multil=True)
            
        else:
            multil = False
            K_xz = K(X,Z,l,multil=False) 
            Ks = K(Z,Z,l,multil=False)    
            
            Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
            A = K_xz.T @ K_xz    
            gradx_K = -grdx_K(X,Z,l,which_dim=which_dim,multil=False)
        sumgradx_K = np.sum(gradx_K ,axis=0)
        if func_out==False: #if output wanted is evaluation at data points
            ### evaluatiion at data points
            res1 = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K
        else:           
            #### for function output 
            if multil:                
                #res = np.ones((x.shape[0],y.shape[0]))                
                #for ii in range(len(l)): 
                K_sz = lambda x: np.multiply(np.exp(-cdist(x[:,0].reshape(-1,1), Z[:,0].reshape(-1,1),'sqeuclidean')/(2*l[0]*l[0])),np.exp(-cdist(x[:,1].reshape(-1,1), Z[:,1].reshape(-1,1),'sqeuclidean')/(2*l[1]*l[1])))
                #return K_sz
            else:
                K_sz = lambda x: np.exp(-cdist(x, Z,'sqeuclidean')/(2*l*l))
                #return K_sz
    
            res1 = lambda x: K_sz(x) @ ( -np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0])) ) @ Ksinv@sumgradx_K
    
    
        
        return res1
        
    #%%
        
    
    def reweight_optimal_transport_multidim(samples, weights):
    
        """
        Computes deterministic transport map for particle reweighting.
        Particle state is multidimensional.
    
        Parameters
        ------------
            samples: array-like,
                Samples from distribution M x dim , with dim>=2.
            weights: array-like,
                weights for each sample M.
    
        Returns
        --------
            T: array like,
                transport map.
    
        Reweighting particles according to ensemble transform particle filter
        (ETPF) algorithm proposed by `Reich 2013`.
        Instead of particle resampling, ETPF employes Optimal Transport to 
        compute a deterministic particle shift which minimises the
        expected distances between the particles before and after the transformation.
        :math: `CO = X^T \\cdot X`
        :math: `CO = diag(CO)*ones(1,M) -2*CO + ones(M,1)*diag(CO)'`
        :math: `[dist,T] = emd(ww,ones(M,1)/M,CO,-1,3)`
        :math: `T = T \\cdot M`
    
        """
    
        num_samples = samples.shape[0] ## this should be the number of points
    
        covar = squareform(pdist(samples, 'euclidean'))
        b = np.ones((num_samples, 1)) / num_samples  # uniform distribution on samples
    
        _, T = emd_with_flow(weights.reshape(-1, ), b.reshape(-1, ), covar, -1)
    
        T = np.array(T)*num_samples
    
    
        return T    #%%
    
    #%%
    class BRIDGE_ND_reweight:
        def __init__(self,t1,t2,y1,y2,f,g,N,M,reweight=False, U=None,dens_est='nonparametric',
                     reject=True,plotting=True,kern='RBF',f_true=None,
                     brown_bridge=False,  uncertend=0, sample='deterministic',dt=0.01, 
                     fwd_sample='deterministic',control_deterministic=False):
            """
            Bridge initialising function
            t1: starting time point
            t2: end time point
            y1: initial observation/position
            y2: end observation/position
            f: drift function handler
            g: diffusion coefficient or function handler 
            N: number of particles/trajectories
            
            M: number of sparse points for grad log density estimation
            reweight: boolean - determines if reweighting will follow
            U: function, reweighting function to be employed during reweighting: dim_y1 \to 1
            dens_est: density estimation function
                      > 'nonparametric' : non parametric density estimation
                      > 'hermit1' : parametic density estimation empoying hermite polynomials (physiscist's)
                      > 'hermit2' : parametic density estimation empoying hermite polynomials (probabilists's)
                      > 'poly' : parametic density estimation empoying simple polynomials
                      > 'rbf' : parametric density estimation employing radial basis functions
            kern: type of kernel: 'RBF' or 'periodic'
            reject: boolean parameter indicating whether non valid bridge trajectories will be rejected
            plotting: boolean parameter indicating whether bridge statistics will be plotted
            f_true: in case of Brownian bridge reweighting this is the true forward drift for simulating the forward dynaics
            brown_bridge: boolean,determines if the reweighting concearns contstraint or reweighting with respect to brownian bridge
            uncertend: determines condition for terminal point: 0: delta function, 
                                                                >0: std of a normal density centered around y2
                                                                -1: uses last step of forward pass as initial condition
                                                                -2: uses the covariance of Z[:,:,-1] but around y2
                                                                -3: uses single point (the mean) of the last step of forward pass (this is to have good predictions in subsequent estimations)
            """
            self.dim = y1.size # dimensionality of the problem
            self.t1 = t1
            self.t2 = t2
            self.y1 = y1
            self.y2 = y2
    
            
            ##density estimation stuff
            self.kern = kern
            # DRIFT /DIFFUSION
            self.f = lambda x,t=0: f(x, t=0)
            self.g = g #scalar or array
            
            ### PARTICLES DISCRETISATION
            self.N = N        
            
            self.N_sparse = M
            self.sample = sample
            self.fwd_sample = fwd_sample
            
            self.dt = 0.01 #((t2-t1)/k)
            ### reject
            self.reject = reject
            ## pointer to set to False when returned bridge is incorrect
            self.valid = True
            self.delete = False ##id true deletes invalid trajectories if reject is true
            ###else it replaces the invalids with stochastic ones
            self.control_deterministic = control_deterministic
            self.finer = 1#200 #discetasation ratio between numerical BW solution and particle bridge solution
            self.timegrid = np.arange(self.t1,self.t2+self.dt/2,self.dt)
            self.k = self.timegrid.size
            ### reweighting
            self.brown_bridge = brown_bridge
            self.reweight = reweight
            if self.reweight:
              self.U = U
              if self.brown_bridge:
                  self.Ztr = np.zeros((self.dim,self.N,self.k)) #storage for forward trajectories with true drift
                  self.f_true = f_true
            ##determines std of a gaussian density around the terminal point
            ## if it is -1 it takes the std of the final step of the forward density
            self.uncertend = uncertend
            #self.timegrid_fine = np.arange(self.t1, self.t2+self.dt*(1./self.finer)/2, self.dt*(1./self.finer) )
            
            # print(self.k == self.timegrid.size)
            # print(self.timegrid)
            
            self.Z = np.zeros((self.dim,self.N,self.k)) #storage for forward trajectories
            self.B = np.zeros((self.dim,self.N,self.k)) #storage for backward trajectories
            self.ln_roD = [] 
            self.BPWE = np.zeros((self.dim,self.N,self.timegrid.size))
            self.BPWEmean = np.zeros((self.dim,self.k*self.finer))
            self.BPWEstd = np.zeros((self.dim,self.k*self.finer))
            self.BPWEskew = np.zeros((self.dim,self.k*self.finer))
            self.BPWEkurt = np.zeros((self.dim,self.k*self.finer))
    
            self.gbB = np.zeros((self.dim,self.N,self.k))
            
            self.gfFc = np.zeros((self.dim,self.N,self.k))                  
            self.Fc =np.zeros((self.dim,self.N,self.k)) 
            
            
            #self.forward_sampling()
            self.forward_sampling_Otto()
            if self.reweight and self.brown_bridge:
                self.forward_sampling_Otto_true()
    
               
            #self.density_estimation()
            self.backward_simulation()
            self.reject_trajectories() 
            self.compute_controlled_flow()
            #self.calculate_true_statistics()
            #if plotting:
            #    self.plot_statistics()
            
        def forward_sampling(self): 
            print('Sampling forward...')
            W = np.ones((self.N,1))/self.N
            for ti,tt in enumerate(self.timegrid):
    
                if ti == 0:
                    self.Z[0,:,0] = self.y1[0]
                    self.Z[1,:,0] = self.y1[1]
                else:
                    for i in range(self.N):
                        #self.Z[:,i,:] = sdeint.itoint(self.f, self.g, self.Z[i,0], self.timegrid)[:,0] 
                        self.Z[:,i,ti] = ( self.Z[:,i,ti-1] + self.dt* self.f(self.Z[:,i,ti-1], tt) + \
                                          (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,)) )
            
                    ###WEIGHT
                    if self.reweight == True:
                      if ti>0:
                          W[:,0] = np.exp(self.U(self.Z[:,:,ti]))                    
                          W = W/np.sum(W)
                          
                          ###REWEIGHT                    
                          Tstar = reweight_optimal_transport_multidim(self.Z[:,:,ti].T,W)
                          #P = Tstar *N
                          # print(Tstar.shape)
                          # print(X.shape)
                          self.Z[:,:,ti] = (  (self.Z[:,:,ti])@Tstar  )
                    
            #for di in range(self.dim):
              #self.Z[di,:,-1] = self.y2[di]
            print('Forward sampling done!')
            return 0
        
        
        
        
        ### effective forward drift - estimated seperatelly for each dimension
        def f_seperate_true(self,x,t):#plain GP prior
            
            dimi, N = x.shape        
            bnds = np.zeros((dimi,2))
            for ii in range(dimi):
                bnds[ii] = [np.min(x[ii,:]),np.max(x[ii,:])]
            #sum_bnds = np.sum(bnds)        
    
            Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )        
            gpsi = np.zeros((dimi, N))
            lnthsc = 2*np.std(x,axis=1)   
               
            for ii in range(dimi):            
                gpsi[ii,:]= score_function_multid_seperate(x.T,Sxx.T,False,C=0.001,which=1,l=lnthsc,which_dim=ii+1, kern=self.kern)     
            
            return (self.f_true(x,t)-0.5* self.g**2* gpsi)
        
        
        ### effective forward drift - estimated seperatelly for each dimension
        def f_seperate(self,x,t):#plain GP prior
            
            dimi, N = x.shape        
            bnds = np.zeros((dimi,2))
            for ii in range(dimi):
                bnds[ii] = [np.min(x[ii,:]),np.max(x[ii,:])]
            #sum_bnds = np.sum(bnds)
            
    
            Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )        
            gpsi = np.zeros((dimi, N))
            lnthsc = 2*np.std(x,axis=1)    
               
            for ii in range(dimi):            
                gpsi[ii,:]= score_function_multid_seperate(x.T,Sxx.T,False,C=0.001,which=1,l=lnthsc,which_dim=ii+1, kern=self.kern)     
            
            return (self.f(x,t)-0.5* self.g**2* gpsi)
        
         ###same as forward sampling but without reweighting - this is for bridge reweighting
            ### not for constraint reweighting    
        def forward_sampling_Otto_true(self):
            print('Sampling forward with deterministic particles and true drift...')
            #W = np.ones((self.N,1))/self.N
            for ti,tt in enumerate(self.timegrid):  
                #print(ti)          
                if ti == 0:
                    for di in range(self.dim):
                        self.Ztr[di,:,0] = self.y1[di]
                        #self.Z[di,:,-1] = self.y2[di]   
                        #self.Z[di,:,0] = np.random.normal(self.y1[di], 0.05, self.N)
                elif ti==1: #propagate one step with stochastic to avoid the delta function
                    #for i in range(self.N):                            #substract dt because I want the time at t-1
                    self.Ztr[:,:,ti] = (self.Ztr[:,:,ti-1] + self.dt*self.f_true(self.Ztr[:,:,ti-1],tt-self.dt)+\
                                     (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,self.N)) )
                else:                
                    self.Ztr[:,:,ti] = ( self.Ztr[:,:,ti-1] + self.dt* self.f_seperate_true(self.Ztr[:,:,ti-1],tt-self.dt) )                
                      
            print('Forward sampling with Otto true is ready!')        
            return 0
        
        
        
        def forward_sampling_Otto(self):
            print('Sampling forward with deterministic particles...')
            W = np.ones((self.N,1))/self.N
            for ti,tt in enumerate(self.timegrid):  
                #print(ti)          
                if ti == 0:
                    for di in range(self.dim):
                        self.Z[di,:,0] = self.y1[di]
                        if self.brown_bridge:
                            self.Z[di,:,-1] = self.y2[di]   
                        #self.Z[di,:,0] = np.random.normal(self.y1[di], 0.05, self.N)
                elif (ti>=1 and self.sample=='stochastic') or (ti==1 and self.sample=='deterministic'): #propagate one step with stochastic to avoid the delta function
                    #for i in range(self.N):                            #substract dt because I want the time at t-1
                    self.Z[:,:,ti] = (self.Z[:,:,ti-1] + self.dt*self.f(self.Z[:,:,ti-1],tt-self.dt)+\
                                     (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,self.N)) )
                else:                
                    self.Z[:,:,ti] = ( self.Z[:,:,ti-1] + self.dt* self.f_seperate(self.Z[:,:,ti-1],tt-self.dt) )
                    ###WEIGHT
                if self.reweight == True:
                  if ti>0:
                      #print(self.U(self.Z[:,:,ti]))
                      W[:,0] = np.exp(self.U(self.Z[:,:,ti],ti) ) #-1                   
                      W = W/np.sum(W)       
                      
                      ###REWEIGHT  full optimal transport  
                      start = time.time()
                      Tstar = reweight_optimal_transport_multidim(self.Z[:,:,ti].T,W)
                      self.Z[:,:,ti] = ((self.Z[:,:,ti])@Tstar ) ##### 
                      ####Reweight fast optimal transport
                      #M = ot.dist(self.Z[:,:,ti].T, self.Z[:,:,ti].T)
                      #M /= M.max()
                      #a = W[:,0]
                      #b =  np.ones_like(W[:,0])/self.N
                      #T2 = ot.emd(a, b, M)
                      #self.Z[:,:,ti] = (self.N*self.Z[:,:,ti]@T2)
                      
                      if ti ==3:
                          stop = time.time()
                          print('Timepoint: %d needed '%ti, stop-start)  
                      ###      
            print('Forward sampling with Otto is ready!')        
            return 
        
        def density_estimation(self, ti,rev_ti):
            rev_t = rev_ti-1#########################################################-1
            grad_ln_ro = np.zeros((self.dim,self.N))
            lnthsc = 2*np.std(self.Z[:,:,rev_t],axis=1)
            
            bnds = np.zeros((self.dim,2))
            for ii in range(self.dim):
                bnds[ii] = [min(np.min(self.Z[ii,:,rev_t]),np.min(self.B[ii,:,rev_ti])),max(np.max(self.Z[ii,:,rev_t]),np.max(self.B[ii,:,rev_ti]))]
            #sum_bnds = np.sum(bnds)
            
            
            #sparse points
            Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )
            
            for di in range(self.dim):     
                #estimate density from forward (Z) and evaluate at current postitions of backward particles (B)       
                grad_ln_ro[di,:] = score_function_multid_seperate(self.Z[:,:,rev_t].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)(self.B[:,:,rev_ti].T)
                         
            
            return grad_ln_ro 
    
    
        def bw_density_estimation(self, ti, rev_ti):
            grad_ln_b = np.zeros((self.dim,self.N))
            lnthsc = 2*np.std(self.B[:,:,rev_ti],axis=1)
            #print(ti, rev_ti, rev_ti-1)
            bnds = np.zeros((self.dim,2))
            for ii in range(self.dim):
                bnds[ii] = [max(np.min(self.Z[ii,:,rev_ti]),np.min(self.B[ii,:,rev_ti])),min(np.max(self.Z[ii,:,rev_ti]),np.max(self.B[ii,:,rev_ti]))]
            #sparse points
            #print(bnds)
            #sum_bnds = np.sum(bnds)
            
            Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )
            
            for di in range(self.dim):            
                grad_ln_b[di,:] = score_function_multid_seperate(self.B[:,:,rev_ti].T,Sxx.T,func_out= False,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)#(self.B[:,:,-ti].T)
            
            return grad_ln_b # this should be function
        
        
        def backward_simulation_nonused(self):   
            
            for ti,tt in enumerate(self.timegrid[:-1]): 
                #W = np.ones((N,1))/N           
                if ti==0:                
                    for di in range(self.dim):
                        self.B[di,:,-1] = self.y2[di]                
                else:
                    
                    Ti = self.timegrid.size
                    rev_ti = Ti-ti#self.k -ti-1#Ti- ti     
                    
                    grad_ln_ro = self.density_estimation(ti,rev_ti+1) #density estimation of forward particles  
                    
                    if ti==1: 
                      #print(rev_ti,rev_ti-1)
                      self.B[:,:,rev_ti] = (self.B[:,:,rev_ti+1] - self.f(self.B[:,:,rev_ti+1], self.timegrid[rev_ti])*self.dt + self.dt*self.g**2*grad_ln_ro \
                                             + (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,self.N)) )
                    else:
                      grad_ln_b = self.bw_density_estimation(ti,rev_ti)
                      self.B[:,:,rev_ti] = (self.B[:,:,rev_ti+1] -\
                                            ( self.f(self.B[:,:,rev_ti+1], self.timegrid[rev_ti])- self.g**2*grad_ln_ro +0.5*self.g**2 * grad_ln_b )*self.dt)
                    
            for di in range(self.dim):
                self.B[di,:,0] = self.y1[di]
                
            return 0 
    
        def backward_simulation(self):
            print('Sampling backward with deterministic particles...')
            for ti,tt in enumerate(self.timegrid[:-1]):
                #print(ti)
                if ti==0:
                    if self.uncertend ==0:
                        for di in range(self.dim):
                            self.B[di,:,-1] = self.y2[di]
                    elif self.uncertend ==-1:
                        self.B[:,:,-1] = self.Z[:,:,-1]
                    elif self.uncertend ==-3:
                        for di in range(self.dim):
                            self.B[di,:,-1] = np.mean(self.Z[di,:,-1])
                    elif self.uncertend ==-2:
                        self.B[:,:,-1] = np.random.multivariate_normal(self.y2, 
                                                                        np.cov(self.Z[:,:,-1]), 
                                                                        self.N).T
                    else:                    
                        self.B[:,:,-1] = np.random.multivariate_normal(self.y2, 
                                                                        np.eye(self.dim)*self.uncertend, 
                                                                        self.N).T
                        
    
                    
                else:
                    rev_ti = self.k -ti-1#np.where(self.timegrid==tt)[0][0]  
                    #print(rev_ti)
                    
                    grad_ln_ro = self.density_estimation(ti,rev_ti+1) #this estimates grad log rho on Bs
                    
                    self.gbB[:,:,rev_ti+1] =  -self.f(self.B[:,:,rev_ti+1], self.timegrid[rev_ti]) + self.g**2*grad_ln_ro   
                    if (ti>=1 and self.sample=='stochastic') or (ti==1 and self.sample=='deterministic'):#False:#callable(self.grad_log_pss):
                      #for i in range(self.N):  
                                   
                      self.B[:,:,rev_ti] = self.B[:,:,rev_ti+1] - self.f(self.B[:,:,rev_ti+1], self.timegrid[rev_ti])*self.dt + \
                      self.dt*self.g**2*grad_ln_ro+ \
                      (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,self.N))
                      #0.5*self.dt*self.g**2*self.grad_log_pss( self.B[:,i,rev_ti+1] ,self.g ) ###this is the gradient of the stationary p
                          
                           
                    else:
                      grad_log_q = self.bw_density_estimation( rev_ti+1, rev_ti+1)
                      #for i in range(self.N):                       
                      self.B[:,:,rev_ti] = self.B[:,:,rev_ti+1] - self.f(self.B[:,:,rev_ti+1], self.timegrid[rev_ti])*self.dt - \
                      0.5*self.dt*self.g**2*grad_log_q + self.dt*self.g**2*grad_ln_ro
                           
    
            for di in range(self.dim):
                self.B[di,:,0] = self.y1[di]
                
            return 0
    
        def reject_trajectories(self):      
          std_changes_before = np.std(np.abs(self.B[:, :, 2] - self.B[:, :, 1]), axis=1)
          changes_now = np.abs(self.B[:, :, 1] - self.B[:, :, 0])
          keep = np.ones(self.N, dtype=bool)
          are_there_nan = np.isnan(self.B).any()
          for di in range(self.dim):
              keep = keep * (changes_now[di] < 4*std_changes_before[di])
              if are_there_nan:
                  keep = keep * ~np.isnan(self.B[di]).any(axis=1)

          sinx = np.arange(self.N)[~keep]
          
          dist_to_end = cdist(self.B[:,:,1].T, np.atleast_2d(self.y1))
          fl = np.where( dist_to_end[:,0]> 4*np.median(dist_to_end))[0]
          
          sinx = np.hstack((sinx,fl))
          
          temp = len(sinx)
          print("Identified %d invalid bridge trajectories "%len(sinx))
          
          if temp > int(self.N/2):
              self.valid = False
          if self.reject:
              if self.delete:
                  print("Deleting invalid trajectories...")
                  sinx = sinx[::-1]
                  for element in sinx:
                      self.B = np.delete(self.B, element, axis=1)
              else: #replace with stochastic   
                    while sinx.size>0:           
                        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Replacing!!!')  
                        for ti,tt in enumerate(self.timegrid[:-1]):
                            #print(ti)
                            if ti==0:
                                if self.uncertend ==0:
                                    for di in range(self.dim):
                                        self.B[di,sinx,-1] = self.y2[di]
                                elif self.uncertend ==-1:
                                    self.B[:,sinx,-1] = self.Z[:,sinx,-1]
                                elif self.uncertend ==-3:
                                    for di in range(self.dim):
                                        self.B[di,sinx,-1] = np.mean(self.Z[di,:,-1])
                                elif self.uncertend ==-2:
                                    self.B[:,sinx,-1] = np.random.multivariate_normal(self.y2, 
                                                                                    np.cov(self.Z[:,:,-1]), 
                                                                                    len(sinx)).T
                                else:                    
                                    self.B[:,sinx,-1] = np.random.multivariate_normal(self.y2, 
                                                                                    np.eye(self.dim)*self.uncertend, 
                                                                                    len(sinx)).T                                   
        
                                
                            else:
                                rev_ti = self.k -ti-1#np.where(self.timegrid==tt)[0][0]  
                                #print(rev_ti)
                                
                                grad_ln_ro = self.density_estimation(ti,rev_ti+1) #this estimates grad log rho on Bs
                                
                                self.gbB[:,sinx,rev_ti+1] =  -self.f(self.B[:,sinx,rev_ti+1], self.timegrid[rev_ti]) + self.g**2*grad_ln_ro[:,sinx]   
                                if (ti>=1) :#False:#callable(self.grad_log_pss):
                                  #for i in range(self.N):                                         
                                  self.B[:,sinx,rev_ti] = self.B[:,sinx,rev_ti+1] - self.f(self.B[:,sinx,rev_ti+1], self.timegrid[rev_ti])*self.dt + \
                                  self.dt*self.g**2*grad_ln_ro[:,sinx] + \
                                  (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,len(sinx)))
                                  #0.5*self.dt*self.g**2*self.grad_log_pss( self.B[:,i,rev_ti+1] ,self.g ) ###this is the gradient of the stationary p
                                      
                                       
                                
        
                        for di in range(self.dim):
                            self.B[di,sinx,0] = self.y1[di]
                            
                        dist_to_end = cdist(self.B[:,:,1].T, np.atleast_2d(self.y1))
                        sinx = np.where( dist_to_end[:,0]> 4*np.median(dist_to_end))[0] 
                        print('New sinx size: %d'%sinx.size)                         

          
          return 
    
        def calculate_u(self,grid_x,ti):
            """
            
    
            Parameters
            ----------
            grid_x : array of size d x number of points on the grid
            ti     : time index in timegrid for the computation of u
                Computes the control u on the grid or on a the point .
            
    
            Returns
            -------
            The control u(grid_x, t), where t=timegrid[ti].
    
            """
            #a = 0.001
            #grad_dirac = lambda x,di: - 2*(x[di] -self.y2[di])*np.exp(- (1/a**2)* (x[0]- self.y2[0])**2)/(a**3 *np.sqrt(np.pi))                 
            u_t = np.zeros(grid_x.T.shape)
            
            
            lnthsc1 = 2*np.std(self.B[:,:,ti],axis=1)
            lnthsc2 = 2*np.std(self.Z[:,:,ti],axis=1)
            
      
            bnds = np.zeros((self.dim,2))
            for ii in range(self.dim):
                if self.reweight==False or self.brown_bridge==False:
                    bnds[ii] = [max(np.min(self.Z[ii,:,ti]),np.min(self.B[ii,:,ti])),min(np.max(self.Z[ii,:,ti]),np.max(self.B[ii,:,ti]))]
                else:
                    bnds[ii] = [max(np.min(self.Ztr[ii,:,ti]),np.min(self.B[ii,:,ti])),min(np.max(self.Ztr[ii,:,ti]),np.max(self.B[ii,:,ti]))]
                    
            if ti<=5:# or (ti>= self.k-5):
                if self.reweight==False or self.brown_bridge==False:
                    ##for the first 5 timesteps, to avoid numerical singularities just assume gaussian densities
                    for di in range(self.dim):
                        mutb = np.mean(self.B[di,:,ti])
                        stdtb = np.std(self.B[di,:,ti])
                        mutz = np.mean(self.Z[di,:,ti])
                        stdtz = np.std(self.Z[di,:,ti])                
                        u_t[di] =  -(grid_x[:,di]- mutb)/stdtb**2 - (  -(grid_x[:,di]- mutz)/stdtz**2 )
                elif self.reweight==True and self.brown_bridge==True:
                    for di in range(self.dim):
                        mutb = np.mean(self.B[di,:,ti])
                        stdtb = np.std(self.B[di,:,ti])
                        mutz = np.mean(self.Ztr[di,:,ti])
                        stdtz = np.std(self.Ztr[di,:,ti])                
                        u_t[di] =  -(grid_x[:,di]- mutb)/stdtb**2 - (  -(grid_x[:,di]- mutz)/stdtz**2 )
            elif ti>5:
                ###if point for evaluating control falls out of the region where we have points, clip the points to 
                ###fall within the calculated region - we do not change the position of the point, only the control value will be
                ###calculated with clipped positions 
                bndsb = np.zeros((self.dim,2))
                bndsz = np.zeros((self.dim,2))
                for di in range(self.dim):
                    bndsb[di] = [np.min(self.B[di,:,ti]), np.max(self.B[di,:,ti])]
                    bndsz[di] = [np.min(self.Z[di,:,ti]), np.max(self.Z[di,:,ti])]            
                
                ###cliping the values of points when evaluating the grad log p
                grid_b = grid_x#np.clip(grid_x, bndsb[0], bndsb[1]) 
                grid_z = grid_x#np.clip(grid_x, bndsz[0], bndsz[1])        
      
                Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )
                for di in range(self.dim): 
                    score_Bw = score_function_multid_seperate(self.B[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc1,which_dim=di+1, kern=self.kern)(grid_b)
                    if self.reweight==False or self.brown_bridge==False: 
                        score_Fw = score_function_multid_seperate(self.Z[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc2,which_dim=di+1, kern=self.kern)(grid_z)
                    else:
                        bndsztr = np.zeros((self.dim,2))
                        for ii in range(self.dim):                        
                            bndsztr[di] = [np.min(self.Ztr[di,:,ti]), np.max(self.Ztr[di,:,ti])]  
                        grid_ztr = np.clip(grid_x, bndsztr[0], bndsztr[1])
                        lnthsc3 = 2*np.std(self.Ztr[:,:,ti],axis=1)
                        score_Fw = score_function_multid_seperate(self.Ztr[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc3,which_dim=di+1, kern=self.kern)(grid_ztr)
                    
                    u_t[di] = score_Bw - score_Fw
                # for di in range(self.dim):  
                #     u_t[di] = score_function_multid_seperate(self.B[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)(grid_x.T) \
                #              - score_function_multid_seperate(self.Z[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)(grid_x.T)
                         
                        
            return u_t
            
            
            
        def compute_controlled_flow(self):
            to_omit = []
            not_covered = np.zeros((self.N, self.k), dtype=bool)
            for ti, tt in enumerate(self.timegrid[:-1]):                                                           
            
                
                if ti==0:
                    for di in range(self.dim):
                        self.Fc[di,:, ti] = self.y1[di]
                else:
                    dtt = self.dt
                    uu1 = self.calculate_u(np.atleast_2d(self.Fc[:,:,ti-1]).T,ti)                
                    
                    if (ti==1) or (not self.control_deterministic):
                        
                        self.gfFc[:, :, ti] = self.f(self.Fc[:,:,ti-1])+self.g**2 *uu1#.T
                        self.Fc[:, :, ti] =  self.Fc[:,:,ti-1]+ dtt* self.f(self.Fc[:,:,ti-1])+dtt*self.g**2 *uu1\
                            + self.g*np.random.normal(loc = 0.0, scale = np.sqrt(dtt),size=self.N)
                        if ti<self.k-2:    
                            not_covered[self.check_if_covered(self.Fc[:, :, ti], ti),ti] = True
                            # print(ti, not_covered)               
                        
                    else:
                        self.gfFc[:, :, ti] = self.f(self.Fc[:,:,ti-1])+self.g**2 *uu1#.T                    
                        self.Fc[:,:,ti]  = self.Fc[:,:,ti-1] + dtt*self.f_seperate(self.Fc[:,:,ti-1],tt)+dtt*self.g**2 *uu1
                        where_big = np.where( (np.abs(self.Fc[:,:,ti]) >10   ).all(axis=0))
                        
                        if len(where_big[0])>0 or np.isnan(self.Fc[:,:,ti]).any():
                            print('hehehe')
                            nani = np.where(np.isnan(self.Fc[:,:,ti]))[0]
                            to_omit = np.hstack((nani, where_big[0])).reshape(-1,)
                            mask = np.ones(self.N,dtype='bool') 
                            mask[to_omit] = False   
                            for indx in to_omit:
                                self.Fc[:,indx,ti] = np.mean(self.Fc[:,mask,ti], axis=1)
                            print('-----------------')
                            print(self.Fc[:,:,ti])
                            print('-----------------')
            # compound = not_covered[:,0:-10]
            # for ti in range(1,10):
            #     compound += not_covered[:,ti:-10+ti]
            compound = np.sum(not_covered, axis=-1)
            to_omit = list(np.where(compound>=10)[0])    
            ###check for diverging
            dist_to_end = cdist(self.Fc[:,:,-2].T, np.atleast_2d(self.y2))
            sinx = np.where( dist_to_end[:,0]> 2*np.median(dist_to_end))[0]
            to_omit.extend(sinx)
            
            sinx = np.unique(np.array(to_omit))
            print('>>Fc sinx size: %d'%sinx.size)
            #print(sinx)
            counter = 0
            while sinx.size>0 and counter<100: 
                counter = counter + 1 ##to avoid infinite looping
                to_omit2 =[]
                not_covered = np.zeros((self.N, self.k), dtype=bool)
                print('>>>Replacing Fc: %d'%sinx.size)
                for ti, tt in enumerate(self.timegrid[:-1]):
                    
                    if ti==0:
                        for di in range(self.dim):
                            self.Fc[di,sinx, ti] = self.y1[di]
                    else:
                        dtt = self.dt
                        uu1 = self.calculate_u(np.atleast_2d(self.Fc[:,sinx,ti-1]).T,ti)                
                        
                        
                            
                        self.gfFc[:, sinx, ti] = self.f(self.Fc[:,sinx,ti-1])+self.g**2 *uu1#.T
                        self.Fc[:, sinx, ti] =  self.Fc[:,sinx,ti-1]+ dtt* self.f(self.Fc[:,sinx,ti-1])+dtt*self.g**2 *uu1\
                            + self.g*np.random.normal(loc = 0.0, scale = np.sqrt(dtt),size=sinx.size)
                            
                        if ti<self.k-2:    
                            not_covered[self.check_if_covered(self.Fc[:, :, ti], ti),ti] = True
                            # print(ti, not_covered)
                            # print('---------------')
                            #to_omit2.extend( list(not_covered))    
                # compound = not_covered[:,0:-10]
                # for ti in range(1,10):
                #     compound += not_covered[:,ti:-10+ti]
                compound = np.sum(not_covered, axis=-1)
                to_omit2 = list(np.where(compound>=10)[0])   
                #to_omit2 = list(np.where(compound>=2)[0])  
                dist_to_end = cdist(self.Fc[:,:,-2].T, np.atleast_2d(self.y2))
                sinx = np.where( dist_to_end[:,0]> 2*np.median(dist_to_end))[0]
                to_omit2.extend(sinx)
                sinx = np.unique(np.array(to_omit2))
                print('New Fc sinx size: %d'%sinx.size)                                         
            return 
        
        
        def check_if_covered(self, X, ti):
            """
            Checks if test point X falls within forward and backward densities at timepoint timegrid[ti]

            Parameters
            ----------
            X : TYPE
                DESCRIPTION.
            ti : TYPE
                DESCRIPTION.

            Returns
            -------
            Boolean variable - True if the text point X falls within the densities.

            """
            covered = np.ones(X.shape[1], dtype=bool)
            bnds = np.zeros((self.dim,2))
            for ii in range(self.dim):
                bnds[ii] = [max(np.min(self.Z[ii,:,ti]),np.min(self.B[ii,:,ti])),min(np.max(self.Z[ii,:,ti]),np.max(self.B[ii,:,ti]))]
                #bnds[ii] = [np.min(self.B[ii,:,ti]),np.max(self.B[ii,:,ti])]
            
                covered = covered & ( (X[ii] >= bnds[ii][0]) & (X[ii] <= bnds[ii][1]) )
                
            return np.where(~ covered)[0]
    
    
    
        def check_ifwhen_fw_covered(self, X):
            """
            Checks if test point X falls within forward densities and returns latest index in timegrid
            This is used for split bridge
    
            Parameters
            ----------
            X : nd array
                Point to check if covered.
            
    
            Returns
            -------
            Scalar - Index in bridge timegrid axis or -1 if not covered.
    
            """
            covered = np.ones(self.timegrid.size,dtype=bool)
            bnds = np.zeros((self.dim,  2))
    
            for ti in range(self.timegrid.size):
                for ii in range(self.dim):            
                    bnds[ii] = [np.min(self.Z[ii,:,ti]),np.max(self.Z[ii,:,ti])]
                
                    covered[ti] = covered[ti] * ( (X[ii] >= bnds[ii][0]) and (X[ii] <= bnds[ii][1]) )
            if np.sum(covered)==0:
                return -1
            else:
                timepoints_covered = np.where(covered==1)[0]
                ##return last
                return timepoints_covered[-1]
    
          
    
    #%%
    def construct_geodesic(start, stop, mani_samples, num_t, omit=None, N_nodes=10, plotting=False):
        """
        Constructs the geodesic between "start" and "stop" on the manifold represented by the 
        "mani_samples" after omitting the indices in "omit".
        Omited indices are used to force the geodesic to have the correct chirality/orientation, and 
        prevent the algorithm to return a backward geodesic.
    
        """
        # Construct an artificial data set
        #data_params = {'N': Ob.shape[1], 'data_type': 1, 'sigma': 0.1, 'r':0.5}
        if omit is not None:
            data = np.delete(mani_samples.T, omit, axis=0)
        else:
            data = mani_samples.T
        
    
        # Construct a Riemannian metric from the data
        manifold = manifolds.LocalDiagPCA(data=data, sigma=0.15, rho=1e-3)
    
        
        # Initialize the geodesic solvers
    
        #N_nodes = 10#0
        solver_graph = geodesics.SolverGraph(manifold, 
                                             data=KMeans(n_clusters=N_nodes,
                                                         n_init=10,
                                                         max_iter=1000).fit(data).cluster_centers_, 
                                             kNN_num=int(np.log(N_nodes)), tol=1e-2)
    
    
    
        # Compute the shortest path between two points
        c0 = utils.my_vector(start)
        c1 = utils.my_vector(stop)
    
        curve_graph, logmap_graph, curve_length_graph, failed_graph, solution_graph \
                        = geodesics.compute_geodesic(solver_graph, manifold, c0, c1)
        
        # Print results:
        print('===== Solvers result =====')
        print('[GRAPH solver] [Failed: ' + str(failed_graph) + '] [Length: ' + str(np.round(curve_length_graph, 2)) + ']')
        return curve_graph
    
    
    #%%
    
    
    ####brownian bridge from x1 to x2
    ## to be called with  a wrapper that sets initial and terminal point and final time
    def f1(x,t,  x2, T):
        if t==T:
            return np.tile(np.array([0,0]),(x.shape[1],1) ).T
        else:
            return np.array([(x2[0]-x[0])/(T-t),  (x2[1]-x[1])/(T-t)  ]) 
    
    
    #%%    
    

    def run_bridge_augmentation(f_est, bridge_type, ksteps, N, M, reps, strength=1, savetitle=''):
        """
        f_est : estimated drift for bridge sampling
        bridge_type : string, determines what kind of bridge will be employed
                      options:
                              - 'normal' : normal bridge without path cost
                              - 'geodesic' : geodesic bridge with f_est drift
                              - 'grodesic-brownian' : geodesic bridge as controlled brownian bridge
        ksteps : int, number of timepoints to be sampled in the each bridge
        N : int, number of particles to be employed for computing each bridge
        M : int, number of inducing points to be employed for the gradient log density estimation
        reps: int, determines how many copies of controled paths will be sampled
        strength: float, determines the strength of forcing towards the geodesic
        savetitle : string, title for saving the plots of each bridge
        
        """
        
        gbALL = np.zeros((dim, N, timegrid.size))
        BALL = np.zeros((dim, N, timegrid.size))
        to_keep = np.ones(Ob.shape[1], dtype=bool)
        epsilon_ball = 0.85
        for tindx, tim in enumerate(timegrid_obs[:-1]): ##loop over the observation coarse timegrid
          if tindx>=0:
              print('Currently at brindge:',tindx)
              t1 = timegrid_obs[tindx]
              t2 = timegrid_obs[tindx+1]
              y1 = Ob[:,tindx]
              y2 = Ob[:,tindx+1]
              tindx_outer1 = tindx*ksteps
              tindx_outer2 = (tindx+1)*ksteps
              
              double_bridge = False
              brown_bridge = False
              brown_bridgeinit = False
              try:
                  if bridge_type=='normal':
                      #### Initial try #################################################################
                      bridg2d = BRIDGE_ND_reweight(t1, t2, y1, y2, f_est, g, N, M, dens_est='nonparametric',
                                                  reject=True, plotting=True, sample='deterministic')
                      print('The bridge is valid:', bridg2d.valid)
                      
                      
    
                  #### if  try with geodesic bridge
                  elif bridge_type=='geodesic':
                      if tindx>0:
                          to_keep = distances[tindx-1]>epsilon_ball
                          to_keep[tindx] = True
                          to_keep[tindx+1] = True
                      print('Sampling with geodesic bridge...')
                      ##get geodesic function
                      to_plot = tindx<15
                      geodesic_fun = construct_geodesic(y1, y2, Ob[:,to_keep], timegrid_obs.size, omit=None, N_nodes=10, plotting=to_plot)          
                      
                      def Ugeod2(x,t, beta=strength): 
                        To = np.linspace(0, 1, ksteps+1)
                        curve_eval = geodesic_fun(To)[0]                 
                        return beta*(-(np.linalg.norm(np.atleast_2d(curve_eval[:,t]).T - x[:,:], axis=0)))
                      
                      brown_bridgeinit = False
                      bridg2d = BRIDGE_ND_reweight(t1, t2, y1, y2, f_est, g, N, M, dens_est='nonparametric',
                                              reject=True, plotting=True, reweight=True, U=Ugeod2 ,uncertend=-3)
                      
                      print('The bridge is valid:', bridg2d.valid)
                      if bridg2d.valid:
                          
                          pass
                      else:  
                          #### if  not valid try with geodesic bridge              
                          print()
                          print('Resampling with geodesic bridge...')
                          if tindx>0:
                              to_keep = distances[tindx-1]>epsilon_ball
                              to_keep[tindx] = True
                              to_keep[tindx+1] = True
                          ##get geodesic function
                          geodesic_fun = construct_geodesic(y1, y2, Ob[:,to_keep], timegrid_obs.size, omit=None, N_nodes=10)          
                          
                          def Ugeod2(x,t, beta=strength): 
                            To = np.linspace(0, 1, ksteps+1)
                            curve_eval = geodesic_fun(To)[0]                 
                            return beta*(-(np.linalg.norm(np.atleast_2d(curve_eval[:,t]).T - x[:,:], axis=0)))
                          def f_br(x,t):
                              return f1(x,t,  y2, t2-t1)
                          brown_bridgeinit = True
                          bridg2d = BRIDGE_ND_reweight(t1, t2, y1, y2, f_br, g, N, M, dens_est='nonparametric',
                                                  reject=True, plotting=True, reweight=True, U=Ugeod2,uncertend=-3)
                          
                          print('The bridge is valid:', bridg2d.valid)
                          #if bridg2d.valid==True:
                          
              except OverflowError:
                  print('skipping bridge %d'%tindx)
              else:
    
                  BALL[:,:,tindx_outer1+2:tindx_outer2] = bridg2d.Fc[:,:,2:-1]
                  gbALL[:,:,tindx_outer1+2:tindx_outer2] = bridg2d.gfFc[:,:,2:-1]
                
                  # if tindx%50==0: ##store every 50 briddges
                  #     filehandler = open(save_dir+'Estimated_Dense_'+savetitle+'.dat',"wb")
                  #     pickle.dump(BALL[:,:,:tindx*ksteps],filehandler)
                  #     filehandler = open(save_dir+'Estimated_Dense_gb_'+savetitle+'.dat',"wb")
                  #     pickle.dump(gbALL[:,:,:tindx*ksteps],filehandler)
        ##final store all          
        # filehandler = open(save_dir+'Estimated_Dense_'+savetitle+'.dat',"wb")
        # pickle.dump(BALL, filehandler)
        # filehandler = open(save_dir+'Estimated_Dense_validity_'+savetitle+'.dat',"wb")
        # pickle.dump(gbALL, filehandler)
    
        return BALL, gbALL  
#%%
        
        
#%%    
    import torch
    def pairwise_distances(x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)
        
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)
    
    
    
    def K(x,y,l,multil=False):
        if multil:                         
            res = torch.ones((x.shape[0],y.shape[0]))  
            l = torch.tensor(l)              
            for ii in range(l.size()[0]): 
                #tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
                ##puts into tempi the cdist result
                #my_cdist(x[:,ii:ii+1], y[:,ii:ii+1],tempi,'sqeuclidean')
                
                tempi = pairwise_distances(x[:,ii:ii+1], y[:,ii:ii+1])
                res = torch.multiply(res, torch.exp(-tempi/(2*l[ii]*l[ii])))                    
            return res
        else:
            #tempi = np.zeros((x.shape[0], y.shape[0] ))                
            #my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
            tempi = pairwise_distances(x, y)
            
            return torch.exp(-tempi/(2*l*l))
    
    


    #%%
    ## EM GP path likelihood inference

    def GP_path_likelihood(Sp, n_sparsegp, BALL2, gbALL2, l1=1, l2=1):
        ##I1 is ks A(x)Ks.T integral
        I1 = torch.zeros(n_sparsegp, n_sparsegp, dim) ###at the end multiply time dt/N
    
        ##I2 is ks g(x,t)
        I2 = torch.zeros(n_sparsegp, dim)
    
        ########################## GP estimation
        ##this largely follows the GP as described by Ruttor, BAtz, Opper 2013 
        ##with the difference that we employ the SAME kernel for all dimensions
        ## while Ruttor et al. mention only using the relevant componet of the kernel 
        ## for each dimension.
        l1 = l1
        l2 = l2
        sample_num = N
        lnth = torch.ones(dim)#, requires_grad=True)
        lnth[0] = l1
        lnth[1]  = l2
        
        Ls = torch.zeros((n_sparsegp, n_sparsegp, dim)) #Lambda_s
        ds = torch.zeros((n_sparsegp, dim))
        #start = time.time()
        #Kss = [ K(torch.atleast_2d(Sp[di]).T, torch.atleast_2d(Sp[di]).T, l=lnth[di].clone(), multil=False) for di in range(dim) ]
        #Kss_inv = [ torch.linalg.inv( Kss[di] + torch.eye(n_sparsegp)*1e-3 ) for di in range(dim)]
        Kssall = torch.tensor(K(torch.atleast_2d(Sp).T, torch.atleast_2d(Sp).T, l=lnth.clone(), multil=True) )
        Kssall_inv = torch.linalg.inv( Kssall + torch.eye(n_sparsegp)*1e-3 ) 
    
        #ks = [ lambda x, di=di: K(torch.atleast_2d(Sp[di]).T, x, l=lnth[di].clone(), multil=False)  for di in range(dim) ]
        ksa = [ lambda x, di=di: torch.tensor(K(torch.atleast_2d(Sp).T, x, l=lnth.clone(), multil=True))  for di in range(dim) ] ##all dimensions are the same
        for di in range(dim):
          for ti in range(BALL2.shape[-1]):
              Xtemp = BALL2[:,:sample_num,ti] #dim x N        
              #gbB = gbALL2[:,:sample_num,ti]     
              
              ###signle kernel
              I1[:,:,di] = I1[:,:,di] + ksa[di](torch.atleast_2d(Xtemp).T) @ ksa[di](torch.atleast_2d(Xtemp).T).T   ##consider alldimensions all together
              I2[:, di] = I2[:, di]+ (ksa[di](torch.atleast_2d(Xtemp).T) @ torch.atleast_2d( gbALL2[di,:sample_num,ti]).T )[:,0]
        #stop = time.time()   
              
        for di in range(dim):
            I1[:, :, di] = I1[:, :, di]* dt /(sample_num) #(T/timegrid.size-to_keep.size)
            I2[:, di]  = I2[:, di]* dt/sample_num #(T/timegrid.size-to_keep.size)
    
        for di in range(dim):    
    
            ###when consider all dimensions together
            Ls[:, :, di] = (1/torch.tensor(g**2))* Kssall_inv @ I1[:,:,di].double()@Kssall_inv.T  #sparse x sparse
            ds[:, di] = (1/g**2)* Kssall_inv@ I2[:,di].double()
        ###TO DO: find a way to do this properly in one functoin
        ## f= ks(x).T [ I+Ls Kss]^-1 ds
        ## single kernel
        f_est1 = lambda x: ksa[0]( torch.atleast_2d(x).T ).T @ torch.linalg.inv(  torch.eye(n_sparsegp)+ Ls[:,:,0].double()@Kssall.double()   ).double() @ ds[:,0].double()
        f_est2 = lambda x: ksa[1]( torch.atleast_2d(x).T ).T @ torch.linalg.inv(  torch.eye(n_sparsegp)+ Ls[:,:,1].double()@Kssall.double()   ).double() @ ds[:,1].double()
    
    
        ################## compute expected negative log data likelihood
    
        fX = torch.zeros(dim)
        gradfX = torch.zeros(dim)
        fgb = torch.zeros((dim, dim))
        print('Computing the likelihood')
        for ti in range(BALL2.shape[-1]):        
            fgb = fgb + (1/g**2)* f_est1(BALL2[:,:sample_num,ti]) @ gbALL2[0, :sample_num, ti].T + (1/g**2)* f_est2(BALL2[:,:sample_num,ti]) @ gbALL2[1, :sample_num, ti].T        
            fX = fX + f_est1(BALL2[:,:sample_num,ti]) @f_est1(BALL2[:,:sample_num,ti]).T  + f_est2(BALL2[:,:sample_num,ti]) @f_est2(BALL2[:,:sample_num,ti]).T 
            gradfest1 = torch.autograd.functional.jacobian(f_est1, inputs=BALL2[:,:sample_num,ti].T , create_graph=True)
            gradfest2 = torch.autograd.functional.jacobian(f_est2, inputs=BALL2[:,:sample_num,ti].T , create_graph=True)
            
            gradfX = gradfX + gradfest1 +gradfest2
        ll = (dt/sample_num)*(0.5* torch.sum(fX) + torch.sum(gradfX) + torch.sum(fgb)).clone()
        print('Expected negative log data likelihood:',ll)
    
        return f_est1, f_est2, ll, Ls, ds, I1,I2, lnth

        
        
        
        
    #%%
    
    import os
    Ts = [500, 1000]#, 1500]
    obs_denz = [160, 200, 240, 280]#[80,120,160,200, 240, 280,  320]
    gss = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9, 1.]
    
    seed = os.environ['SGE_JOB_ID_FULL'].split(".")[1]
    Ti, obi, gi, seed = int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(seed)
    T = Ts[Ti]  #length of simulation
    obs_dens = obs_denz[obi] #interval between successive observations
    g = gss[gi] #noise amplitude
    np.random.seed(seed)
    
    h = 0.01 #sim_prec
    dt = h
    t_start = 0.
    D = g**2
    correct_g = 0.5
    
    
    timegrid = np.arange(0,T,h)
    timegrid_obs = timegrid[::obs_dens]  
    
    save_dir = '/work/maoutsa/wrong_sigmas5/FW_2Geodesic_'+dsystem+'_nose_%.2f_dens_%d_time_%d_seed_%d/'%(g, obs_dens, T, seed)
    
    if os.path.exists(save_dir) == False:  
        os.mkdir(save_dir)  
        
        
    with open(save_dir+"output.txt", "a") as fi:
        fi.truncate(0) ###delete existing output - start anew
        print('Running inference for '+dsystem+ ' system with dimension: %d'%dim)
        print('Running inference for '+dsystem+ ' system with dimension: %d'%dim, file=fi)
    
        print('Problem characterisitcs: \n Simulation duration: %d \n Observation interval: %d \n Noise amplitude: %.2f \n Simulation precision: %.3f'%(T, obs_dens, g, h))
        print('Problem characterisitcs: \n Simulation duration: %d \n Observation interval: %d \n Noise amplitude: %.2f \n Simulation precision: %.3f'%(T, obs_dens, g, h), file=fi)
        print('Seed:  %d '%(seed))
        print('Seed:  %d '%(seed), file=fi)
    
    F = simulate_path(f, timegrid, correct_g)
    
    ##get observations
    Ob = F[:,::obs_dens]
    obs_dt=obs_dens*h
    
    with open(save_dir+"output.txt", "a") as fi: 
        print('Saved simulation details and true f')   
        print('Number of observations: %d'%Ob.shape[1])
        print('Number of observations: %d'%Ob.shape[1], file=fi)
    
    ##Simulate 10 other paths with correct drift and compute their Wasserstein 
    ##distance to the observations
    default_wasser_m = np.zeros(5)
    default_wasser_std = np.zeros(5)
    for ii in  range(5):
        Fake = simulate_path(f, timegrid, correct_g)
        default_wasser_m[ii], default_wasser_std[ii] = compute_Wasserstein(Fake[:,::obs_dens], Ob)
    
    filehandler = open(save_dir+"Default_Wasser.dat","wb")
    to_sav = dict()    
    to_sav['default_wasser_m'] = default_wasser_m
    to_sav['default_wasser_std'] = default_wasser_std    
    pickle.dump(to_sav,filehandler, protocol=4)
    
    del Fake
    
    
    with open(save_dir+"output.txt", "a") as fi: 
        print('Saved default Wasserstein distances')  
        print('Saved default Wasserstein distances', file=fi)  
        print('Mean default Wassertstein: %.4f, Std default Wasserstei: %.4f'%(np.mean(default_wasser_m), np.std(default_wasser_m) ))
        print('Mean default Wassertstein: %.4f, Std default Wasserstei: %.4f'%(np.mean(default_wasser_m), np.std(default_wasser_m) ), file=fi)
        
    ##Evaluate true vector field
    bounds = np.array([np.min(Ob, axis=1), np.max(Ob,axis=1)]).T
    n_grid = 20
    start = time.time()
    xt, ztrue = evaluate_vector_field( f, etype= 'pure', n=n_grid, bounds=bounds)
    stop = time.time()
    
    with open(save_dir+"output.txt", "a") as fi:    
        print('Evaluated the true vector field. Time taken: %.4f'%(stop-start))
        print('Evaluated the true vector field. Time taken: %.4f'%(stop-start), file=fi)
    
    filehandler = open(save_dir+"Observations_true_f_and_sim_details.dat","wb")
    to_sav = dict()
    if seed==15:
        to_sav['Ob'] = Ob
        to_sav['F'] = F
    to_sav['g'] = g
    to_sav['obs_dt'] = obs_dt
    to_sav['h'] = h
    to_sav['timegrid'] = timegrid
    to_sav['timegrid_obs'] = timegrid_obs
    to_sav['obs_dt'] = obs_dt
    to_sav['obs_dens'] = obs_dens
    
    to_sav['ztrue'] = ztrue
    to_sav['xt'] = xt
    to_sav['bounds'] = bounds
    to_sav['n_grid'] = n_grid
    pickle.dump(to_sav,filehandler, protocol=4)
    
    
    ##Estimate drift from sparse Observations Ob
    nsparse = 200
    mf_init = GP_drift_estimation(Ob, obs_dt, nsparse=nsparse)
    posterior = mf_init.posterior()
    def f_init(xo,t=0):
        return (posterior.predict_f(np.atleast_2d(xo.T))[0] ).numpy().T
    def f_initvar(xo,t=0):
        return (posterior.predict_f(np.atleast_2d(xo)) )
    mu_init, var_init, RMSE_init, F_init, Wm_init, Wstd_init = evaluation_for_given_drift(f_init, f_initvar, ztrue, timegrid, obs_dens, 
                                                                                          Ob, bounds=bounds,g=g, etype='est')
    
    with open(save_dir+"output.txt", "a") as fi:    
        print('RMSE: ',RMSE_init)
        print('RMSE: %.6f'%(RMSE_init), file=fi)
        print('Mean Wasserstein distance: ',np.mean(Wm_init))
        print('Mean Wasserstein distance: %.6f'%np.mean(Wm_init), file=fi)
    
    
    ###Estimate initial likelihood of observations
    fX = f_init(Ob)
    
    ll_init = (0.5*h/T)* np.sum(np.linalg.norm(fX, axis=0)**2)/D -2*np.sum(fX[0,:-1]@np.diff(Ob, axis=1)[0]) -2*np.sum(fX[1,:-1]@np.diff(Ob, axis=1)[1])
    
    
    with open(save_dir+"output.txt", "a") as fi:    
        print('Negative log likelihood: ',ll_init)
        print('Negative log likelihood: %.6f'%(ll_init), file=fi)
    
    
    filehandler = open(save_dir+"Init_f_est.dat","wb")
    to_sav = dict()
    to_sav['ll_init'] = ll_init
    to_sav['nsparse'] = nsparse
    to_sav['mu_init'] = mu_init
    to_sav['var_init'] = var_init
    to_sav['RMSE_init'] = RMSE_init
    if seed==15:
        to_sav['F_init'] = F_init
    to_sav['Wm_init'] = Wm_init
    to_sav['Wstd_init'] = Wstd_init
    pickle.dump(to_sav,filehandler, protocol=4)
    print('Saved initial drift estimation')
    
    
    #bridge_type = 'geodesic'  #'normal' #'geodesic-brownian'
    import copy
    distances = squareform(pdist(Ob.T))
    ksteps = obs_dens        
    #N = 100
    dim = 2
    reps = 10
    M = 40
    dt = h
    augmentation_num = 0
    strengths = [0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1]
    if obs_dens==50 and obi==0:
        bridge_types = ['normal']*10
    else:
        bridge_types = ['geodesic', 'normal', 'geodesic', 'geodesic', 
                        'geodesic', 'geodesic', 'geodesic', 'geodesic', 'geodesic', 'normal']
    Ns = [100]*10
    epsilon_ball = 0.75
    n_sparsegp = 300
    f_est = None
    likehoods = [ll_init]
    #dim x nsparse 
    Sp = torch.from_numpy(np.random.uniform(low=[np.min(Ob[0])-0.5,np.min(Ob[1])-0.5], high=[np.max(Ob[0])+0.5,np.max(Ob[1])+0.5], size=(n_sparsegp,dim) ).T)

    while augmentation_num<2:
        N = Ns[augmentation_num]
        bridge_type = bridge_types[augmentation_num]
        strength = strengths[augmentation_num]
        if augmentation_num==0:
            BALL, gbALL = run_bridge_augmentation(f_init, bridge_type, ksteps, 
                                                          N, M, reps, strength=strength,
                                                          savetitle='Augmentation_num_%d_augm_type_%s'%(augmentation_num, bridge_type) )
        else:
            BALL, gbALL = run_bridge_augmentation(f_est, bridge_type, ksteps, 
                                                          N, M, reps, strength=strength,
                                                          savetitle='Augmentation_num_%d_augm_type_%s'%(augmentation_num, bridge_type) )
    
    
        with open(save_dir+"output.txt", "a") as fi:    
            print('Ready with augmentation ')
            print('Ready with augmentation', file=fi)
            
        if seed==15:
            filehandler = open(save_dir+"bridges_from_%d_augmentation.dat"%augmentation_num,"wb")
            to_sav = dict()
            to_sav['BALL'] = BALL
            to_sav['gbALL'] = gbALL    
            pickle.dump(to_sav,filehandler)
            print('Saved bridges of %d augemntation'%augmentation_num)
            
        to_keep = np.where(BALL[0,0,:]!=0)[0]
        #print(to_keep)
        BALL2 = torch.from_numpy(copy.deepcopy(BALL[:,:,to_keep]))
        gbALL2 = torch.from_numpy(copy.deepcopy(gbALL[:,:,to_keep]))
    
        
        
        ##compute lenthscales###############################
        """
        l1s =np.arange(0.5, 2.1, 0.25)
        l2s = np.arange(0.5, 2.1, 0.25)
        all_like = np.zeros((l1s.size, l2s.size))
        for iii,l1 in enumerate(l1s):
            with open(save_dir+"output.txt", "a") as fi:    
                print('l1: ',l1)
                print('l1: %.3f'%(l1), file=fi)
            for jjj,l2 in enumerate(l2s):
                _, _, all_like[iii,jjj], _, _, _,_, _ = GP_path_likelihood(Sp, n_sparsegp, BALL2[:,:,:1000], gbALL2[:,:,:1000], l1, l2)
        l1ind, l2ind = np.unravel_index(np.argmin(all_like), all_like.shape) 
        """
        l1best = 1#l1s[l1ind]
        l2best = 1#l2s[l2ind]        
        
        
        f_est1, f_est2, ll, Ls, ds, I1,I2, lnth = GP_path_likelihood(Sp, n_sparsegp, BALL2, gbALL2,l1best, l2best)
        
        
        likehoods.append(ll)
        with open(save_dir+"output.txt", "a") as fi:                
            print('Likelihood: %.5f'%ll)
            print('Likelihood: %.5f'%ll, file=fi)
            print('Best lenghtscales: ', lnth)
            print('Best lenghtscales: ',lnth, file=fi)
        
        ##I know this is ugly but these are the same functions constructed to becalled from different functions
        ##but for now this works
        ##TO DO: FIX THIS SHIT!!!!
        #def f_est_eval(xo,t=0):
        #    xo = torch.atleast_2d(torch.tensor(xo)).T        
        #    return torch.tensor([f_est1(xo), f_est2(xo)])
    
        def f_est(xo,t=0):
            ### this is used to eventually simulate bridges for the next iteration.
            ##xo shape is N x dim
            xo = torch.atleast_2d(torch.tensor(xo))     
            return np.array([f_est1(xo).cpu().detach().numpy(), f_est2(xo).cpu().detach().numpy()])
    
        
    
        filehandler = open(save_dir+"f_est_after_%d_augm.dat"%augmentation_num,"wb")
        to_sav = dict()
        to_sav['f_est'] = f_est    
        pickle.dump(to_sav,filehandler)
        print('Saved drift estimation after %d augemntation'%augmentation_num)
            
        mu_1, var_1, RMSE_1, F_1, Wm_1, Wstd_1 = evaluation_for_given_drift(f_est, f_est, ztrue, timegrid, obs_dens, 
                                                                                          Ob, bounds=bounds, g=g, etype='pure')
        
    
        filehandler = open(save_dir+"f_est_after_%d_augm_n_eval.dat"%augmentation_num,"wb")
        to_sav = dict()
        to_sav['ll'] = ll
        to_sav['Ls'] = Ls
        to_sav['ds'] = ds
        to_sav['I1'] = I1
        to_sav['I2'] = I2
        to_sav['lnth'] = lnth
        to_sav['likehoods'] = likehoods
        to_sav['nsparse'] = nsparse
        to_sav['mu_1'] = mu_1
        to_sav['var_1'] = var_1
        to_sav['RMSE_1'] = RMSE_1
        if seed==15:
            to_sav['F_1'] = F_1
        to_sav['Wm_1'] = Wm_1
        to_sav['Wstd_1'] = Wstd_1
        pickle.dump(to_sav,filehandler, protocol=4)
    
        print('Saved drift estimation after %d augemntation and evaluation'%augmentation_num)
    
    
        with open(save_dir+"output.txt", "a") as fi:    
            print('RMSE: ',RMSE_1)
            print('RMSE: %.6f'%(RMSE_1), file=fi)
            print('Mean Wasserstein distance: ',np.mean(Wm_1))
            print('Mean Wasserstein distance: %.6f'%np.mean(Wm_1), file=fi)
    
    
    
        augmentation_num += 1
        ###end-line