# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 04:03:31 2022

@author: maout
"""

import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import seaborn as sns
import numpy as np
import pickle

dsystem = 'LC'
##for neurips plot I used the fw_22 folder
save_folder = 'C:\\Users\\maout\\results_LC_single_instance\\fw_22\\'
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib import cm
Ts = [500, 1000]#, 2000]
obs_denz = [50,200, 240, 280]#,  320]#, 50]
gss = [0.25, 0.5, 0.75, 1.]

h = 0.01 #sim_prec
dt = h
t_start = 0.
dim = 2

seeds = [13, 14] #[15,16,17,18,19] #
RMSES_init = np.empty((len(Ts),len(obs_denz),len(gss),len(seeds)))#.fill(np.nan)
RMSES1 = np.empty((len(Ts),len(obs_denz),len(gss),len(seeds)))#.fill(np.nan)
RMSES2 = np.empty((len(Ts),len(obs_denz),len(gss),len(seeds)))#.fill(np.nan)

wRMSES_init = np.empty((len(Ts),len(obs_denz),len(gss),len(seeds)))#.fill(np.nan)
wRMSES1 = np.empty((len(Ts),len(obs_denz),len(gss),len(seeds)))#.fill(np.nan)
wRMSES2 = np.empty((len(Ts),len(obs_denz),len(gss),len(seeds)))#.fill(np.nan)

Wm_init = np.empty((len(Ts),len(obs_denz),len(gss),len(seeds)))#.fill(np.nan)
Wm1 = np.empty((len(Ts),len(obs_denz),len(gss),len(seeds)))#.fill(np.nan)
Wm2 = np.empty((len(Ts),len(obs_denz),len(gss),len(seeds)))#.fill(np.nan)

LL_init = np.empty((len(Ts),len(obs_denz),len(gss),len(seeds)))#.fill(np.nan)
LL1 = np.empty((len(Ts),len(obs_denz),len(gss),len(seeds)))#.fill(np.nan)
LL2 = np.empty((len(Ts),len(obs_denz),len(gss),len(seeds)))#.fill(np.nan)


pal = sns.color_palette("deep")
colors = pal.as_hex()

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV




def wcompute_mean_squared_error(A, B, weights):
    #dim x num_poins
    mse = np.average((A - B)**2,axis=1,  weights=weights)
    return np.mean(np.sqrt(mse))

from math import acos
def compute_mean_squared_error(A, B):
    mse = np.mean((A-B)**2)    
    return np.sqrt(mse)

def  compute_angle(u,v):
    return acos(u@v/(np.linalg.norm(u)*np.linalg.norm(v)))
    #return atan2( abs(u[0]*v[1]-v[0]*u[1]), u[0]*u[1]+v[0]*v[1] )    


def compute_the_angles(m1, m2):
     grid_x, grid_y = m1[0].shape#size of the grid at each direction
     angles = np.zeros((grid_x, grid_y))
     for i in range(grid_x):
         for j in range(grid_y):
             
             angles[i, j] = compute_angle(m1[:, i,j], m2[:, i,j])
     return angles
 
#%%
#"""
for Ti,T in enumerate(Ts):
    for obi, obs_dens in enumerate(obs_denz):
        for gi,g in enumerate(gss):
            for sedi,seed in enumerate(seeds):
                D = g**2
                timegrid = np.arange(0,T,h)
                timegrid_obs = timegrid[::obs_dens]  
                if obi==5:
                    save_dir = 'FW_2Geodesic_'+dsystem+'_nose_%.2f_dens_%d_time_%d_seed_%d_b\\'%(g, obs_dens, T, seed)
                else:
                    save_dir = 'FW_2Geodesic_'+dsystem+'_nose_%.2f_dens_%d_time_%d_seed_%d\\'%(g, obs_dens, T, seed)
                
                    
                try:
                    augmentation_num = -1 ##just for errors
                    filename = save_folder+save_dir+"Observations_true_f_and_sim_details.dat"
                    file = open(filename,'rb')
                    to_sav = pickle.load(file)
                    ztrue = to_sav['ztrue'] 
                    xt = to_sav['xt'] 
                    file.close()
                    
                    filename = save_folder+save_dir+"Observations.dat"
                    file = open(filename,'rb')
                    to_sav = pickle.load(file)
                    Obs = to_sav['Ob'] 
                    
                    params = {"bandwidth": np.logspace(-1, 1, 20)}
                    grid = GridSearchCV(KernelDensity(), params)
                    grid.fit(Obs.reshape(dim, -1).T)
                    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
                    # use the best estimator to compute the kernel density estimate
                    kde = grid.best_estimator_
                    weights = np.exp(kde.score_samples(xt.reshape(dim, -1).T))
                    
                    # plt.figure()
                    # plt.scatter(xt.reshape(dim, -1)[0],xt.reshape(dim, -1)[1], 10*weights)

                    
                    filename = save_folder+save_dir+"Init_f_est.dat"
                    file = open(filename,'rb')
                    to_sav = pickle.load(file)                    
                    mu_init = to_sav['mu_init']                     
                    wRMSE_init = wcompute_mean_squared_error(ztrue.reshape(dim, -1),
                                                              mu_init.reshape(dim, -1),
                                                              weights)                    
                    
                    file.close()
                    ##1st augmentation
                    augmentation_num = 0
                    #f_est_after_%d_augm_n_eval.dat"%augmentation_num
                    filename = save_folder+save_dir+"f_est_after_%d_augm_n_eval.dat"%augmentation_num
                    file = open(filename,'rb')
                    to_sav = pickle.load(file)
                    
                    
                    
                    mu_1 = to_sav['mu_1'] 
                    
                    wRMSE1 = wcompute_mean_squared_error(ztrue.reshape(dim, -1),
                                                              mu_1.reshape(dim, -1),
                                                              weights)
                    file.close()
                    
                    
                    ##2nd augmentation
                    augmentation_num = 1
                    #f_est_after_%d_augm_n_eval.dat"%augmentation_num
                    filename = save_folder+save_dir+"f_est_after_%d_augm_n_eval.dat"%augmentation_num
                    file = open(filename,'rb')
                    to_sav = pickle.load(file)
                    
                    
                    mu_2 = to_sav['mu_1'] 
                    
                    wRMSE2 = wcompute_mean_squared_error(ztrue.reshape(dim, -1),
                                                              mu_2.reshape(dim, -1),
                                                              weights)
                    file.close()
                    
                    filehandler = open(save_folder+save_dir+"wRMSEs.dat","wb")
                    to_sav = dict()    
                    to_sav['wRMSE1'] = wRMSE1 
                    to_sav['wRMSE2'] = wRMSE2 
                    to_sav['wRMSE_init'] = wRMSE_init 
                    pickle.dump(to_sav,filehandler)
                    
                except EOFError:
                    print('Not found: '+ save_dir, augmentation_num)
                    
                    
                except FileNotFoundError:
                    print('Not found: '+ save_dir, augmentation_num)
#"""             
#%%

for Ti,T in enumerate(Ts):
    for obi, obs_dens in enumerate(obs_denz):
        for gi,g in enumerate(gss):
            for sedi,seed in enumerate(seeds):
                D = g**2
                timegrid = np.arange(0,T,h)
                timegrid_obs = timegrid[::obs_dens]  
                if obi==5:
                    save_dir = 'FW_2Geodesic_'+dsystem+'_nose_%.2f_dens_%d_time_%d_seed_%d_b\\'%(g, obs_dens, T, seed)
                else:
                    save_dir = 'FW_2Geodesic_'+dsystem+'_nose_%.2f_dens_%d_time_%d_seed_%d\\'%(g, obs_dens, T, seed)
                
                    
                try:
                    augmentation_num = -1 ##just for errors
                    filename = save_folder+save_dir+"Observations_true_f_and_sim_details.dat"
                    file = open(filename,'rb')
                    to_sav = pickle.load(file)
                    ztrue = to_sav['ztrue'] 
                    xt = to_sav['xt'] 
                    file.close()
                    
                    filename = save_folder+save_dir+"wRMSEs.dat"
                    file = open(filename,'rb')
                    to_sav = pickle.load(file)
                    wRMSES_init[Ti, obi, gi, sedi] = to_sav['wRMSE_init']
                    wRMSES1[Ti, obi, gi, sedi] = to_sav['wRMSE1']
                    wRMSES2[Ti, obi, gi, sedi] = to_sav['wRMSE2']
                    file.close()

                    
                    filename = save_folder+save_dir+"Init_f_est.dat"
                    file = open(filename,'rb')
                    to_sav = pickle.load(file)
                    
                    mu_init = to_sav['mu_init'] 
                    RMSES_init[Ti, obi, gi, sedi] = to_sav['RMSE_init']
                    
                    
                    Wm_init[Ti, obi, gi, sedi] = np.mean(to_sav['Wm_init'])
                    LL_init[Ti, obi, gi, sedi] = to_sav['ll_init']
                    file.close()
                    ##1st augmentation
                    augmentation_num = 0
                    #f_est_after_%d_augm_n_eval.dat"%augmentation_num
                    filename = save_folder+save_dir+"f_est_after_%d_augm_n_eval.dat"%augmentation_num
                    file = open(filename,'rb')
                    to_sav = pickle.load(file)
                    
                    
                    
                    mu_1 = to_sav['mu_1'] 
                    LL1[Ti, obi, gi, sedi] = to_sav['ll'] 
                    Wm1[Ti, obi, gi, sedi] = np.mean(to_sav['Wm_1'] )
                    RMSES1[Ti, obi, gi, sedi] = to_sav['RMSE_1'] 
                    file.close()
                    
                    
                    ##2nd augmentation
                    augmentation_num = 1
                    #f_est_after_%d_augm_n_eval.dat"%augmentation_num
                    filename = save_folder+save_dir+"f_est_after_%d_augm_n_eval.dat"%augmentation_num
                    file = open(filename,'rb')
                    to_sav = pickle.load(file)
                    
                    
                    mu_2 = to_sav['mu_1'] 
                    LL2[Ti, obi, gi, sedi] = to_sav['ll'] 
                    Wm2[Ti, obi, gi, sedi] = np.mean(to_sav['Wm_1']) 
                    RMSES2[Ti, obi, gi, sedi] = to_sav['RMSE_1'] 
                    file.close()
                    
                    if False:
                    
                        fig= plt.figure(figsize=(19,11))
                        gs = gridspec.GridSpec(nrows=2, ncols=3, height_ratios=[1,1])
        
                        #  Varying density along a streamline
                        ax0 = fig.add_subplot(gs[0:1, 0])
                        #plt.subplot(1,3,1)
                        q1=plt.quiver(xt[0],xt[1], ztrue[0],ztrue[1], color='grey', linewidths=4,label='true')
                        q2 =plt.quiver(xt[0],xt[1], mu_init[0],mu_init[1], color='#bc1d41', alpha=0.65, linewidths=4,label='est')
                        plt.title('Estimated force field\nwith Gaussian likelihood',fontsize=20)
                        plt.xlabel('x',fontsize=20)
                        plt.ylabel('y',fontsize=20)
                        plt.locator_params(axis='y',nbins=5)
                        plt.legend(frameon = 0, bbox_transform=fig.transFigure, 
                                   bbox_to_anchor=(0.085, 0.735), fontsize=18)
        
        
                        ax0 = fig.add_subplot(gs[0:1, 1])
                        q1=plt.quiver(xt[0],xt[1], ztrue[0],ztrue[1], color='grey', linewidths=4,label='true')
                        q2 =plt.quiver(xt[0],xt[1], mu_1[0],mu_1[1], color='#bc1d41', alpha=0.65, linewidths=4,label='est')
                        plt.title('Force field after\nfirst augmentation',fontsize=20)
                        plt.xlabel('x',fontsize=20)
                        plt.ylabel('y',fontsize=20)
                        plt.locator_params(axis='y',nbins=5)
                        # plt.legend(frameon = 0, bbox_transform=fig.transFigure, 
                        #            bbox_to_anchor=(0.085, 0.735), fontsize='large')
        
                        ax0 = fig.add_subplot(gs[0:1, 2])
                        q1=plt.quiver(xt[0],xt[1], ztrue[0],ztrue[1], color='grey', linewidths=4,label='true')
                        q2 =plt.quiver(xt[0],xt[1], mu_2[0],mu_2[1], color='#bc1d41', alpha=0.65, linewidths=4,label='est')
                        plt.title('Force field after\nsecond augmentation',fontsize=20)
                        plt.xlabel('x',fontsize=20)
                        plt.ylabel('y',fontsize=20)
                        plt.locator_params(axis='y',nbins=5)
                        ##############################################################
                        ax0 = fig.add_subplot(gs[1:2, 0])               
        
                        data = {'type':['Gaussian', '1st augm.', '2nd augm.'],
                                'RMSE':[RMSES_init[Ti, obi, gi, sedi], 
                                        RMSES1[Ti, obi, gi, sedi], 
                                        RMSES2[Ti, obi, gi, sedi]]}
                        df=pd.DataFrame(data)
                        sns.barplot(x = 'type',y = 'RMSE',data = df,palette="deep",)
                        plt.xlabel('')
                        plt.ylabel('RMSE', fontsize=20)
                        
                        ##############################################################
                        
                        ax0 = fig.add_subplot(gs[1:2, 1])               
        
                        data = {'type':['Gaussian', '1st augm.', '2nd augm.'],
                                'RMSE':[wRMSES_init[Ti, obi, gi, sedi], 
                                        wRMSES1[Ti, obi, gi, sedi], 
                                        wRMSES2[Ti, obi, gi, sedi]]}
                        df=pd.DataFrame(data)
                        sns.barplot(x = 'type',y = 'RMSE',data = df,palette="deep",)
                        plt.xlabel('')
                        plt.ylabel('wRMSE', fontsize=20)
                        
                        ##############################################################
                        
                        # ax1 = fig.add_subplot(gs[1:2, 1])               
        
                        # data = {'type':['Gaussian', '1st augm.', '2nd augm.'],
                        #         'Wm':[Wm_init[Ti, obi, gi, sedi], 
                        #                 Wm1[Ti, obi, gi, sedi], 
                        #                 Wm2[Ti, obi, gi, sedi]]}
                        # df=pd.DataFrame(data)
                        # sns.barplot(x = 'type',y = 'Wm',data = df,palette="deep",)
                        # plt.xlabel('')
                        # plt.ylabel('$\mathcal{W}$', fontsize=20)
                        
                        ##############################################################
                        ax2 = fig.add_subplot(gs[1:2, 2])               
        
                       
                        ax2.plot([0,1,2], [LL_init[Ti, obi, gi, sedi],
                                           -LL1[Ti, obi, gi, sedi],
                                           -LL2[Ti, obi, gi, sedi] ], lw=2, marker='o')
                        plt.xlabel('augmentation num.', fontsize=20)
                        plt.ylabel('neg. log. lik.', fontsize=20)
                        
                        
        
                        plt.subplots_adjust(wspace=0.35, hspace=0.45)
        
                        plt.savefig(save_folder+'After_first_'+save_dir[:-1]+'.png', bbox_inches='tight', transparent='False',  facecolor='white')
                        plt.savefig(save_folder+'After_first_'+save_dir[:-1]+'.pdf', bbox_inches='tight', transparent='False',  facecolor='white')
                        plt.close()
                    
                    
                except EOFError:
                    print('Not found: '+ save_dir, augmentation_num)
                    
                    
                except FileNotFoundError:
                    print('Not found: '+ save_dir, augmentation_num)
                    
                
                
#%%

Ts = [500, 1000, 2000]
obs_denz = [50,200, 240, 280]#,  320]#, 50]
gss = [0.25, 0.5, 0.75, 1.]

Ti = 0
T = Ts[Ti]

obi = 1
obs_dens = obs_denz[obi]

gi = 1
g = gss[gi]

sedi = 0
seed = seeds[sedi]

save_dir = 'FW_2Geodesic_'+dsystem+'_nose_%.2f_dens_%d_time_%d_seed_%d\\'%(g, obs_dens, T, seed)

pal1 = sns.cubehelix_palette(as_cmap=True)  
filename = save_folder+save_dir+"Observations.dat"
file = open(filename,'rb')
to_sav = pickle.load(file)
Obs = to_sav['Ob'] 

params = {"bandwidth": np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(Obs.reshape(dim, -1).T)
print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_
weights = np.exp(kde.score_samples(xt.reshape(dim, -1).T))

def scale(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))

weightsa = scale(weights)

filename = save_folder+save_dir+"Observations_true_f_and_sim_details.dat"
file = open(filename,'rb')
to_sav = pickle.load(file)
ztrue = to_sav['ztrue'] 
xt = to_sav['xt'] 
file.close()

filename = save_folder+save_dir+"Init_f_est.dat"
file = open(filename,'rb')
to_sav = pickle.load(file)

mu_init = to_sav['mu_init'] 
# RMSES_init[Ti, obi, gi, sedi] = to_sav['RMSE_init']

# Wm_init[Ti, obi, gi, sedi] = np.mean(to_sav['Wm_init'])
# LL_init[Ti, obi, gi, sedi] = to_sav['ll_init']
file.close()
##1st augmentation
augmentation_num = 0
#f_est_after_%d_augm_n_eval.dat"%augmentation_num
filename = save_folder+save_dir+"f_est_after_%d_augm_n_eval.dat"%augmentation_num
file = open(filename,'rb')
to_sav = pickle.load(file)

mu_1 = to_sav['mu_1'] 
# LL1[Ti, obi, gi, sedi] = to_sav['ll'] 
# Wm1[Ti, obi, gi, sedi] = np.mean(to_sav['Wm_1'] )
# RMSES1[Ti, obi, gi, sedi] = to_sav['RMSE_1'] 
file.close()


##2nd augmentation
augmentation_num = 1
#f_est_after_%d_augm_n_eval.dat"%augmentation_num
filename = save_folder+save_dir+"f_est_after_%d_augm_n_eval.dat"%augmentation_num
file = open(filename,'rb')
to_sav = pickle.load(file)

mu_2 = to_sav['mu_1'] 
# LL2[Ti, obi, gi, sedi] = to_sav['ll'] 
# Wm2[Ti, obi, gi, sedi] = np.mean(to_sav['Wm_1']) 
# RMSES2[Ti, obi, gi, sedi] = to_sav['RMSE_1'] 
file.close()



fig= plt.figure(figsize=(19,8))
gs = gridspec.GridSpec(nrows=3, ncols=3, height_ratios=[1,1,1])

#  Varying density along a streamline
ax0 = fig.add_subplot(gs[0:2, 0])
#plt.subplot(1,3,1)
q1=plt.quiver(xt[0],xt[1], ztrue[0],ztrue[1], color='grey', linewidths=0.5,label='true', edgecolors='grey')
q2 =plt.quiver(xt[0],xt[1], mu_init[0],mu_init[1], color='#bc1d41', alpha=0.65, linewidths=0.5,label='est', edgecolors='#bc1d41')
plt.title('Estimated force field\nwith Gaussian likelihood',fontsize=20)
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20)
plt.locator_params(axis='y',nbins=5)
plt.legend(frameon = 0, bbox_transform=fig.transFigure, 
           bbox_to_anchor=(0.085, 0.735), fontsize=18)

#ax0.set_facecolor("#5a5a5a")


ax1 = fig.add_axes([0.075, 0.44, 0.08, 0.15])
plt.plot( [0,3.2], [0, 3.2],'-', c='#4f4949')
ax1.patch.set_facecolor('white')
ax1.patch.set_alpha(0.85)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel(r'true $\theta$')
ax1.set_ylabel(r'est $\theta$')
oness = np.zeros_like(ztrue)
oness[1] = 1
angles_z =  compute_the_angles(ztrue, oness).reshape(-1)
angles_mu =  compute_the_angles(mu_init, oness).reshape(-1)
plt.plot(angles_z, angles_mu,'.', c=colors[0])#, alpha=weightsa)

ax0 = fig.add_subplot(gs[0:2, 1])
q1=plt.quiver(xt[0],xt[1], ztrue[0],ztrue[1], color='grey', linewidths=0.5,label='true', edgecolors='grey')
q2 =plt.quiver(xt[0],xt[1], mu_1[0],mu_1[1], color='#bc1d41', alpha=0.65, linewidths=0.5,label='est', edgecolors='#bc1d41')
plt.title('Force field after\nfirst augmentation',fontsize=20)
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20)
plt.locator_params(axis='y',nbins=5)
# plt.legend(frameon = 0, bbox_transform=fig.transFigure, 
#            bbox_to_anchor=(0.085, 0.735), fontsize='large')

ax1 = fig.add_axes([0.356, 0.44, 0.08, 0.15])
plt.plot( [0,3.2], [0, 3.2],'-', c='#4f4949')
ax1.patch.set_facecolor('white')
ax1.patch.set_alpha(0.85)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel(r'true $\theta$')
ax1.set_ylabel(r'est $\theta$')
oness = np.zeros_like(ztrue)
oness[1] = 1
angles_z =  compute_the_angles(ztrue, oness)
angles_mu =  compute_the_angles(mu_1, oness)
plt.plot(angles_z, angles_mu,'.', c=colors[1])#, alpha=weightsa)

ax0 = fig.add_subplot(gs[0:2, 2])
q1=plt.quiver(xt[0],xt[1], ztrue[0],ztrue[1], color='grey', linewidths=0.5,label='true', edgecolors='grey')
q2 =plt.quiver(xt[0],xt[1], mu_2[0],mu_2[1], color='#bc1d41', alpha=0.65, linewidths=0.5,label='est', edgecolors='#bc1d41')
plt.title('Force field after\nsecond augmentation',fontsize=20)
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20)
plt.locator_params(axis='y',nbins=5)

ax1 = fig.add_axes([0.641, 0.44, 0.08, 0.15])
plt.plot( [0,3.2], [0, 3.2],'-', c='#4f4949')
ax1.patch.set_facecolor('white')
ax1.patch.set_alpha(0.85)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel(r'true $\theta$')
ax1.set_ylabel(r'est $\theta$')
oness = np.zeros_like(ztrue)
oness[1] = 1
angles_z =  compute_the_angles(ztrue, oness)
angles_mu =  compute_the_angles(mu_2, oness)
plt.plot(angles_z, angles_mu,'.', c=colors[2])#, alpha=weightsa)
##############################################################
ax0 = fig.add_subplot(gs[2:3, 1])               

data = {'type':['Gaussian', '1st augm.', '2nd augm.'],
        'RMSE':[wRMSES_init[Ti, obi, gi, sedi], 
                wRMSES1[Ti, obi, gi, sedi], 
                wRMSES2[Ti, obi, gi, sedi]]}
df=pd.DataFrame(data)
sns.barplot(x = 'type',y = 'RMSE',data = df,palette ="deep",)
plt.xlabel('')
plt.ylabel('wRMSE', fontsize=20)

##############################################################
ax2 = fig.add_subplot(gs[2:3, 0])               
pal2 = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
# ax2.plot([0,1,2], [LL_init[Ti, obi, gi, sedi],
#                    -LL1[Ti, obi, gi, sedi],
#                    -LL2[Ti, obi, gi, sedi] ], lw=3, marker='o',c=pal2(0.7))
# plt.xlabel('augmentation num.', fontsize=20)
# plt.ylabel('neg. log. lik.', fontsize=20)
# ax2.set_xticks([0,1,2])


plt.plot( obs_denz[1:], np.nanmean(wRMSES_init[0,1:, 0], axis=-1),'o',linestyle='--', label='gaus',lw=2.5, c=pal1(0.1),markersize=8)

plt.plot( obs_denz[1:], np.nanmean(wRMSES1[0,1:, 0], axis=-1),'v',linestyle='-.', label='1st',lw=2.5, c=pal1(0.4),markersize=8)

plt.plot( obs_denz[1:], np.nanmean(wRMSES2[0,1:, 0], axis=-1),'s', linestyle='-', label='2nd',lw=2.5, c=pal1(0.8),markersize=8)

plt.plot( obs_denz[1:], np.nanmean(wRMSES_init[0,1:, 1], axis=-1),'o',linestyle='--', label='gaus',lw=2.5, c=pal2(0.1),markersize=8)

plt.plot( obs_denz[1:], np.nanmean(wRMSES1[0,1:, 1], axis=-1),'v',linestyle='-.', label='1st',lw=2.5, c=pal2(0.4),markersize=8)

plt.plot( obs_denz[1:], np.nanmean(wRMSES2[0,1:, 1], axis=-1),'s',linestyle='-', label='2nd',lw=2.5, c=pal2(0.8),markersize=8)

plt.xticks([200,240,280])

plt.xlabel(r'inter-observation interval $\tau$', fontsize=20)
plt.ylabel('wRMSE', fontsize=20)
leg = plt.legend(frameon = 0, bbox_transform=fig.transFigure,loc=3,
           bbox_to_anchor=(-0.025, 0.105), fontsize='x-large', ncol=2, handlelength=0.35)
leg.set_title('      noise    \n 0.25     0.5', prop = {'size':18})
##############################################################
ax1 = fig.add_subplot(gs[2:3, 2])               

   
# plt.plot( Ts, np.mean(RMSES_init[:,1, 0], axis=-1), label='init-0.25')
# plt.plot( Ts, np.mean(RMSES_init[:,1, 1], axis=-1), label='init-0.50')
# plt.plot( Ts, np.mean(RMSES_init[:,1, 2], axis=-1), label='init-0.75')

# #%%

plt.plot( gss[:-1], np.nanmean(wRMSES_init[0,1, :-1], axis=-1),'o', label='gaus',lw=2.5, c=pal1(0.1),linestyle='--',markersize=8)

plt.plot( gss[:-1], np.nanmean(wRMSES1[0,1, :-1], axis=-1),'v', label='1st',lw=2.5, c=pal1(0.4),linestyle='-.',markersize=8)

plt.plot( gss[:-1], np.nanmean(wRMSES2[0,1, :-1], axis=-1),'s', label='2nd',lw=2.5, c=pal1(0.8),linestyle='-',markersize=8)


plt.plot( gss[:-1], np.nanmean(wRMSES_init[1,1, :-1], axis=-1),'o',linestyle='--', label='gaus',lw=2.5, c=pal2(0.1),markersize=8)
plt.plot( np.array(gss[:-1])+0.01, np.nanmean(wRMSES1[1,1, :-1], axis=-1),'v',linestyle='-.',label='1st',lw=2.5, c=pal2(0.4),markersize=8)
plt.plot( np.array(gss[:-1])+0.02, np.nanmean(wRMSES2[1,1, :-1], axis=-1),'s',linestyle='-', label='2nd',lw=2.5, c=pal2(0.8),markersize=8)
#ax.set_yticklabels(ax.get_yticks(), font1)
ax=plt.gca()
ax.set_xticks(gss[:-1])
#plt.locator_params(axis='x', nbins=6)



plt.xlabel(r'noise amplitude $\sigma$', fontsize=20)
plt.ylabel('wRMSE', fontsize=20)
leg = plt.legend(frameon = 0, bbox_transform=fig.transFigure,
           bbox_to_anchor=(1.01, 0.335), fontsize='x-large', ncol=2, handlelength=0.35)
leg.set_title('        T    \n 500     1000', prop = {'size':18})
##%%
plt.subplots_adjust(wspace=0.35, hspace=0.45)


plt.savefig(save_folder+'After_first_'+save_dir[:-1]+'.png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')#, transparent='False'  )
plt.savefig(save_folder+'After_first_'+save_dir[:-1]+'.pdf', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')#, transparent='False')
#plt.close()
                    
                    

