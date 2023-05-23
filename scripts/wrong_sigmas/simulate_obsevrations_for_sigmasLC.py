# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:05:46 2022

@author: maout
"""


##simulate only observations for evaluation


import numpy as np
import pickle

dsystem = 'LC'

if dsystem == 'LC':
    def f(x, t=0,mu=1):#Van der Pol oscillator
        x0 = mu*(x[0] - (1/3)*x[0]**3-x[1])
        x1 = (1/mu)*x[0]
        return np.array([x0,x1])
    x0 = np.array([1.81, -1.41])
    dim = 2
    
    
    
    
    
def simulate_path(f_drift, timegrid):
    
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
          F[:,ti] = F[:,ti-1]+ h* f_drift(F[:,ti-1].reshape(-1,1)).T+(g)*np.random.normal(loc = 0.0,
                                                                    scale = np.sqrt(h),
                                                                    size=(dim,))
      
  return F


save_folder = 'C:\\Users\\maout\\results_LC_single_instance\\wrong_sigmas5\\'

#T = Ts[Ti]  #length of simulation
#obs_dens = obs_denz[obi] #interval between successive observations
#g = gss[gi] #noise amplitude
import os

#Ts = [500, 1000, 2000]
#obs_denz = [50,200, 240, 280,  320, 50]
#gss = [0.25, 0.5, 0.75, 1.]

Ts = [500, 1000]#, 1500]
obs_denz = [160, 200, 240, 280]#[80,120,160,200, 240, 280,  320]
gss = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9, 1.]

seeds = [13, 14]
seeds = [16,17,18,19]

for seedi, seed in enumerate(seeds):
    for Ti,T in enumerate(Ts):
        for obi, obs_dens in enumerate(obs_denz):
            for gi, gest in enumerate(gss):
                

                np.random.seed(seed)
                g = 0.5
                h = 0.01 #sim_prec
                dt = h
                t_start = 0.
                D = g**2
                
                save_dir = 'FW_2Geodesic_'+dsystem+'_nose_%.2f_dens_%d_time_%d_seed_%d\\'%(gest, obs_dens, T, seed)
                #print(save_dir)
                if os.path.exists(save_folder+save_dir) == False:  
                    os.mkdir(save_folder+save_dir)  
                    
                timegrid = np.arange(0,T,h)
                timegrid_obs = timegrid[::obs_dens]
                
                F = simulate_path(f, timegrid)
                
                ##get observations
                Ob = F[:,::obs_dens]
                
                filehandler = open(save_folder+save_dir+"Observations.dat","wb")
                to_sav = dict()    
                to_sav['Ob'] = Ob                
                pickle.dump(to_sav,filehandler)