import numpy as np
from numpy.random import exponential as exp
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.ndimage import gaussian_filter1d as smooth
import sys
import json

import covid_funs as cf

def SIR_model(t,X,params):
    s,i,r= X
    beta,recrate = params
    if callable(beta):
        dsdt = -(beta(s,i,r,t))*s*i
    else:
        dsdt = -beta*s*i
    didt = -dsdt - recrate*i
    drdt = recrate*i
    return [dsdt,didt,drdt]


if len(sys.argv) > 1:
    Bias = float(sys.argv[1])
else:
    Bias = 1


timescale = 15 #1/gamma





init_inf = 0.01

s0 = (1-init_inf)
i0 = init_inf
r0 = 0

R0 = 2.2




t0 = 0
end_time = 150
tst = 0.01

sir_sol = ode(SIR_model)
sir_sol.set_f_params([R0/timescale,1/timescale])
sir_sol.set_initial_value([s0,i0,r0],0)

time_array = np.array([t0])
sol_list = [[s0,i0,r0]]
for tpt in np.arange(t0,end_time,tst):
    s,i,r = sir_sol.integrate(sir_sol.t + tst)
    sol_list = sol_list + [[s,i,r]]
    time_array = np.append(time_array,sir_sol.t)
sol_array = np.array(sol_list)

dynamic_map = {"TimePoints":list(time_array), "Symptomatic":list(sol_array[:,1]),"Asymptomatic":list(np.zeros(len(time_array))),"NonInfected":list(sol_array[:,0]+sol_array[:,2])}
biasarr = [Bias]*len(time_array)

capacity_map = {}
for i in [100,500,1000,2500,5000]:
    capacity_map[str(i)] = [i]*len(time_array)

with open("json_io/dynamics.json","w") as outfile:
    json.dump(dynamic_map, outfile)

with open("json_io/bias.json","w") as outfile:
    json.dump(biasarr,outfile)

with open("json_io/capacity.json","w") as outfile:
    json.dump(capacity_map,outfile)
