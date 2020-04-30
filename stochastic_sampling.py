import numpy as np
from numpy.random import exponential as exp
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.ndimage import gaussian_filter1d as smooth
import sys

import covid_funs as cf

def SIR_model(t,X,R0):
    s,i,r= X
    if callable(R0):
        dsdt = -R0(s,i,r,t)*s*i
    else:
        dsdt = -R0*s*i
    didt = -dsdt - i
    drdt = i
    return [dsdt,didt,drdt]

def Bias2(t):
    if t > 0.2:
        return 1 + 3*(1/(5*t))
    else:
        return 3

if len(sys.argv) > 1:
    Bias = float(sys.argv[1])
else:
    Bias = 1

N = 2000
LambdaMax = 100

def increase_lambda(t,l1 = 10, l2 = 90):
    return l1 + (l2)*(t/(2+t))


init_inf = 0.01

s0 = (1-init_inf)
i0 = init_inf
r0 = 0

R0 = 1.5



t0 = 0
end_time = 10
tst = 0.01

sir_sol = ode(SIR_model)
sir_sol.set_f_params(R0)
sir_sol.set_initial_value([s0,i0,r0],0)

time_array = np.array([t0])
sol_list = [[s0,i0,r0]]
for tpt in np.arange(t0,end_time,tst):
    s,i,r = sir_sol.integrate(sir_sol.t + tst)
    sol_list = sol_list + [[s,i,r]]
    time_array = np.append(time_array,sir_sol.t)
sol_array = np.array(sol_list)

cases = s0-sol_array.T[0] + i0

test_times, test_results = cf.sim_sampling_stochastic(sol_array, time_array, increase_lambda, Bias2)

total_tests = np.sum(test_results,axis = 1)

cases_jtimes = np.array([cf.linear_interp(t,time_array,cases) for t in test_times])

fig,ax = plt.subplots(figsize = (10,5))
ax.step(test_times,test_results.T[0]/(N*cases_jtimes),label = "Total positive tests per infected") ### Cannot estimate
ax.step(test_times,test_results.T[0]/N, label = "Total positive Tests per capita")### Can estimate
ax.legend()
ax.set_title("Total positive tests, L=" + str(LambdaMax) + ", N=" + str(N))


positive_proportion = test_results.T[0]/np.maximum(1,total_tests)

fig,ax = plt.subplots(figsize = (10,5))
ax.step(test_times,positive_proportion,label = "Total positive tests per tests performed") #Can estimate
ax.plot(time_array,cases, label = "Total cases per capita")## Cannot estimate
ax.legend()
ax.set_title("Infection Propotion, L=" + str(LambdaMax) + ", N=" + str(N))



smth_positive = smooth(test_results.T[0].astype(float),20)
smth_pos_delta = [0] + [smth_positive[i] - smth_positive[i-1] for i in range(1,len(test_times))]

fig,ax = plt.subplots(figsize = (10,5))
ax.step(test_times,smth_positive,label = "Rate of positive tests") #Can estimate
ax.legend()
ax.set_title("Positive Tests")


fig,ax = plt.subplots(figsize = (10,5))
ax.step(test_times,smth_pos_delta,label = "Rate of positive tests per tests performed") #Can estimate
ax.plot(time_array,sol_array[:,1], label = "Active cases per capita") ## Cannot estimate
ax.legend()
ax.set_title("Instantaneous Cases")

plt.show()
