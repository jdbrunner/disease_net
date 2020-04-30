import numpy as np
from numpy.random import exponential as exp
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.ndimage import gaussian_filter1d as smooth
import sys

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


deriv = np.diff(sol_array[:,1])/np.diff(time_array)#np.array([SIR_model(0,X,[R0/timescale,1/timescale])[1] for X in sol_array])
test_times, test_results = cf.sim_sampling_stochastic(sol_array, time_array, 100, 1)


total_tests = np.sum(test_results,axis = 1)


perday,daytimes = cf.count_perday(test_times,test_results[:,0], interval = 1)
perdaytotal,daytimestotal = cf.count_perday(test_times,total_tests, interval = 1)



for LambdaMax in [100,500,1000]:#,5000,10000]:

    print(cf.assess_peakfind(cf.peak_quadfit,LambdaMax,(time_array,sol_array),[20]))
    print(cf.assess_peakfind(cf.peak_quadfit2,LambdaMax,(time_array,sol_array),[20]))
    print(cf.assess_peakfind(cf.peak_deriv,LambdaMax,(time_array,sol_array),[5]))


    # test_times, test_results = cf.sim_sampling_stochastic(sol_array, time_array, LambdaMax, Bias)
    #
    #
    # total_tests = np.sum(test_results,axis = 1)
    #
    # interval_size = 1
    #
    # perday,daytimes = cf.count_perday(test_times,test_results[:,0], interval = interval_size)
    # perdaytotal,daytimestotal = cf.count_perday(test_times,total_tests, interval = interval_size)
    #
    #
    # propotion_per_day = perday/np.maximum(1,perdaytotal)
    # #
    # # cases_jtimes = np.array([cf.linear_interp(t,time_array,cases) for t in test_times])
    #
    #
    # window_size = 20
    #
    #
    # peak_times2 = peak_quadfit(perday,daytimes,window_size)
    #
    # fig,ax = plt.subplots(figsize = (10,5))
    # ax.plot(time_array,LambdaMax*sol_array[:,1], label = "Real Cases Per Test", color = "r")
    # ax.bar(daytimes,perday,label = "Positive Propotion Per Day") #Can estimate
    # # ax.plot(timescale*peak_times,max(sol_array[:,1])*np.ones(len(peak_times)),"x", color = "g")
    # ax.plot(peak_times2,max(perday)*np.ones(len(peak_times2)),"x", color = "g",label ="Detected Peak")
    #
    # ax.legend()
    # ax.set_title("Peak Detection, " + str(LambdaMax) + " tests per day, Bias = " + str(Bias))
    #
    # fig.savefig("pk_" + str(LambdaMax))
    # # plt.show()
