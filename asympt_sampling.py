import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode,cumtrapz
import covid_funs as cf

import sys

#timescale of gamma2 (recovery of symnptomatic)

def SIR_model_asympt(t,X,params):
    R01f,R02f,asympt,g,bet = params
    if callable(R01f):
        R01 = R01f(X,t)
    else:
        R01 = R01f
    if callable(R02f):
        R02 = R02f(X,t)
    else:
        R02 = R02f
    r01 = R01*asympt
    r02 = R01*(1-asympt)
    r03 = R02*asympt
    r04 = R02*(1-asympt)
    s,i1,i2,r= X
    dsdt = -(r01+r02)*s*i1 - (r03 + r04)*s*i2
    di1dt = r01*s*i1 + r03*s*i2 - g*i1 - bet*i1
    di2dt = r02*s*i1 + r04*s*i2 - i2 + bet*i1
    drdt = g*i1 + i2
    return [dsdt,di1dt,di2dt,drdt]



N = 20000
LambdaMax = 50
init_inf = 0.01

if len(sys.argv) > 1:
    Bias = float(sys.argv[1])
else:
    Bias = 1

# if len(sys.argv) > 2:
#     R0 = float(sys.argv[2])
# else:
#     R0 = 1.5

# R01 = 2
R02 = 1.1

def R01(X,t):
    s,i1,i2,r = X
    return 2 - i2



asympt_chance = 0.8

s0 = (1-init_inf)
i10 = init_inf*asympt_chance
i20 = init_inf*(1-asympt_chance)
r0 = 0


lam0 = LambdaMax/N

t0 = 0
end_time = 30
tst = 0.01


no_sympts = 0.2 #rate of recovery of asymptomatic/ rate of recovery of symptomatic
become_sympts = 0.8 #rate of transition to symptomatic/rate of recovery of symptomatic
#no real need for these to sum to 1.

sir_sol = ode(SIR_model_asympt)
sir_sol.set_f_params([R01,R02,asympt_chance,no_sympts,become_sympts])
sir_sol.set_initial_value([s0,i10,i20,r0],0)

time_array = np.array([t0])
sol_list = [[s0,i10,i20,r0]]
for tpt in np.arange(t0,end_time,tst):
    s,i1,i2,r = sir_sol.integrate(sir_sol.t + tst)
    sol_list = sol_list + [[s,i1,i2,r]]
    time_array = np.append(time_array,sir_sol.t)
sol_array = np.array(sol_list)



#Total cases
cases = s0-sol_array.T[0] + i10 + i20

fig,ax = plt.subplots(figsize = (10,5))
ax.plot(sol_array[:,1],label = "Asymptomatic") ### Cannot estimate
ax.plot(sol_array[:,2], label = "Symptomatic")### Cannot estimate
ax.plot(sol_array[:,1] + sol_array[:,2], label = "Total")### Cannot estimate
ax.legend()
ax.set_title("Disease Dynamics, R01 = " + str(R01) + ", R02 = " + str(R02))



sample_dict = cf.sim_sampling_deterministic(sol_array, time_array, lam0, Bias,infected_coord = [1,2], symptomatic_coord = 2)


#Total tests run (at rate lambda*t)
total_tests = sample_dict["TotalTests"]
total_tests2 = total_tests.copy()
total_tests2[0] = 1

#biased sampling
positive_rate = sample_dict["PositiveRate"]#lam0*Bias*sol_array.T[1]/(sol_array.T[0] + Bias*sol_array.T[1] + sol_array.T[2])
#biased sampling (integrated)
total_positve = sample_dict["TotalPositive"]#cumtrapz(positive_rate,time_array,initial = 0)


fig,ax = plt.subplots(figsize = (10,5))
ax.plot(time_array,total_positve/(cases),label = "Total positive tests per infected") ### Cannot estimate
ax.plot(time_array,total_positve, label = "Total positive Tests per capita")### Can estimate
ax.legend()
ax.set_title("Total positive tests, L=" + str(LambdaMax) + ", N=" + str(N))


positive_proportion = sample_dict["TotalPositiveProportion"]#total_positve/total_tests2

fig,ax = plt.subplots(figsize = (10,5))
ax.plot(time_array[1:],positive_proportion[1:],label = "Total positive tests per tests performed") #Can estimate
ax.plot(time_array,cases, label = "Total cases per capita")## Cannot estimate
ax.legend()
ax.set_title("Infection Propotion, L=" + str(LambdaMax) + ", N=" + str(N))


positive_prop_rate = sample_dict["PositiveProportionRate"]#positive_rate/lam0


fig,ax = plt.subplots(figsize = (10,5))
ax.plot(time_array,positive_prop_rate,label = "Positive test proportion (instant)") ## Can estimate
ax.plot(time_array,sol_array.T[1] + sol_array.T[2], label = "Infection Propotion (instant)") ## Cannot estimate
ax.legend()
ax.set_title("Instantaneous Cases")

plt.show()
