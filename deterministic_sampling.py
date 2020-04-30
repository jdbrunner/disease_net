import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode,cumtrapz
import covid_funs as cf

import sys


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




N = 20000
LambdaMax = 50
init_inf = 0.001

s0 = (1-init_inf)
i0 = init_inf
r0 = 0

if len(sys.argv) > 1:
    Bias = float(sys.argv[1])
else:
    Bias = 1

if len(sys.argv) > 2:
    R0 = float(sys.argv[2])
else:
    R0 = 1.5

#For double peak:
# def R0(s,i,r,t):
#     return 2+np.cos(np.pi*(t/5))

lam0 = LambdaMax/N

def increase_lambda(t,l1 = 0.1*lam0, l2 = 1.2*lam0):
    return l1 + (l2-l1)*(t/(2+t))

t0 = 0
end_time = 30
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



#Total cases
cases = s0-sol_array.T[0] + i0


fig,ax = plt.subplots(figsize = (10,5))
ax.plot(time_array,sol_array[:,0],label = "Suscepible (per capita)") ### Cannot estimate
ax.plot(time_array,sol_array[:,1], label = "Infected (per capita)")### Can estimate
ax.legend()
ax.set_title("Total positive tests, L=" + str(LambdaMax) + ", N=" + str(N))


sample_dict = cf.sim_sampling_deterministic(sol_array, time_array, increase_lambda, Bias2)
sample_dict2 = cf.sim_sampling_deterministic(sol_array, time_array, lam0, Bias2)


#Total tests run (at rate lambda*t)
total_tests = sample_dict["TotalTests"]
total_tests22 = sample_dict["TotalTests"]

total_tests2 = total_tests.copy()
total_tests2[0] = 1

#biased sampling
positive_rate = sample_dict["PositiveRate"]#lam0*Bias*sol_array.T[1]/(sol_array.T[0] + Bias*sol_array.T[1] + sol_array.T[2])
#biased sampling (integrated)
total_positve = sample_dict["TotalPositive"]#cumtrapz(positive_rate,time_array,initial = 0)


total_positve2 = sample_dict2["TotalPositive"]#cumtrapz(positive_rate,time_array,initial = 0)

fig,ax = plt.subplots(figsize = (10,5))
# ax.plot(time_array,total_tests,label = "Total tests per capita (constant lam0)") ### Cannot estimate
# ax.plot(time_array,total_tests22,label = "Total tests per capita (varying lam0)") ### Cannot estimate
ax.plot(time_array,total_positve2, label = "Total positive tests per capita (constant lam0)")### Can estimate
ax.plot(time_array,total_positve, label = "Total positive tests per capita (varying lam0)")### Can estimate
ax.legend()
ax.set_title("Total positive tests, L=" + str(LambdaMax) + ", N=" + str(N))



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
ax.plot(time_array,sol_array.T[1], label = "Infection Propotion (instant)") ## Cannot estimate
ax.legend()
ax.set_title("Instantaneous Cases")

plt.show()
