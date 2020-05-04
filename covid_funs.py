import numpy as np
from scipy.integrate import cumtrapz
from scipy.integrate import ode
from numpy.random import exponential as exp
from numpy.random import rand

import subprocess
import sys
import shlex
# from scipy.ndimage import gaussian_filter1d as smooth
import json



import matplotlib.pyplot as plt

def gen_dynamics(end_time,initial_condtions,dynamics,symptomaticVar,asymptomaticVar,noninfectedVar,params,tst = 0.01):

    sir_sol = ode(dynamics)
    sir_sol.set_f_params(params)
    sir_sol.set_initial_value(initial_condtions,0)

    time_array = np.array([0])
    sol_list = [list(initial_condtions)]
    for tpt in np.arange(0,end_time,tst):
        solt = list(sir_sol.integrate(sir_sol.t + tst))
        sol_list = sol_list + [solt]
        time_array = np.append(time_array,sir_sol.t)
    sol_array = np.array(sol_list)

    if not np.isscalar(symptomaticVar):
        Symptomatic = np.sum(sol_array[:,symptomaticVar],axis = 1)
    else:
        Symptomatic = sol_array[:,symptomaticVar]

    if not np.isscalar(asymptomaticVar):
        Asymptomatic = np.sum(sol_array[:,asymptomaticVar],axis = 1)
    else:
        if asymptomaticVar != -1:
            Asymptomatic = sol_array[:,asymptomaticVar]
        else:
            Asymptomatic = np.zeros(len(time_array))

    if not np.isscalar(noninfectedVar):
        NonInfected = np.sum(sol_array[:,noninfectedVar],axis = 1)
    else:
        NonInfected = sol_array[:,noninfectedVar]

    dynamic_map = {"TimePoints":list(time_array), "Symptomatic":list(Symptomatic),"Asymptomatic":list(Asymptomatic),"NonInfected":list(NonInfected)}

    return dynamic_map

def gen_jsons(folder,dynamic_map,Bias,maxcapacity,capacityfun = False):

    time_array = dynamic_map["TimePoints"]

    if callable(Bias):
        biasarr = [Bias(t) for t in time_array]
    elif np.isscalar(Bias):
        biasarr = [Bias]*len(time_array)
    else:
        biasarr = Bias

    if np.isscalar(maxcapacity):
        maxcapacity = [maxcapacity]

    if callable(capacityfun):
        capacity_map = {}
        for i in maxcapacity:
            capacity_map[str(i)] = [i]*[capacityfun(t) for t in time_array]
    else:
        capacity_map = {}
        for i in maxcapacity:
            capacity_map[str(i)] = [i]*len(time_array)


    with open(folder+"/dynamics.json","w") as outfile:
        json.dump(dynamic_map, outfile)

    with open(folder+"/bias.json","w") as outfile:
        json.dump(biasarr,outfile)

    with open(folder+"/capacity.json","w") as outfile:
        json.dump(capacity_map,outfile)

    return None



def fit_slope(full_y,window,*argv):

    y = full_y[window[0]:window[1]]

    if len(argv):
        x = argv[0][window[0]:window[1]]
    else:
        x = np.ones(len(y))


    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    return m

def find_trend(data,times,twindow):
    data = np.array(data)
    times = np.array(times)
    #get index
    if twindow[1] < times[-1]:
        indx1 = np.argwhere(twindow[0] <= times)[0,0]
        indx2 = np.argwhere(twindow[1] <= times)[0,0]
    elif twindow[0]<times[-1]:
        indx1 = np.argwhere(twindow[0] <= times)[0,0]
        indx2 = len(times)
    else:
        print("No data in window")
        return None

    trend = fit_slope(data,[indx1,indx2],times)

    return trend

def test_all_windows(data,times,windowsize,start = 0):
    window_trend = []
    window_ending = []

    t = start

    while t<times[-1]:
        window_trend += [find_trend(data,times,[t,t+windowsize])]
        window_ending += [t+windowsize]
        t += windowsize

    return np.array(window_trend),np.array(window_ending)

def trendError(realTrends,sampleTrends):
    realTrends = realTrends[:len(sampleTrends)]
    sq_error = (np.array(realTrends) - np.array(sampleTrends))**2
    rt_mn_sq_error =np.sqrt(sum(sq_error)/len(realTrends))
    prod = np.array(realTrends)*np.array(sampleTrends)
    same_sign_prob = sum([p > 0 for p in prod])/len(realTrends)
    return sq_error,rt_mn_sq_error,same_sign_prob

def trendConfidence(samplevals,sampletimes,realvals,realtimes,windowsize):
    real_trend,_ = test_all_windows(realvals,realtimes,windowsize)
    tot = 0
    for i in range(len(samplevals)):
        samp_trend,_ = test_all_windows(samplevals[i],sampletimes[i],windowsize)
        _,_,conf = trendError(real_trend,samp_trend)
        tot += conf
    return tot/len(samplevals)


#from https://www.endpoint.com/blog/2015/01/28/getting-realtime-output-using-python
def run_command(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr = subprocess.STDOUT, text = False)
    outlist = []
    while True:
        output = process.stdout.readline()
        outlist += [output]
        if output.decode("utf-8") == '' and process.poll() is not None:
            break
        if output:
            if '\r' in output.decode("utf-8"):

                sys.stdout.write('\r' + output.decode("utf-8").strip())
            else:
                sys.stdout.write(output.decode("utf-8"))
    rc = process.poll()
    return rc,outlist

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
