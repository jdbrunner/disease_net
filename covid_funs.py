import numpy as np
from scipy.integrate import cumtrapz
from scipy.integrate import ode
from numpy.random import exponential as exp
from numpy.random import rand
# from scipy.ndimage import gaussian_filter1d as smooth
import json



import matplotlib.pyplot as plt

def gen_jsons(end_time,initial_condtions,dynamics,symptomaticVar,asymptomaticVar,noninfectedVar,params,Bias,maxcapacity,capacityfun = False,tst = 0.01):

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


    with open("json_io/dynamics.json","w") as outfile:
        json.dump(dynamic_map, outfile)

    with open("json_io/bias.json","w") as outfile:
        json.dump(biasarr,outfile)

    with open("json_io/capacity.json","w") as outfile:
        json.dump(capacity_map,outfile)


    return dynamic_map


def mysmooth(y,sigmasq,x = []):
    if len(x) != len(y):
        x = np.ones(len(y))
    smthd = np.empty(len(y))
    for i in range(len(x)):
        sm = 0
        for j in range(len(y)):
            sm += (y[j]/(np.sqrt(2*np.pi)*sigmasq))*np.exp(-((x[i]-x[j])**2)/(2*sigmasq))
        smthd[i] = sm
    return smthd


def cross_zero(arr):
    indx = []
    for i in range(1,len(arr)):
        if arr[i-1]>0 and arr[i]< 0:
            indx += [i]
    return indx

def findPeak_smth(real_dynamics,sample,smoothing = 2):
    total_infected = np.array(real_dynamics['Symptomatic']) + np.array(real_dynamics['Asymptomatic'])
    realDiff = np.diff(total_infected)/np.diff(real_dynamics['TimePoints'])

    real_peaks = cross_zero(realDiff)
    real_peak_vals = np.array(real_dynamics['TimePoints'])[real_peaks]
    rlpk_map = {}
    for i in range(len(real_peaks)):
        rlpk_map["R"+str(i)] = real_dynamics['TimePoints'][real_peaks[i]]

    sample_ratio = np.array(sample["DailyPositive"])/np.array(sample["DailyTotal"])
    sampleDiff =  np.diff(sample_ratio)/np.diff(sample["DayTimes"])

    dataPeakIndx = []
    foundOnIndx = []
    for i in range(21,len(sample_ratio)):
        tempPk = cross_zero(mysmooth(sampleDiff[:i-1],smoothing,x = sample["DayTimes"][1:i]))
        if len(tempPk) > 0:
            dataPeakIndx += [tempPk[-1] - 1]
            foundOnIndx += [i]

    peaks_found = np.array(sample["DayTimes"])[dataPeakIndx]
    found_on_day = np.array(sample["DayTimes"])[foundOnIndx]


    return real_peak_vals,rlpk_map,peaks_found,found_on_day


def judge_peaks(method,real_dynamics,sample,tol = 2,**kwargs):

    real_peak_vals,rlpk_map,peaks_found,found_on_day = method(real_dynamics,sample,**kwargs)

    recall = {}
    for ky,val in rlpk_map.items():
        mp = {"Peak":val,"Found":False,"On":[],"As":[]}
        sqdist = (peaks_found - val)**2
        didfind = np.where(sqdist < tol**2)
        if len(didfind[0]):
            mp["Found"] = True
            mp["On"] = found_on_day[didfind]
            mp["As"] = peaks_found[didfind]
        recall[ky] = mp

    precision = {}
    for i in range(len(peaks_found)):
        val = peaks_found[i]
        prv_found = [ky for ky in precision.keys() if precision[ky]["Peak"] == val]
        if len(prv_found) == 0:
            mp = {"Peak":val,"Real":False,"FoundOn":[found_on_day[i]]}
            sqdist = (real_peak_vals - val)**2
            didfind = np.where(sqdist < tol**2)
            mp["SqDist"] = min(sqdist)
            if len(didfind[0]):
                mp["Real"] = True
            precision["D" + str(i)] = mp
        else:
            precision[prv_found[0]]["FoundOn"] += [found_on_day[i]]

    return recall,precision
