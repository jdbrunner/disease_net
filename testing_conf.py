import numpy as np
import json
import matplotlib.pyplot as plt
import covid_funs
import importlib
import subprocess
import sys
import shlex
import os


fldername = "BiasSIRPop1000"
try:
    os.mkdir(fldername)
except:
    None





Bias = 10 #Bias = 1 means no bias. Higher puts bias towards testing symptomatic individuals.
capacities = [100,300,900,2700,8100]



timescale = 15 #1/gamma


init_inf = 0.01

s0 = (1-init_inf)
i0 = init_inf
r0 = 0

R0 = 2.2

end_time = 100

# time = np.arange(0,end_time,0.01)
# Symptomatic = 0.1*np.sin(time/5) + 0.2
# Asymptomatic = np.zeros(len(time))
# NonInfected = 1 - Symptomatic


# dynamics = {"TimePoints":list(time), "Symptomatic":list(Symptomatic),"Asymptomatic":list(Asymptomatic),"NonInfected":list(NonInfected)}

dynamics = covid_funs.gen_dynamics(end_time,[s0,i0,r0],covid_funs.SIR_model,1,-1,[0,2],[R0/timescale,1/timescale])

covid_funs.gen_jsons(fldername,dynamics,Bias,capacities)

total_infected = np.array(dynamics['Symptomatic']) + np.array(dynamics['Asymptomatic'])

svfl = fldername+"/testresults.json"
dynamicsfl = fldername+"/dynamics.json"
biasfl = fldername+"/bias.json"
capfl = fldername+"/capacity.json"
num_trials = 100
falsePos = 0
falseNeg = 0
smth = 5
peak_tol = 3
popsize = 1000

base_command = "./disease_confidence"
opts = ["-Dynamics="+dynamicsfl]
opts +=["-TestingBias="+biasfl]
opts +=["-TestingCapacities="+capfl]
opts +=["-Trials="+str(num_trials)]
opts +=["-SaveFile="+svfl]
opts +=["-FalsePositive="+str(falsePos)]
opts +=["-FalseNegative="+str(falseNeg)]
opts +=["-Smoothing="+str(smth)]
opts +=["-PeakTol="+str(peak_tol)]
opts +=["-TotalPop="+str(popsize)]


full_command = base_command + " " + " ".join(opts)
so = covid_funs.run_command(full_command)

with open(svfl) as fl:
    results = json.load(fl)




pos_prop = {}
for ky in results["SimulatedData"]:
    smps = []
    times = []
    for sample in results["SimulatedData"][ky]:
        smps += [np.array(sample["DailyPositive"])/np.maximum(1,np.array(sample["DailyTotal"]))]
        times += [np.array(sample["DayTimes"])]
    pos_prop[ky] = (smps,times)


for ky in pos_prop.keys():
    fig,ax = plt.subplots(figsize = (10,5))
    ax.plot(dynamics["TimePoints"],total_infected, label = "Infection Proportion", color = 'red')
    ax.bar(pos_prop[ky][1][0],pos_prop[ky][0][0], label = "Positive Test Proportion")

    ax.legend()
    fig.savefig(fldername+"/"+ky)
    plt.close()





five_day_conf = {}
for ky in pos_prop.keys():
    five_day_conf[ky] = covid_funs.trendConfidence(pos_prop[ky][0],pos_prop[ky][1],total_infected,dynamics["TimePoints"],5)

with open(fldername+"/five_day_conf.json","w") as outfile:
    json.dump(five_day_conf, outfile)


recall = {}
for ky,val in results["Performance"].items():
    if sum([len(v["Recalls"]) for v in val]):
        recall[ky] = sum([pk["Found"] for dyn in val for  pk in dyn["Recalls"]])/sum([len(v["Recalls"]) for v in val])
    else:
        recall[ky] = 0

with open(fldername+"/recall.json","w") as outfile:
    json.dump(recall, outfile)

precision = {}
for ky,val in results["Performance"].items():
    if sum([len(v["Precisions"]) for v in val]):
        precision[ky] = sum([pk["Real"] for dyn in val for  pk in dyn["Precisions"]])/sum([len(v["Precisions"]) for v in val])
    else:
        precision[ky] = 0

with open(fldername+"/precision.json","w") as outfile:
    json.dump(precision, outfile)

mean_sq_error = {}
for ky,val in results["Performance"].items():
    if sum([len(v["Precisions"]) for v in val]):
        mean_sq_error[ky] =[(sum([(pk["SqDist"])**2 for pk in dyn["Precisions"]])/len(dyn["Precisions"]))**(1/2) for dyn in val]
    else:
        mean_sq_error[ky] = 0

with open(fldername+"/mean_sq_error.json","w") as outfile:
    json.dump(mean_sq_error, outfile)
