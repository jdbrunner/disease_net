import numpy as np
import json
import matplotlib.pyplot as plt
import covid_funs
import importlib
import subprocess
import sys
import shlex
import os

num_trials = sys.argv[1]#1000
Bias = sys.argv[2]#10 #Bias = 1 means no bias. Higher puts bias towards testing symptomatic individuals.

fldername = sys.argv[3]#"biasSIR"
try:
    os.mkdir(fldername)
except:
    None





capacities = [100,300,900,2700,8100]



timescale = 15 #1/gamma


init_inf = 0.01

s0 = (1-init_inf)
i0 = init_inf
r0 = 0

R0 = 2.2

end_time = 100

# time = np.arange(0,end_time,0.01)
# # Symptomatic = 0.1*np.sin(time/5) + 0.2
# # Asymptomatic = np.zeros(len(time))
# # NonInfected = 1 - Symptomatic
#
# slp = 5
#
# if slp >0:
#     total_infected = slp*time
# else:
#     total_infected = -slp*end_time + slp*time
#
# symptProp = 1
#
# Symptomatic = symptProp*total_infected
# Asymptomatic = (1-symptProp)*total_infected
# NonInfected = 1 - total_infected
#
#
# dynamics = {"TimePoints":list(time), "Symptomatic":list(Symptomatic),"Asymptomatic":list(Asymptomatic),"NonInfected":list(NonInfected)}

dynamics = covid_funs.gen_dynamics(end_time,[s0,i0,r0],covid_funs.SIR_model,1,-1,[0,2],[R0/timescale,1/timescale])

covid_funs.gen_jsons(fldername,dynamics,Bias,capacities)

total_infected = np.array(dynamics['Symptomatic']) + np.array(dynamics['Asymptomatic'])

svfl = fldername+"/testresults.json"
dynamicsfl = fldername+"/dynamics.json"
biasfl = fldername+"/bias.json"
capfl = fldername+"/capacity.json"
falsePos = 0.1
falseNeg = 0.1
smth = 5
peak_tol = 3
# popsize = 1000

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
    ax.set_xlabel("Time")
    ax.legend()
    fig.savefig(fldername+"/"+ky)
    plt.close()





five_day_conf = {}
five_day_OverTime = {}
five_day_err = {}
for ky in pos_prop.keys():
    five_day_conf[ky],five_day_OverTime[ky],five_day_err[ky] = covid_funs.trendConfidence(pos_prop[ky][0],pos_prop[ky][1],total_infected,dynamics["TimePoints"],5)

with open(fldername+"/five_day_conf.json","w") as outfile:
    json.dump(five_day_conf, outfile)

# with open(fldername+"/five_day_conf_overtime.json","w") as outfile:
#     json.dump(five_day_OverTime, outfile)

fig,ax = plt.subplots(2,1,figsize = (10,10))
for ky in pos_prop.keys():
    ax[0].plot(five_day_OverTime[ky][1],five_day_OverTime[ky][0], label = "Samples/Day:" + str(ky))
    ax[1].plot(five_day_err[ky][1],five_day_err[ky][0], label = "Samples/Day:" + str(ky))



ax[0].set_ylabel("Five-day confidence")
ax[0].set_xlabel("Time")
ax[0].legend()
ax[1].set_ylabel("Mean-Square Error of Estimated Slope")
ax[1].set_xlabel("Time")
ax[1].legend()
fig.savefig(fldername+"/5Dovertime")
plt.close()



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
