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
Bias = sys.argv[2]# #Bias = 1 means no bias. Higher puts bias towards testing symptomatic individuals.

fldername = sys.argv[3]#"biasSIR"
try:
    os.mkdir(fldername)
except:
    None





capacities = [100,300,900,2700,8100]



symptProp = 1

five_day_conf = {}
five_day_slope_error ={}

slopes = np.arange(-0.1,0.1,0.0025)

for slp in slopes:

    end_time = 5

    time = np.arange(0,end_time,0.01)

    if slp >0:
        total_infected = slp*time
    else:
        total_infected = slp*time-slp*5

    Symptomatic = symptProp*total_infected
    Asymptomatic = (1-symptProp)*total_infected
    NonInfected = 1 - total_infected


    dynamics = {"TimePoints":list(time), "Symptomatic":list(Symptomatic),"Asymptomatic":list(Asymptomatic),"NonInfected":list(NonInfected)}


    covid_funs.gen_jsons(fldername,dynamics,Bias,capacities)

    svfl = fldername+"/testresults.json"
    dynamicsfl = fldername+"/dynamics.json"
    biasfl = fldername+"/bias.json"
    capfl = fldername+"/capacity.json"
    falsePos = 0.1
    falseNeg = 0.1

    base_command = "./disease_confidence"
    opts = ["-Dynamics="+dynamicsfl]
    opts +=["-TestingBias="+biasfl]
    opts +=["-TestingCapacities="+capfl]
    opts +=["-Trials="+str(num_trials)]
    opts +=["-SaveFile="+svfl]
    opts +=["-FalsePositive="+str(falsePos)]
    opts +=["-FalseNegative="+str(falseNeg)]
    opts +=["-ComputePeaks=false"]


    full_command = base_command + " " + " ".join(opts)
    so = covid_funs.run_command(full_command)

    with open(svfl) as fl:
        results = json.load(fl)




    five_day_conf_lam = {}
    five_day_slope_error_lam = {}
    for ky in results:
        tot = 0
        tot2 = 0
        for sample in results[ky]:
            smps = np.array(sample["DailyPositive"])/np.maximum(1,np.array(sample["DailyTotal"]))
            times = np.array(sample["DayTimes"])
            got_trend,sqer = covid_funs.trendConfidenceInd(smps,times,total_infected,dynamics["TimePoints"],5)
            tot += got_trend
            tot2 += sqer
        five_day_conf_lam[ky] = tot/len(results[ky])
        five_day_slope_error_lam[ky] = tot2/len(results[ky])

    five_day_conf[slp] = five_day_conf_lam
    five_day_slope_error[slp] = five_day_slope_error_lam

with open(fldername+"/five_day_conf.json","w") as outfile:
    json.dump(five_day_conf, outfile)



conf_vs_slope = {}
var_vs_slope = {}
for lam in capacities:
    conf_vs_slope[str(lam)] = [[],[]]
    var_vs_slope[str(lam)] = [[],[]]


for ky in five_day_conf.keys():
    for ky2 in five_day_conf[ky].keys():
        conf_vs_slope[ky2][0] += [five_day_conf[ky][ky2]]
        conf_vs_slope[ky2][1] += [ky]

for ky in five_day_slope_error.keys():
    for ky2 in five_day_slope_error[ky].keys():
        var_vs_slope[ky2][0] += [five_day_slope_error[ky][ky2]]
        var_vs_slope[ky2][1] += [ky]

fig,ax = plt.subplots(2,1,figsize = (10,10))
for ky in conf_vs_slope.keys():
    ax[0].plot(conf_vs_slope[ky][1],conf_vs_slope[ky][0], label = "Samples/Day:" + str(ky))
    ax[1].plot(var_vs_slope[ky][1],var_vs_slope[ky][0], label = "Samples/Day:" + str(ky))

ax[0].set_xlabel("Slope of dynamics")
ax[0].set_ylabel("Five-day confidence")
ax[0].legend()
ax[1].set_xlabel("Slope of dynamics")
ax[1].set_ylabel("Mean-Square Error of Estimated Slope")
ax[1].legend()
fig.savefig(fldername+"/vsSlope")
plt.close()
