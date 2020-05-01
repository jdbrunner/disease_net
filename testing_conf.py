import numpy as np
import json
import matplotlib.pyplot as plt
import covid_funs
import importlib
import subprocess
import sys
import shlex
import os


fldername = "periodic"
try:
    os.mkdir(fldername)
except:
    None


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


Bias = 1 #Bias = 1 means no bias. Higher puts bias towards testing symptomatic individuals.
capacities = [100,300,900,2700,8100]

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

timescale = 15 #1/gamma


init_inf = 0.01

s0 = (1-init_inf)
i0 = init_inf
r0 = 0

R0 = 2.2

end_time = 100

time = np.arange(0,end_time,0.01)
Symptomatic = 0.1*np.sin(time/5) + 0.2
Asymptomatic = np.zeros(len(time))
NonInfected = 1 - Symptomatic


dynamics = {"TimePoints":list(time), "Symptomatic":list(Symptomatic),"Asymptomatic":list(Asymptomatic),"NonInfected":list(NonInfected)}

#covid_funs.gen_dynamics(end_time,[s0,i0,r0],SIR_model,1,-1,[0,2],[R0/timescale,1/timescale])

covid_funs.gen_jsons(fldername,dynamics,Bias,capacities)

total_infected = np.array(dynamics['Symptomatic']) + np.array(dynamics['Asymptomatic'])

svfl = fldername+"/testresults.json"
dynamicsfl = fldername+"/dynamics.json"
biasfl = fldername+"/bias.json"
capfl = fldername+"/capacity.json"
num_trials = 100
falsePos = 0.1
falseNeg = 0.1
smth = 5
peak_tol = 3

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
so = run_command(full_command)

with open(svfl) as fl:
    results = json.load(fl)


pos_prop = {}
for ky in results["SimulatedData"]:
    smps = []
    times = []
    for sample in results["SimulatedData"][ky]:
        smps += [np.array(sample["DailyPositive"])/np.array(sample["DailyTotal"])]
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
    recall[ky] = sum([pk["Found"] for dyn in val for  pk in dyn["Recalls"]])/sum([len(v["Recalls"]) for v in val])

with open(fldername+"/recall.json","w") as outfile:
    json.dump(recall, outfile)

precision = {}
for ky,val in results["Performance"].items():
    precision[ky] = sum([pk["Real"] for dyn in val for  pk in dyn["Precisions"]])/sum([len(v["Precisions"]) for v in val])

with open(fldername+"/precision.json","w") as outfile:
    json.dump(precision, outfile)

mean_sq_error = {}
for ky,val in results["Performance"].items():
    mean_sq_error[ky] =[(sum([(pk["SqDist"])**2 for pk in dyn["Precisions"]])/len(dyn["Precisions"]))**(1/2) for dyn in val]

with open(fldername+"/mean_sq_error.json","w") as outfile:
    json.dump(mean_sq_error, outfile)
