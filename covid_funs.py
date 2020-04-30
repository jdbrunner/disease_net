import numpy as np
from scipy.integrate import cumtrapz
from numpy.random import exponential as exp
from numpy.random import rand
from scipy.ndimage import gaussian_filter1d as smooth



import matplotlib.pyplot as plt

def linear_interp(t,time_array,value_array):
    time_where= np.argwhere(time_array > t)
    if len(time_where):
        time_coord = time_where[0][0] -1
    else:
        time_coord = len(time_array) - 2
    tstep = time_array[time_coord+1]-time_array[time_coord]
    return value_array[time_coord] + ((value_array[time_coord+1] - value_array[time_coord])/tstep)*(t-time_array[time_coord])

def sim_sampling_deterministic(disease_dynamics, time_array, sampling_capacity, sampling_bias, infected_coord = 1, symptomatic_coord = 1):

    '''

    disease_dynamics should be given as array with columns representing compartments and rows representing time points. Give
    column number of infected people (infected_coord, default 1) and symptomatic people (symptomatic_coord, default 1).

    sampling_capacity and sampling_bias may be given as functions of (time,disease_dynamics)

    '''

    disease_dynamics = np.array(disease_dynamics)

    if not np.isscalar(infected_coord):
        infected = np.sum(disease_dynamics[:,infected_coord],axis = 1)
    else:
        infected = disease_dynamics[:,infected_coord]

    total_cases = cumtrapz(infected,time_array,initial = infected[0])

    if not np.isscalar(symptomatic_coord):
        infected_symptoms = np.sum(disease_dynamics[:,symptomatic_coord],axis = 1)
    else:
        infected_symptoms = disease_dynamics[:,symptomatic_coord]

    non_infected = np.sum(disease_dynamics,axis = 1) - infected
    infected_non_symptom = infected - infected_symptoms


    if callable(sampling_bias):
        biases = np.array([sampling_bias(t,disease_dynamics) for t in time_array])
    elif np.isscalar(sampling_bias):
        biases = np.array([sampling_bias for t in time_array])
    else:
        biases = np.array(sampling_bias)

    if callable(sampling_capacity):
        test_rates = np.array([sampling_capacity(t,disease_dynamics) for t in time_array])
    elif np.isscalar(sampling_capacity):
        test_rates = np.array([sampling_capacity for t in time_array])
    else:
        test_rates = np.array(sampling_capacity)

    total_tests = cumtrapz(test_rates,time_array,initial = 0)

    total_tests2 = total_tests.copy()
    total_tests2[0] = 1

    #biased sampling
    positive_rate = test_rates*(biases*infected_symptoms + infected_non_symptom)/(biases*infected_symptoms + infected_non_symptom + non_infected)
    #biased sampling (integrated)
    total_positve = cumtrapz(positive_rate,time_array,initial = 0)
    positive_proportion = total_positve/total_tests2
    positive_prop_rate = positive_rate/test_rates

    samp_dict = {"TotalTests":total_tests,"PositiveRate":positive_rate,"TotalPositive":total_positve,"TotalPositiveProportion":positive_proportion,"PositiveProportionRate":positive_prop_rate}

    return samp_dict



#Mixed deterministic disease with stochastic sampling.# Assumes bias is only in a2!
def sim_sampling_stochastic(disease_dynamics, time_array, sampling_capacity, sampling_bias, infected_coord = 1, symptomatic_coord = 1):


    '''disease_dynamics should be given as array with columns representing compartments and rows representing time points. Give
    column number of infected people (infected_coord, default 1) and symptomatic people (symptomatic_coord, default 1).'''

    disease_dynamics = np.array(disease_dynamics)

    if not np.isscalar(infected_coord):
        infected = np.sum(disease_dynamics[:,infected_coord],axis = 1)
    else:
        infected = disease_dynamics[:,infected_coord]

    total_cases = cumtrapz(infected,time_array,initial = infected[0])

    if not np.isscalar(symptomatic_coord):
        infected_symptoms = np.sum(disease_dynamics[:,symptomatic_coord],axis = 1)
    else:
        infected_symptoms = disease_dynamics[:,symptomatic_coord]

    non_infected = np.sum(disease_dynamics,axis = 1) - infected
    infected_non_symptom = infected - infected_symptoms

    pos = 0
    neg = 0

    tests = [[pos,neg]]

    t = 0
    test_times = np.array([t])



    if callable(sampling_capacity):
        Lmax = max([sampling_capacity(t) for t in time_array])
    elif np.isscalar(sampling_capacity):
        Lmax = sampling_capacity
    else:
        Lmax = max(sampling_capacity)

    while t < time_array[-1]:

        dt = exp(1/Lmax)
        t += dt



        if callable(sampling_capacity):
            lamb0 = sampling_capacity(t)
        elif np.isscalar(sampling_capacity):
            lamb0 = sampling_capacity
        else:
            lamb0 = linear_interp(t,time_array,sampling_capacity)

        sympt = linear_interp(t,time_array,infected_symptoms)
        infected_non_sympt = linear_interp(t,time_array,infected_non_symptom)
        non_infect = linear_interp(t,time_array,non_infected)





        if callable(sampling_bias):
            bias = sampling_bias(t)
        elif np.isscalar(sampling_bias):
            bias = sampling_bias
        else:
            bias = linear_interp(t,time_array,sampling_bias)

        N = bias*sympt + infected_non_sympt + non_infect
        lam1 = lamb0*(bias*sympt + infected_non_sympt)/N
        lam2 = lamb0*non_infect/N

        u = Lmax*rand()

        if u < lam1: #tested positive
            pos += 1
            tests += [[pos,neg]]
            test_times = np.append(test_times,t)
        elif u < (lam1 + lam2): #tested negative
            neg += 1
            tests += [[pos,neg]]
            test_times = np.append(test_times,t)


    return test_times,np.array(tests)


def count_perday(time,count, interval = 1.0):
    ''' Given array of times & counts, divide into intervals of given length with Delta Count in each interval. '''

    num_intervals =int(np.ceil(time[-1]/interval))
    perday = []
    interval_midpoints = []
    for i in range(num_intervals):
        interval_start = i*interval
        interval_end = (i+1)*interval
        interval_midpoints += [interval_start + 0.5*interval]
        interval_start_index = np.argwhere(time>interval_start)[0,0]
        interval_end_index_0 = np.argwhere(time>=interval_end)
        if len(interval_end_index_0):
            interval_end_index = interval_end_index_0[0,0]
        else:
            interval_end_index = len(time) - 1
        delta_count = count[interval_end_index] - count[interval_start_index]
        perday += [delta_count]


    return np.array(perday),np.array(interval_midpoints)


def peak_quadfit(samples,sampletimes,ws):
    window_size = ws[0]
    peak_detected = []
    peak_times = []
    for i in range(window_size,len(samples)):
        x = sampletimes[:i]
        y = samples[:i]
        p = np.polyfit(x,y,2)
        if p[0]:
            pk = -p[1]/(2*p[0])
        else:
            pk = -1
        peak_detected += [(pk >= x[0] and pk <= x[-1] and p[0] < 0)]
        peak_times += [x[-1]]
    peak_detected = np.array(peak_detected)
    peak_times = np.array(peak_times)[peak_detected]

    return peak_times

def peak_quadfit2(samples,sampletimes,ws):
    window_size = ws[0]
    peak_detected = []
    peak_times = []
    for i in range(len(samples) - window_size):
        x = sampletimes[i:i+window_size]
        y = samples[i:i+window_size]
        p = np.polyfit(x,y,2)
        if p[0]:
            pk = -p[1]/(2*p[0])
        else:
            pk = -1
        peak_detected += [(pk >= x[0] and pk <= x[-1] and p[0] < 0)]
        peak_times += [x[-1]]
    peak_detected = np.array(peak_detected)
    peak_times = np.array(peak_times)[peak_detected]

    return peak_times

def peak_deriv(samples,sampletimes,smth):

    smth = smth[0]
    smooth_deriv = smooth(np.diff(samples).astype(float),smth)/np.diff(sampletimes)
    declining = np.argwhere(smooth_deriv < 0)[0]

    return declining

def assess_peakfind(peak_fun,sampling_capacity,dynamics,pk_fun_params,sampling_bias = 1,interval_size = 1,infected_coord = 1, symptomatic_coord = 1,tol =2):

    time_array,sol_array = dynamics

    if not np.isscalar(infected_coord):
        infected = np.sum(sol_array[:,infected_coord],axis = 1)
    else:
        infected = sol_array[:,infected_coord]

    test_times, test_results = sim_sampling_stochastic(sol_array, time_array, sampling_capacity, sampling_bias)


    total_tests = np.sum(test_results,axis = 1)


    perday,daytimes = count_perday(test_times,test_results[:,0], interval = interval_size)
    perdaytotal,daytimestotal = count_perday(test_times,total_tests, interval = interval_size)

    propotion_per_day = perday/np.maximum(1,perdaytotal)

    peak_times_found = peak_fun(perday,daytimes,pk_fun_params)

    real_peak = time_array[np.argmax(infected)]


    # fig,ax = plt.subplots(figsize = (10,5))
    # ax.plot(time_array,sampling_capacity*sol_array[:,1], label = "Real Cases Per Test", color = "r")
    # ax.bar(daytimes,perday,label = "Positive Propotion Per Day") #Can estimate
    # ax.plot(peak_times_found,max(perday)*np.ones(len(peak_times_found)),"x", color = "g",label ="Detected Peak")
    #
    # ax.legend()
    # ax.set_title("Peak Detection, " + str(sampling_capacity) + " tests per day")
    #
    # plt.show()

    return peak_times_found[0] - real_peak #(peak_times_found[0] >= real_peak-tol) and (peak_times_found[0] <= real_peak+tol)
