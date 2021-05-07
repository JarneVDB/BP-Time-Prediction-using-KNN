from pm4py.objects.petri import semantics
from pm4py.objects.stochastic_petri import utils as stochastic_utils
from numpy.random import normal
from numpy.random import exponential
from numpy.random import uniform
from stochastic_maps import *
from side_functions import *
import random

# this algorithm is based on the paper of Rogge-Solti and Weske.

# further simulation of the petri net to the final marking given the initial marking,
# this method returns the duration of the simulation of this process until it is finished
def simulate_one_trace(net, im, fm, smap, distributiondic, time_enabled, time_passed=0):
    marking = im
    timelist = dict()
    timelist = initialize_net(marking, timelist, time_passed)
    while marking != fm:
        et = semantics.enabled_transitions(net, marking)
        trans = stochastic_utils.pick_transition(list(et), smap)
        maxi = 0
        for in_arc in trans.in_arcs:
            place_before = in_arc.source
            if len(place_before.in_arcs) == 0:
                timelist[place_before] = 0
            maxi = max(maxi, timelist[place_before])
        duration = float((transition_duration(distributiondic, trans, time_enabled)))
        for out_arc in trans.out_arcs:
            place_after = out_arc.target
            timelist[place_after] = int(duration + maxi)
        marking = semantics.execute(trans, net, marking)
    x = timelist.values()
    x = max(x)
    return x


# given a distribution and time passed, will truncate the density function if the function is enabled
# and will randomly draw a value given the probability density function
def transition_duration(distributiondic, trans, time_enabled):
    trunctime = 0
    if distributiondic[trans][0] == "IMMEDIATE":
        return 0
    elif distributiondic[trans][0] == "NORMAL":
        if trans.label in time_enabled.keys():
            trunctime = time_enabled[trans.label]
            return trunc_normal(trunctime, distributiondic[trans][1], distributiondic[trans][2])
        else:
            return normal(distributiondic[trans][1], distributiondic[trans][2])
    elif distributiondic[trans][0] == "EXPONENTIAL":
        duration = exponential(1 / (distributiondic[trans][1]))
        return duration
    elif distributiondic[trans][0] == "UNIFORM":
        if trans.label in time_enabled.keys():
            trunctime = time_enabled[trans.label]
        return uniform(trunctime, distributiondic[trans][2]) - trunctime
    else:
        print("wrong")


# loop that given a number of simulations will execute it multiple times to average away the variance
def simulate_multiple_iterations(net, im, fm, simulations, smap, time_enabled, time_passed=0):
    time = 0
    distributiondic = create_distributiondic(smap)
    for i in range(0, simulations):
        time = time + simulate_one_trace(net, im, fm, smap, distributiondic, time_enabled, time_passed)
    return time/simulations


# this method replays the petri net until the point of time passed and calls other functions
# that start calculating the prediction of the remaining time of the process instance
def predict_end_time(log, net, im, fm, traces, smap, trace, time_passed=0):
    if type(trace) is int:
        logtrace = list(log[trace])
    elif type(trace) is list:
        logtrace = trace
    else:
        logtrace = list(trace)
    time = logtrace[0]["time:timestamp"] + datetime.timedelta(0, time_passed)
    marking = im
    time_enabled = dict()
    time_enabled[logtrace[0]["concept:name"]] = logtrace[0]["time:timestamp"]
    done_transitions = list()
    done_transitions2 = list()
    for i in range(0, len(logtrace)):
        timestamp = (logtrace[i]["time:timestamp"])
        if time >= timestamp:
            trans = logtrace[i]["concept:name"]
            done_transitions.append(trans)
            done_transitions2.append(trans)

    for i in range(0, len(logtrace)):
        timestamp = (logtrace[i]["time:timestamp"])
        if time >= timestamp:
            trans = logtrace[i]["concept:name"]
            done_transitions2.remove(trans)
            l = 0
            start_marking = marking
            start_enabled = semantics.enabled_transitions(net, start_marking)
            for j in net.transitions:
                enabled = semantics.enabled_transitions(net, marking)
                enabled_str = str(list(enabled))
                if l != 1:
                    while trans not in enabled_str:
                        if "skip" not in enabled_str and "tau" not in enabled_str:
                            marking = start_marking
                            enabled = start_enabled
                        for m in random.sample(list(enabled), len(list(enabled))):
                            if m.label == None:
                                newmark = semantics.execute(m, net, marking)
                                if str(newmark) == "None":
                                    break
                                else:
                                    newenabled = semantics.enabled_transitions(net, newmark)
                                    removed = list(set(enabled) - set(newenabled))
                                    removed_label = list()

                                    if newmark == fm and len(done_transitions2) != 0:
                                        marking = start_marking
                                        enabled = start_enabled
                                        break

                                    for k in range(0, len(removed)):
                                        removed_label.append(removed[k].label)
                                    if not any(item in removed_label for item in done_transitions2):
                                            marking = newmark
                                            enabled_str = str(list(newenabled))
                                            enabled = newenabled
                l = 1
                if j.label == trans:
                    marking = semantics.execute(j, net, marking)
        for ok in semantics.enabled_transitions(net, marking):
            if ok.label not in time_enabled.keys():
                time_enabled[ok.label] = timestamp
    time_enabled2 = dict()
    for ok in semantics.enabled_transitions(net, marking):
        time_enabled2[ok.label] = (time - time_enabled[ok.label]).total_seconds()
    return simulate_multiple_iterations(net, marking, fm, traces, smap, time_enabled2, time_passed)

