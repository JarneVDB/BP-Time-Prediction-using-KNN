import datetime
import scipy.stats as stats
from datetime import timedelta
from pm4py.statistics.traces.log import case_statistics
from pm4py.objects.log.log import Trace

def give_mean_duration(log):
    from pm4py.statistics.traces import log as duration
    dur = duration.case_statistics.get_all_casedurations(log)
    avgduration = sum(dur) / len(dur)

    return avgduration


def initialize_net(im, timelist, time=0):
    for i in im:
        timelist[i] = time

    return timelist


def give_effective_duration(log, i):
    list_of_trace = list(log[i])
    duration = (list_of_trace[len(list_of_trace)-1]["time:timestamp"] - list_of_trace[0]["time:timestamp"])
    duration = (datetime.timedelta.total_seconds(duration))

    return duration


def trunc_normal(trunctime, mu, stdv):
    a = (trunctime - mu) / stdv
    b = (100000000000000000 - mu) / stdv

    return stats.truncnorm.rvs(a, b, loc=mu, scale=stdv, size=1, random_state=None)-trunctime


# create partial traces given the current number of the interval
def partial_traces_by_interval(log, testlog, number_of_interval, intervals):
    all_case_durations = case_statistics.get_all_casedurations(log, parameters={
        case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
    avg = sum(all_case_durations) / (len(all_case_durations))
    time_passed = number_of_interval * ((2 * avg) / intervals)
    interval_list = list()
    for i in range(0, len(testlog)):
        temp_list = list()
        trace = testlog[i]
        list_trace = list(testlog[i])
        start_time = list_trace[0]["time:timestamp"]
        stop_time = start_time + timedelta(seconds=time_passed)
        effective_duration = give_effective_duration(testlog, i)
        if list_trace[len(trace)-1]["time:timestamp"] <= stop_time:
            continue
        for event in list_trace:
            if event["time:timestamp"] <= stop_time:
                event["time_passed"] = time_passed
                event["effective_duration"] = effective_duration
                temp_list.append(event)
        newtrace = Trace(temp_list)
        newtrace._set_attributes(trace._get_attributes())
        interval_list.append(newtrace)

    return interval_list
