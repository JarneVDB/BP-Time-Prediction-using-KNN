from sklearn import neighbors
from pm4py.objects.log.util import get_log_representation
import numpy as np
from stochastic_maps import create_smap
from pm4py.algo.discovery.inductive import algorithm
from truncated_prediction_simulation import predict_end_time
import math
from sklearn import preprocessing
from side_functions import *


# before creating the petri net, we do a feature construction so that we can build the petri net based
# on the specified number of nearest neighbors of the test trace
# we also calculate the RMSE and ME for both the benchmarks as our algorithm
# the number of nearest neighbors can be adjusted in this function
def GDT_SPN_KNN(log, testlog, number_neighbors, number_simulations):
    data, feature_names = get_log_representation.get_representation(log, str_ev_attr=["concept:name"],
                                                                    str_tr_attr=[], num_ev_attr=[], num_tr_attr=[])
    events = list()
    for i in feature_names:
        if "event:concept:name@" in i:
            x = i.lstrip("event:concept:name")
            y = x.lstrip("@")
            events.append(y)

    maxdic = [0] * len(events)
    all_values = list()
    all_prefixes = list()
    for j in range(0, len(log)):
        y = log[j]
        dic = dict()
        temp2 = list()
        for i in range(0, len(y)):
            dic[y[i]["concept:name"]] = datetime.timedelta.total_seconds(y[i]["time:timestamp"] - y[0]["time:timestamp"])
            temp2.append(y[i]["concept:name"])
        temp = list()
        n = 0
        for event in events:
            if event in dic.keys():
                temp.append(dic[event])
                temp2.append(event)
                maxdic[n] = max(maxdic[n], dic[event])
            else:
                temp.append(-1)
            n = n + 1
        all_values.append(temp)
        all_prefixes.append(temp2)
    all_values = np.array(all_values)
    all_test_values = list()
    all_test_prefixes = list()
    for j in range(0, len(testlog)):
        testtrace = testlog[j]
        dic = dict()
        temp2 = list()
        for i in range(0, len(testtrace)):
            dic[testtrace[i]["concept:name"]] = datetime.timedelta.total_seconds(testtrace[i]["time:timestamp"] - testtrace[0]["time:timestamp"])
            temp2.append(testtrace[i]["concept:name"])
        temp = list()
        for event in events:
            if event in dic.keys():
                temp.append(int(dic[event]))
            else:
                temp.append(-1)
        all_test_values.append(temp)
        all_test_prefixes.append(temp2)
    all_test_values = np.array(all_test_values)

    knnerror = 0
    knnrm = 0
    knnabserror = 0
    knn100error = 0
    knn10error = 0
    knnabs100error = 0
    knnabs10error = 0
    knn100sqerror = 0
    knn10sqerror = 0
    for i in range(0, len(all_test_values)):
        partial_trace = testlog[i]
        new_all_test_values = list()
        new_all_values = list()
        partial_trace_values = all_test_values[i]
        count = 0
        for j in all_values:
            test_temp = list()
            all_temp = list()
            wrong_list = list()
            wrong = 0
            for k in range(0, len(j)):
                if k < len(all_test_prefixes[i]) and k < len(all_prefixes[count]):
                    if all_test_prefixes[i][k] != all_prefixes[count][k]:
                        wrong = 1
                if partial_trace_values[k] > 0 and j[k] <= 0:
                    test_temp.append(partial_trace_values[k])
                    all_temp.append((maxdic[k]) * -1)
                    wrong_list.append((maxdic[k]) * -1)
                elif partial_trace_values[k] != -1:
                    test_temp.append(partial_trace_values[k])
                    all_temp.append(j[k])
                    wrong_list.append((maxdic[k]) * -1)

            if wrong == 1:
                all_temp = wrong_list
            new_all_values.append(all_temp)
            count = count + 1
        new_all_test_values.append(test_temp)
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = min_max_scaler.fit_transform(new_all_values)
        X_test_minmax = min_max_scaler.transform(new_all_test_values)
        new_all_test_values = X_test_minmax
        new_all_values = X_train_minmax
        knn = neighbors.NearestNeighbors(n_neighbors=number_neighbors, algorithm="auto")
        baselineknn = neighbors.NearestNeighbors(n_neighbors=10, algorithm="auto")
        knn.fit(new_all_values)
        baselineknn.fit(new_all_values)
        if len(new_all_test_values) != 1:
            distances, indices = knn.kneighbors([new_all_test_values])
            distances10, indices10 = baselineknn.kneighbors([new_all_test_values])
        else:
            distances, indices = knn.kneighbors(new_all_test_values)
            distances10, indices10 = baselineknn.kneighbors(new_all_test_values)
        baselineneighbor_list = list()
        neighbor_list = list()
        total_duration100 = 0
        for l in indices[0]:
            neighbor_list.append(log[l])
            total_duration100 = total_duration100 + give_effective_duration(log, l)
        avg_duration100 = total_duration100 / len(neighbor_list)
        total_duration10 = 0
        for l in indices10[0]:
            baselineneighbor_list.append(log[l])
            total_duration10 = total_duration10 + give_effective_duration(log, l)
        avg_duration10 = total_duration10/len(baselineneighbor_list)
        net, im, fm = algorithm.apply(neighbor_list)
        smap = create_smap(neighbor_list, net, im, fm,"NORMAL")
        time_passed = partial_trace[0]["time_passed"]
        x = predict_end_time(neighbor_list, net, im, fm, number_simulations, smap, partial_trace, time_passed=time_passed)
        effective_duration = partial_trace[0]["effective_duration"]

        knnerror = knnerror + (x - effective_duration)
        knnabserror = knnabserror + abs(x - effective_duration)
        knnrm = knnrm + ((x - effective_duration) * (x - effective_duration))
        knnabs10error = knnabs10error + abs(avg_duration10 - effective_duration)
        knnabs100error = knnabs100error + abs(avg_duration100 - effective_duration)
        knn10error = knn10error + (avg_duration10 - effective_duration)
        knn10sqerror = knn10sqerror + ((avg_duration10 - effective_duration)*(avg_duration10 - effective_duration))
        knn100error = knn100error + (avg_duration100 - effective_duration)
        knn100sqerror = knn100sqerror + ((avg_duration100 - effective_duration) * (avg_duration100 - effective_duration))

    if len(testlog) == 0:
        knnerror, knnabserror, knnrm, knn10error, knn10abserror, knn10sqerror, knn100error, knn300abserror, knn100sqerror = 0, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        knnerror = (knnerror / len(testlog))
        me = knnrm / len(testlog)
        knnrm = (math.sqrt(me))
        knn10error = (knn10error / len(testlog))
        me = knn10sqerror / len(testlog)
        knn10sqerror = (math.sqrt(me))
        knn100error = (knn100error / len(testlog))
        me = knn100sqerror / len(testlog)
        knn100sqerror = (math.sqrt(me))
        knnabserror = (knnabserror / len(testlog))
        knnabs10error = (knnabs10error / len(testlog))
        knnabs100error = (knnabs100error / len(testlog))

    return knnerror, knnabserror, knnrm, knn10error, knnabs10error, knn10sqerror, knn100error, knnabs100error, knn100sqerror
