from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm
from truncated_prediction_simulation import *
from pm4py.statistics.traces.log import case_statistics
from GDT_SPN_KNN import GDT_SPN_KNN
import math
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# this is the execution document; load in test and training log, decide on the parameters and this program
# will show you for each iteration interval the errors of the benchmarks and our prediction framework

test_log = xes_importer.apply()
training_log = xes_importer.apply()
number_of_iterations = 40
number_of_simulations = 500
number_of_neighbors = 100

for i in range(0, number_of_iterations):
    print("NEW INTERVAL:", i, "-----------------", "NEW INTERVAL:", i)
    remaining_test_log = partial_traces_by_interval(training_log, test_log, i, number_of_iterations)
    print("number of traces left in the testlog:", len(remaining_test_log))

    knnerror, knnabserror, knnrm, knn10error, knn10abserror, knn10sqerror, knn100error, knn100abserror, knn100sqerror = GDT_SPN_KNN(
        training_log, remaining_test_log, number_of_neighbors, number_of_simulations)

    all_case_durations = case_statistics.get_all_casedurations(training_log, parameters={
        case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
    avg = sum(all_case_durations)/(len(all_case_durations))
    avgerror = 0
    avgabserror = 0
    avgrm = 0
    for i in range(0, len(remaining_test_log)):
        trace = remaining_test_log[i]
        effective_duration = trace[0]["effective_duration"]
        avgerror = avgerror + (avg - effective_duration)
        avgabserror = avgabserror + abs(avg - effective_duration)
        avgrm = avgrm + ((avg - effective_duration) * (avg - effective_duration))

    roggesoltierror = 0
    roggesoltiabserror = 0
    roggesoltisq = 0
    net, im, fm = algorithm.apply(training_log)
    smap = create_smap(training_log, net, im, fm, "NORMAL")
    for i in range(0, len(remaining_test_log)):
        trace = remaining_test_log[i]
        effective_duration = trace[0]["effective_duration"]
        time_passed = trace[0]["time_passed"]
        pred = predict_end_time(training_log, net, im, fm, number_of_simulations, smap, trace, time_passed=time_passed)
        roggesoltierror = roggesoltierror + (pred - effective_duration)
        roggesoltiabserror = roggesoltiabserror + abs(pred - effective_duration)
        roggesoltisq = roggesoltisq + ((pred - effective_duration) * (pred - effective_duration))

    if len(remaining_test_log) != 0:
        print("AVERAGE ERROR")
        print(avgerror / len(remaining_test_log))
        print(avgabserror / len(remaining_test_log))
        me = avgrm/len(remaining_test_log)
        print(math.sqrt(me))

        print("ROGGE SOLTI ERROR")
        print(roggesoltierror / len(remaining_test_log))
        print(roggesoltiabserror / len(remaining_test_log))
        me = roggesoltisq/len(remaining_test_log)
        print(math.sqrt(me))

        print("KNN AVERAGE 10")
        print(knn10error)
        print(knn10abserror)
        print(knn10sqerror)

        print("KNN AVERAGE 100")
        print(knn100error)
        print(knn100abserror)
        print(knn100sqerror)

        print("KNN + ROGGE SOLTI ERROR")
        print(knnerror)
        print(knnabserror)
        print(knnrm)

    else:
        print("----TESTLOG IS EMPTY----")
