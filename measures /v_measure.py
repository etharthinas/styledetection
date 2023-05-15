import numpy as np
import pandas as pd

def marginal_entropy(table, classes, values):
    grouped = table[[classes, values]].groupby(classes).count()
    count = sum(grouped[values])
    entropy_sum = 0
    for i in grouped[values]:
        entropy_sum -= i/count * np.log2(i/count)
    return entropy_sumdef marginal_entropy(table, classes, values):
    grouped = table[[classes, values]].groupby(classes).count()
    count = sum(grouped[values])
    entropy_sum = 0
    for i in grouped[values]:
        entropy_sum -= i/count * np.log2(i/count)
    return entropy_sum

def intercluster_entropy(table, cluster_id):
    col = table[cluster_id].dropna()
    count = sum(col)
    entropy_sum = 0
    for i in col:
        entropy_sum -= i/count * np.log2(i/count)
    return entropy_sum * len(col)def intercluster_entropy(table, cluster_id):
    col = table[cluster_id].dropna()
    count = sum(col)
    entropy_sum = 0
    for i in col:
        entropy_sum -= i/count * np.log2(i/count)
    return entropy_sum * len(col)

def total_entropy(dataframe, classes, clusters, values):
    pivot_table = dataframe[[classes, clusters, values]].pivot_table(index = classes, columns = clusters, values = values, aggfunc = "count")
    length = len(dataframe)
    entropy_sum = 0
    for cluster_id in pivot_table.columns:
        entropy_sum += intercluster_entropy(pivot_table, cluster_id)
    return entropy_sum / lengthdef total_entropy(dataframe, classes, clusters, values):
    pivot_table = dataframe[[classes, clusters, values]].pivot_table(index = classes, columns = clusters, values = values, aggfunc = "count")
    length = len(dataframe)
    entropy_sum = 0
    for cluster_id in pivot_table.columns:
        entropy_sum += intercluster_entropy(pivot_table, cluster_id)
    return entropy_sum / length

def v_measure(dataframe, classes, clusters, values, beta=1):
    h = 1 - total_entropy(dataframe, classes, clusters, values) / marginal_entropy(dataframe, classes, values)
    c = 1 - total_entropy(dataframe, clusters, classes, values) / marginal_entropy(dataframe, clusters, values)
    return (1 + beta) * h * c / (beta * h + c)

