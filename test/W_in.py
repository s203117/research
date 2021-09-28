import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
from numpy.core.fromnumeric import reshape

reservoir_node = 200
input_node = 100
win = 0.5

# Win = ±winの2値ランダム
Win = (np.random.randint(0, 2, (reservoir_node,1)) * 2 - 1) * win
Win[:reservoir_node-input_node] = 0.0

with open('W_in.csv','w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(Win)