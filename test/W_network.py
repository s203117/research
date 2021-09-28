import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
from numpy.core.fromnumeric import reshape

reservoir_node = 200
p = 0.05     #辺を生成する確率

# erdos_renyi グラフ作成
G = nx.erdos_renyi_graph(reservoir_node, p, directed = 1)

ary = nx.to_numpy_array(G)
normal_distribution = np.random.normal(0.0, 1.0, (reservoir_node, reservoir_node))
with open('W_network.csv','w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(normal_distribution*ary)

fig = plt.figure(figsize = (8, 8), dpi = 100)

#pos = nx.spring_layout(G)
pos = nx.random_layout(G)

nx.draw_networkx(G, pos)
plt.axis("off")
plt.show()
fig.savefig('W_network.pdf', bbox_inches='tight', pad_inches=0.05)
fig.savefig('W_network.png', bbox_inches='tight', pad_inches=0.05)

print(G.degree)
print(nx.degree_histogram(G))

fig = plt.figure(figsize = (8, 6), dpi = 100)
G_degree = G.degree()
degree_sequence = dict(G_degree).values()
bins = range(0,reservoir_node)
plt.hist(degree_sequence, bins = bins)

plt.xlim(xmin = 0)

plt.xlabel("degree",fontsize=20)
plt.ylabel("frequency",fontsize=20)
plt.tick_params(direction = "in", labelsize=17, pad = 5, length = 6)
#plt.show()

fig.savefig('W_network_hist.pdf', bbox_inches='tight', pad_inches=0.05)
fig.savefig('W_network_hist.png', bbox_inches='tight', pad_inches=0.05)

