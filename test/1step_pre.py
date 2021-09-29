# -----------------------------------------------------------------------------
# ノード結合重み固定(not random)
# 入力重み可変(±winの2値ランダム)
# online学習(RLS)

# 出力重みの学習 & reservoir 発展
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from numpy.core.fromnumeric import reshape

alpha = 0.8
reservoir_node = 200
input_node = 100
wout = 0.5
T = 100000                   # 学習時間

# ランダムグラフから読み込み
W = np.loadtxt('W.csv', delimiter=',', dtype='float')
spectral_radius = max(abs(np.linalg.eigvals(W)))
print('スペクトル半径', spectral_radius)

# Win 読み込み
Win = np.loadtxt('W_in.csv', dtype='float')
Win = np.reshape(Win, (reservoir_node, 1))

reservoir = np.zeros((reservoir_node,1))
old_reservoir = reservoir
Wout = np.ones((1,reservoir_node)) * wout
y = 0.0

#Wout更新用変数
delta = 0.00001
mu = 0.95
e = 0.0
P = np.identity(reservoir_node) / delta

# ロジスティック写像
logistic_a = 3.9
x = 0.1
old_x = x

aa = 0.0
aa_t = np.array([])
original_signal = np.array([])         #教師信号
while aa <= 1.0 :
    x = logistic_a * aa * (1 - aa)
    aa_t = np.append(aa_t, aa)
    aa = aa + 0.001
    original_signal = np.append(original_signal, x)

with open('original_signal.csv','w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(original_signal)

for a in range (100):
    x = logistic_a * old_x * (1 - old_x)
    old_x = x

#出力重み学習
x_plot = np.array([])
RC_plot = np.array([])
t = np.array([])
for a in range(0,T+1):
    # x1の予測
    u = x
    reservoir = (1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))
    y = Wout @ reservoir
    
    #Wout更新
    x = logistic_a * old_x * (1 - old_x)
    old_x = x

    target = x
    e = target - y
    P = (P - (P @ reservoir @ reservoir.T @ P)/(mu + reservoir.T @ P @ reservoir)) / mu
    L = P @ reservoir / (mu + reservoir.T @ P @ reservoir)
    Wout = Wout + L.T * e

    old_reservoir = reservoir
    mu = (1-0.01)*mu + 0.01

    x_plot = np.append(x_plot, x)
    RC_plot = np.append(RC_plot, y)
    t = np.append(t, a)

y = Wout @ reservoir
print(y)

with open('W_out.csv','w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(Wout)

with open('reservoir.csv','w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(reservoir)

fig = plt.figure(figsize = (12, 5), dpi = 100)
ax1 = fig.add_subplot(1,1,1)
ax1.plot(t, x_plot, lw = 2, zorder=1, c = 'orange', label="")
ax1.plot(t, RC_plot, lw = 2, zorder=2, c = 'blue', label="")

ax1.set_xlabel('t', fontsize=20)
ax1.set_ylabel('x', fontsize=20, labelpad=-2)
ax1.set_xlim(0, T)

ax1.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)

fig.savefig('1step_pre.pdf', bbox_inches='tight', pad_inches=0.05)
fig.savefig('1step_pre.png', bbox_inches='tight', pad_inches=0.05)