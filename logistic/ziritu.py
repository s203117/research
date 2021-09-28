# -----------------------------------------------------------------------------
# ノード結合重み固定(not random)
# 入力重み可変(±winの2値ランダム)
# online学習(RLS)

# 出力重みの学習とシステム同定は1step予測
# その後再帰予測でRCを自立的に動かす(RCの出力を次のRCの入力に)
# RCの予測値が固定点に十分近づいたとき，リザバーのヤコビ行列と制御ゲインを求める
# 制御開始

# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from numpy.core.fromnumeric import reshape

alpha = 0.8
reservoir_node = 200
input_node = 100
win = 0.5
w = 0.5
wout = 0.5
T = 100000                   # 学習時間
predict_T = 300               # 
step = 1                      #step数
result = np.array([])     #予測結果
original_signal = np.array([])         #教師信号

# ランダムグラフから読み込み
W = np.loadtxt('W_network.csv', delimiter=',', dtype='float')
spectral_radius = max(abs(np.linalg.eigvals(W)))
W = W / spectral_radius * 0.9999
spectral_radius = max(abs(np.linalg.eigvals(W)))
with open('W_network.csv','w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(W)
#print(W)
print('スペクトル半径', spectral_radius)

# Win = ±winの2値ランダム
Win = (np.random.randint(0, 2, (reservoir_node,1)) * 2 - 1) * win
Win[:reservoir_node-input_node] = 0.0
#print("Win",Win)

reservoir = np.zeros((reservoir_node,1))
old_reservoir = reservoir
Wout = np.ones((1,reservoir_node)) * wout
y = 0.0

#output feedback
wfb = 0.0
Wfb = np.zeros((reservoir_node,1))
Wfb[0][0] = 1
Wfb = Wfb * wfb

#Wout更新用変数
delta = 0.00001
mu = 0.95
e = 0.0
P_1 = np.identity(reservoir_node) / delta
P_2 = np.identity(reservoir_node) / delta

# ロジスティック写像
logistic_a = 3.9
x = 0.1
old_x = x

for a in range (100):
    x = logistic_a * old_x * (1 - old_x)
    old_x = x

#出力重み学習
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
    P_1 = (P_1 - (P_1 @ reservoir @ reservoir.T @ P_1)/(mu + reservoir.T @ P_1 @ reservoir)) / mu
    L = P_1 @ reservoir / (mu + reservoir.T @ P_1 @ reservoir)
    Wout = Wout + L.T * e

    old_reservoir = reservoir
    mu = (1-0.01)*mu + 0.01
    
    


# 固定点の検出
fixed_x1 = 0.744
"""
for a in range (0, predict_T+1):
    # x1の予測
    u = y
    reservoir = (1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))
    y = Wout @ reservoir
    
    old_reservoir = reservoir
    print(y)
    result = np.append(result, y)
    x = logistic_a * old_x * (1 - old_x)
    old_x = x
    original_signal = np.append(original_signal, x)
t = np.linspace(0, predict_T, predict_T+1)
"""

t = np.array([])
time = 0
standard_value = 0.001      #固定点とRC出力の差の閾値
old_y = 0.0
while standard_value < abs(u - y):
    # x1の予測
    u = y
    reservoir = (1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))
    y = Wout @ reservoir
    
    old_reservoir = reservoir
    print(y)
    result = np.append(result, y)
    x = logistic_a * old_x * (1 - old_x)
    old_x = x
    original_signal = np.append(original_signal, x)
    t = np.append(t, time)
    time = time + 1

    if  (y<0.0) or (y>1.0):
        print("予測値が0~1の範囲外です")
        break

##### plot #####
fig = plt.figure(figsize = (12, 5), dpi = 100)
ax1 = fig.add_subplot(1,1,1)
ax1.plot(t, original_signal, lw = 2, zorder=1, c = 'orange', label="")
ax1.plot(t, result, lw = 2, zorder=2, c = 'blue', label="")

ax1.set_xlabel('t', fontsize=20)
ax1.set_ylabel('x', fontsize=20, labelpad=-2)

if time >= 100 :
    ax1.set_xlim(time-100,time-1)
else :
    ax1.set_xlim(0)
#ax1.set_xlim(0, predict_T)
#ax1.set_ylim(0,1)

#ax.legend(["理論値","RC出力値"], prop={"family":"MS Gothic", "size":17}, loc = "lower center", frameon = False, ncol = 2)
#ax1.legend(prop={"size":17}, loc = "upper center", frameon = False, ncol = 2)
ax1.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)
plt.show()

fig.savefig('ziritu.pdf', bbox_inches='tight', pad_inches=0.05)
fig.savefig('ziritu.png', bbox_inches='tight', pad_inches=0.05)