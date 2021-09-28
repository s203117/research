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
predict_T = 10               # 固定点の制御時間
step = 1                      #step数
result = np.array([])     #予測結果
original_signal = np.array([])         #教師信号

# W_network 読み込み
W = np.loadtxt('W_network.csv', delimiter=',', dtype='float')
spectral_radius = max(abs(np.linalg.eigvals(W)))
W = W / spectral_radius * 0.9999
spectral_radius = max(abs(np.linalg.eigvals(W)))
print('スペクトル半径', spectral_radius)

# Win 読み込み
Win = np.loadtxt('W_in.csv', dtype='float')
Win = np.reshape(Win, (reservoir_node, 1))

# W 読み込み
reservoir = np.loadtxt('reservoir.csv', delimiter=',', dtype='float')
reservoir = np.reshape(Win, (reservoir_node, 1))
old_reservoir = reservoir

# Wout 読み込み
Wout = np.loadtxt('W_out.csv', delimiter=',', dtype='float')

y = 0.0

# 固定点の検出
standard_value = 0.001      #固定点とRC出力の差の閾値
fixed_x1 = 0.744
while standard_value < abs(fixed_x1 - y):
    # x1の予測
    u = y
    reservoir = (1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))
    y = Wout @ reservoir
    
    if  (y<0.0) or (y>1.0):
        print("予測値が0~1の範囲外です")
        break
    
    old_reservoir = reservoir
    print(y)
    result = np.append(result, y)

u = y
reservoir = (1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))
y = Wout @ reservoir
old_reservoir = reservoir
print(y)
result = np.append(result, y)

##### plot #####
fig = plt.figure(figsize = (6, 5), dpi = 100)
ax1 = fig.add_subplot(1,1,1)
#ax1.scatter(aa_t, original_signal, s = 1, zorder=1, c = 'green', label="")
#ax1.plot(result[:-1], result[1:], lw = 2, zorder=2, c = 'red', label="")
ax1.scatter(result[:-1], result[1:], s = 20, zorder=2, c = 'red', label="")
ax1.scatter(fixed_x1, fixed_x1, s = 30, zorder=3, c = 'yellow', label="")

ax1.set_xlabel('x_t', fontsize=20)
ax1.set_ylabel('x_t+1', fontsize=20, labelpad=-2)

#ax1.set_xlim(T-100,T)
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.0, 1.0)

#ax.legend(["理論値","RC出力値"], prop={"family":"MS Gothic", "size":17}, loc = "lower center", frameon = False, ncol = 2)
#ax1.legend(prop={"size":17}, loc = "upper center", frameon = False, ncol = 2)
ax1.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)
plt.show()

fig.savefig('fixpoint_search.pdf', bbox_inches='tight', pad_inches=0.05)
fig.savefig('fixpoint_search.png', bbox_inches='tight', pad_inches=0.05)