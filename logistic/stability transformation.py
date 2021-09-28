#####################################
# Henon写像の予測
# ノード結合重み固定(not random)
# 入力重み可変(±winの2値ランダム)
# ノード数可変
# online学習(RLS)
# output feedback項追加

# 再帰予測(RCの出力を次のRCの入力に)

# 1つ前の出力値と今の出力値が十分に近いとき→1つ前の値を固定点にする

#####################################
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
T = 100000          #学習時間
predict_T = 50            # 固定点の制御時間
step = 1           #step数
x1_result = np.array([])    #予測結果
x1_plot = np.array([])  #教師信号

# ランダムグラフから読み込み
W = np.loadtxt('W_network.csv', delimiter=',', dtype='float')
#W = np.loadtxt('test.csv', delimiter=',', dtype='float')
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
x_step = 0.0
old_x_step = 0.0

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
    old_x_step = x
    for b in range(step):
        x_step = logistic_a * old_x_step * (1 - old_x_step)
        old_x_step = x_step
    # x1の更新
    target = x_step
    e = target - y
    P_1 = (P_1 - (P_1 @ reservoir @ reservoir.T @ P_1)/(mu + reservoir.T @ P_1 @ reservoir)) / mu
    L = P_1 @ reservoir / (mu + reservoir.T @ P_1 @ reservoir)
    Wout = Wout + L.T * e

    old_reservoir = reservoir
    mu = (1-0.01)*mu + 0.01
    x = logistic_a * old_x * (1 - old_x)
    old_x = x
    x1_plot = np.append(x1_plot, x)


# 固定点の検出
standard_value = 0.001      #固定点とRC出力の差の閾値
fixed_x1 = 0.744
ziritu_x = np.array([])
while standard_value < abs(fixed_x1 - y):
    # x1の予測
    u = y
    reservoir = (1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))
    y = Wout @ reservoir
    
    if (y<0.0) or (y>1.0):
        print(y)
        sys.exit("予測値が0~1の範囲外です")
    
    old_reservoir = reservoir
    print(y)
    

ziritu_x = np.append(ziritu_x, y)
print("固定点周りのy",y)

#ヤコビ行列 A = ∂x_j / ∂x_k と制御ゲインKの算出 ---------------------------------------------------------
I = np.identity(reservoir_node)
A = np.identity(reservoir_node) * (1-alpha) # ヤコビ行列
K = np.zeros((reservoir_node,reservoir_node)) # 制御ゲイン

j = 0  
#print("old_reserver",old_reservoir)
#print("u",u)
while j < reservoir_node:
    k = 0
    while k < reservoir_node:
        if j != k:
            if W[j,k] != 0:
                l = 0
                aaa = 0.0
                while l < reservoir_node:
                    if l != k:
                        aaa = aaa + W[j,l]*old_reservoir[l]
                    if l == k:
                        aaa = aaa + W[j,l]*old_reservoir[l]
                    l = l+1
                #print("aaa",aaa)
                #print("Win[j]",Win[j])
                #print("W[j,k]",W[j,k])
                A[j,k] = alpha * W[j,k] / pow(np.cosh(Win[j]*y + aaa), 2)
        k = k+1
    j = j+1
with open('logi_A.csv','w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(A)
K = -(np.linalg.inv(I-A)) @ A
with open('logi_K.csv','w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(K)
print(K)

# 安定化
for a in range (0,predict_T):
    # x1の予測
    #u = x
    u = y
    reservoir = (I-K)@((1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))) + K@old_reservoir
    #x_1 = (1-alpha)*old_x_1 + alpha*(np.tanh(Win*u + W@old_x_1)) + K@((1-alpha)*old_x_1 + alpha*(np.tanh(Win*u + W@old_x_1)) - old_x_1)
    y = Wout @ reservoir

    old_reservoir = reservoir

    # 予測結果plot
    x1_result = np.append(x1_result, y)
print("制御後",x1_result)

##### plot #####
#t = np.linspace(0, T, T+1)
t = np.linspace(0, predict_T, predict_T+1)
fig = plt.figure(figsize = (6, 5), dpi = 100)
ax1 = fig.add_subplot(1,1,1)
result = x
ax1.scatter(x1_plot[:-1], x1_plot[1:], s = 1, zorder=1, c = 'green', label="")
#ax1.scatter(ziritu_x[:-1], ziritu_x[1:], lw = 1, zorder=2, c = 'blue', label="")
ax1.plot(x1_result[:-1], x1_result[1:], lw = 2, zorder=2, c = 'red', label="")
ax1.scatter(fixed_x1, fixed_x1, s = 20, zorder=2, c = 'yellow', label="")

ax1.set_xlabel('x_t', fontsize=20)
ax1.set_ylabel('x_t+1', fontsize=20, labelpad=-2)

#ax1.set_xlim(T-100,T)
ax1.set_xlim(0,1.1)
ax1.set_ylim(0,1.1)

#ax.legend(["理論値","RC出力値"], prop={"family":"MS Gothic", "size":17}, loc = "lower center", frameon = False, ncol = 2)
#ax1.legend(prop={"size":17}, loc = "upper center", frameon = False, ncol = 2)
ax1.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)
plt.show()

fig.savefig('logi_ctrl.pdf', bbox_inches='tight', pad_inches=0.05)
fig.savefig('logi_ctrl.png', bbox_inches='tight', pad_inches=0.05)