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
import networkx as nx
from numpy.core.fromnumeric import reshape

alpha = 0.8
reservoir_node = 200
input_node = 100
T = 100000                   # 学習ステップ数
predict_T = 100              # 再帰予測のステップ数
ctrl_T = 100                  # 固定点の制御ステップ数

reservoir = np.zeros((reservoir_node,1))
old_reservoir = reservoir
Wout = np.ones((1,reservoir_node))
u = 0.0
o = 0.0

standard_value = 0.001      # 固定点判定(RCの入力と出力の差)の閾値

# ロジスティック写像
logistic_a = 3.9    # 係数
logix = 0.1         # 初期値
old_logix = logix

aa = 0.0
aa_t = np.array([])
original_signal = np.array([])         # ロジスティック写像
fixpoint = 0.744
while aa <= 1.0 :
    logix = logistic_a * aa * (1 - aa)
    aa_t = np.append(aa_t, aa)
    original_signal = np.append(original_signal, logix)

    aa = aa + 0.001
    

result = np.stack([aa_t, original_signal], axis=1)
np.savetxt('original_signal.csv', result, delimiter=',', header='t,original', comments='')

for a in range (100):
    logix = logistic_a * old_logix * (1 - old_logix)
    old_logix = logix

# Wout生成(学習) ---------------------------------------------------------------------------------
def make_Wout():
    global u, o, logix, old_logix, reservoir, old_reservoir, Win, W, Wout

    #Wout更新用変数
    delta = 0.00001
    mu = 0.95
    e = 0.0
    P = np.identity(reservoir_node) / delta

    #出力重み学習
    logi_plot = np.array([])
    RC_plot = np.array([])
    t = np.array([])
    for a in range(0,T+1):
        # x1の予測
        u = logix
        reservoir = (1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))
        o = Wout @ reservoir
        
        #Wout更新
        logix = logistic_a * old_logix * (1 - old_logix)
        old_logix = logix

        target = logix
        e = target - o
        P = (P - (P @ reservoir @ reservoir.T @ P)/(mu + reservoir.T @ P @ reservoir)) / mu
        L = P @ reservoir / (mu + reservoir.T @ P @ reservoir)
        Wout = Wout + L.T * e

        old_reservoir = reservoir
        mu = (1-0.01)*mu + 0.01

        logi_plot = np.append(logi_plot, logix)
        o = Wout @ reservoir
        RC_plot = np.append(RC_plot, o)
        t = np.append(t, a)

    result = np.stack([t, logi_plot, RC_plot], axis=1)
    np.savetxt('learn_result.csv', result, delimiter=',', header='t,original,RCoutput', comments='')
    np.savetxt('W_out.csv', Wout, delimiter=',')
    np.savetxt('reservoir.csv', reservoir, delimiter=',')

    fig = plt.figure(figsize = (12, 5), dpi = 100)
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(t, logi_plot, lw = 2, zorder=1, c = 'orange', label="")
    ax1.plot(t, RC_plot, lw = 2, zorder=2, c = 'blue', label="")

    ax1.set_xlabel('t', fontsize=20)
    ax1.set_ylabel('x', fontsize=20, labelpad=-2)
    ax1.set_xlim(0, T)

    ax1.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)

    fig.savefig('1step_pre.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig('1step_pre.png', bbox_inches='tight', pad_inches=0.05)

    print("Wout生成 & システム同定 終了")

# 固定点安定化 ------------------------------------------------------------------------------
def control():
    global u, o, logix, old_logix, reservoir, old_reservoir, Win, W, Wout

    ctrl = np.array([])
    ctrl_time = np.array([])

    u = -1
    while standard_value < abs(u - o):
        # x1の予測
        u = o
        reservoir = (1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))
        o = Wout @ reservoir
    
        old_reservoir = reservoir

        if  (o<0.0) or (o>1.0):
            print("予測値が0~1の範囲外です")
            break
    fixpt = o
    print("固定点", fixpt)

    # ヤコビ行列 A = ∂x_j / ∂x_k と制御ゲインKの算出
    I = np.identity(reservoir_node)
    A = np.identity(reservoir_node) * (1-alpha) # ヤコビ行列
    K = np.zeros((reservoir_node,reservoir_node)) # 制御ゲイン

    j = 0  
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
                    A[j,k] = alpha * W[j,k] / pow(np.cosh(Win[j]*o + aaa), 2)
            k = k+1
        j = j+1
    with open('logi_A.csv','w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(A)
    K = -(np.linalg.inv(I-A)) @ A
    with open('logi_K.csv','w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(K)

    # 安定化システム観察
    input = np.array([])
    for a in range (0,2000):
        # x1の予測
        u = a/1000
        reservoir = (I-K)@((1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))) + K@old_reservoir
        o = Wout @ reservoir

        # 予測結果plot
        ctrl = np.append(ctrl, o)
        input = np.append(input, u)
    np.savetxt('ctrl_system_watch.csv', ctrl, delimiter=',')

    # 安定化
    t = np.array([])
    tt = 0
    o = fixpt
    for a in range (0,ctrl_T+1):
        # x1の予測
        u = o
        reservoir = (I-K)@((1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))) + K@old_reservoir
        o = Wout @ reservoir

        old_reservoir = reservoir

        # 予測結果plot
        ctrl_time = np.append(ctrl_time, o)
        t = np.append(t, tt)
        tt = tt + 1
    print("制御後", ctrl_time)
    np.savetxt('ctrl_result.csv', ctrl_time, delimiter=',')

    fig = plt.figure(figsize = (6, 6), dpi = 100)
    ax2 = fig.add_subplot(1,1,1)
    ax2.scatter(input, ctrl, s = 5, zorder=1, c = 'blue', label="")
    ax2.plot(ctrl_time[:-1], ctrl_time[1:], lw = 2, zorder=2, c = 'red', label="")

    ax2.set_xlabel('u', fontsize=20)
    ax2.set_ylabel('o', fontsize=20, labelpad=-2)
    ax2.set_xlim(0,1.0)
    ax2.set_ylim(0,1.0)

    ax2.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)

    fig.savefig('ctrl_system_watch.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig('ctrl_system_watch.png', bbox_inches='tight', pad_inches=0.05)

    # control systemが安定かどうか判断 --------------------------------
    J = np.array([])
    J = (I-K) @ A + K
    J_eigen = np.linalg.eigvals(J)
    print("固有値",J_eigen)
    np.savetxt('J_eigen.csv', J_eigen, delimiter=',')

# --------------------------------------------------------------------------------------------------------------

# W 読み込み
W = np.loadtxt('W.csv', delimiter=',', dtype='float')
spectral_radius = max(abs(np.linalg.eigvals(W)))
print('スペクトル半径', spectral_radius)

# Win 読み込み
Win = np.loadtxt('W_in.csv', dtype='float')
Win = np.reshape(Win, (reservoir_node, 1))

# Wout 学習
#make_Wout()
reservoir = np.loadtxt('reservoir.csv', delimiter=',', dtype='float')
reservoir = np.reshape(reservoir, (reservoir_node, 1))
old_reservoir = reservoir
Wout = np.loadtxt('W_out.csv', delimiter=',', dtype='float')
old_logix = np.loadtxt('learn_result.csv', delimiter=',', dtype='float', skiprows=T+1, usecols=1)
o = Wout @ reservoir

# stability tranceformation
control()