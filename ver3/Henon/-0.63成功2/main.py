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
reservoir_node = 400
input_node = int(reservoir_node/2)
T = 100000                   # 学習ステップ数
predict_T = 100              # 再帰予測のステップ数
ctrl_T = 100                  # 固定点の制御ステップ数

reservoir = np.zeros((reservoir_node,1))
old_reservoir = reservoir
Wout = np.ones((1,reservoir_node))
u = 0.0
o = 0.0 

standard_value = 0.01      # 固定点判定(RCの入力と出力の差)の閾値
reservoir_value = 10      # 固定点判定(リザバーの内部状態の差)の閾値

# エノン写像
Hx = 0.1
Hy = 0.1
old_Hx = Hx
old_Hy = Hy
Ha = 1.4
Hb = 0.3

original_Hx = np.array([])
original_Hy = np.array([])

fix_Hx = (-7+(609)**0.5)/28
fix_Hy = (-7+(609)**0.5)/28*Hb
#fix_Hx = (-7-(609)**0.5)/28
#fix_Hy = (-7-(609)**0.5)/28*Hb

for a in range (1000):
    Hx = 1 - Ha*old_Hx*old_Hx + old_Hy
    Hy = Hb*old_Hx
    old_Hx = Hx
    old_Hy = Hy

    if a > 100:
        original_Hx = np.append(original_Hx, Hx)
        original_Hy = np.append(original_Hy, Hy)

original_signal = np.stack([original_Hx, original_Hy], axis=1)
np.savetxt('original_signal.csv', original_signal, delimiter=',', header='Hx,Hy', comments='')

# W生成 --------------------------------------------------------------------------------------------------
def make_W():
    p = 0.05     #辺を生成する確率

    # erdos_renyi グラフ作成
    G = nx.erdos_renyi_graph(reservoir_node, p, directed = 1)

    ary = nx.to_numpy_array(G)
    normal_distribution = np.random.normal(0.0, 1.0, (reservoir_node, reservoir_node))
    W = normal_distribution*ary
    spectral_radius = max(abs(np.linalg.eigvals(W)))
    W = W / spectral_radius * 0.9999
    with open('W.csv','w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(W)

    fig = plt.figure(figsize = (8, 8), dpi = 100)

    #pos = nx.spring_layout(G)
    pos = nx.random_layout(G)

    nx.draw_networkx(G, pos)
    plt.axis("off")
    #plt.show()
    fig.savefig('W_network.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig('W_network.png', bbox_inches='tight', pad_inches=0.05)

    #print(G.degree)
    #print(nx.degree_histogram(G))

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

    print("W生成 終了")

# Win生成 ---------------------------------------------------------------------------------------
def make_Win():
    win = 0.5

    # Win = ±winの2値ランダム
    Win = (np.random.randint(0, 2, (reservoir_node,1)) * 2 - 1) * win
    Win[:reservoir_node-input_node] = 0.0

    with open('W_in.csv','w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(Win)
    
    print("Win生成 終了")

# Wout生成(学習) ---------------------------------------------------------------------------------
def make_Wout():
    global u, o, Hx, old_Hx, Hy, old_Hy,reservoir, old_reservoir, Win, W, Wout

    #Wout更新用変数
    delta = 0.00001
    mu = 0.95
    e = 0.0
    P = np.identity(reservoir_node) / delta

    for a in range(0,100):
        u = Hx
        reservoir = (1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))
        old_reservoir = reservoir

        Hx = 1 - Ha*old_Hx*old_Hx + old_Hy
        Hy = Hb*old_Hx
        old_Hx = Hx
        old_Hy = Hy

    #出力重み学習
    Hx_plot = np.array([])
    RC_plot = np.array([])
    error = np.array([])
    t = np.array([])
    for a in range(0,T+1):
        # x1の予測
        u = Hx
        reservoir = (1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))
        o = Wout @ reservoir
        
        #Wout更新
        Hx = 1 - Ha*old_Hx*old_Hx + old_Hy
        Hy = Hb*old_Hx
        old_Hx = Hx
        old_Hy = Hy

        target = Hx
        e = target - o
        P = (P - (P @ reservoir @ reservoir.T @ P)/(mu + reservoir.T @ P @ reservoir)) / mu
        L = P @ reservoir / (mu + reservoir.T @ P @ reservoir)
        Wout = Wout + L.T * e

        old_reservoir = reservoir
        mu = (1-0.01)*mu + 0.01

        Hx_plot = np.append(Hx_plot, Hx)
        o = Wout @ reservoir
        RC_plot = np.append(RC_plot, o)
        t = np.append(t, a)
        e = abs(target - o)
        error = np.append(error, e)

    result = np.stack([t, Hx_plot, RC_plot, error], axis=1)
    np.savetxt('learn_result.csv', result, delimiter=',', header='t,original,RCoutput,absolute error', comments='')
    np.savetxt('W_out.csv', Wout, delimiter=',')
    np.savetxt('reservoir.csv', reservoir, delimiter=',')

    fig = plt.figure(figsize = (12, 5), dpi = 100)
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(t, Hx_plot, lw = 2, zorder=1, c = 'orange', label="")
    ax1.plot(t, RC_plot, lw = 2, zorder=2, c = 'blue', label="")

    ax1.set_xlabel('t', fontsize=20)
    ax1.set_ylabel('x', fontsize=20, labelpad=-2)
    #ax1.set_xlim(0, T)
    ax1.set_xlim(T-100, T)

    ax1.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)

    fig.savefig('1step_pre.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig('1step_pre.png', bbox_inches='tight', pad_inches=0.05)

    fig = plt.figure(figsize = (12, 5), dpi = 100)
    ax2 = fig.add_subplot(1,1,1)
    ax2.plot(t, error, lw = 2, zorder=1, c = 'black', label="")

    ax2.set_xlabel('t', fontsize=20)
    ax2.set_ylabel('absolute error', fontsize=20, labelpad=-2)
    ax2.set_xlim(0, T)
    ax2.set_yscale('log')
    #ax2.set_xscale('log')

    ax2.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)

    fig.savefig('1step_error.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig('1step_error.png', bbox_inches='tight', pad_inches=0.05)

    print("Wout生成 & システム同定 終了")
    print("o", o)

# 再帰予測 --------------------------------------------------------------------------------------
def make_reprediction():
    global u, o, Hx, old_Hx, Hy, old_Hy,reservoir, old_reservoir, Win, W, Wout

    Hx_plot = np.array([])
    RC_plot = np.array([])
    t = np.array([])
    print("saiki yosoku u", o)
    for a in range(0, predict_T+1):
        # x1の予測
        u = o
        reservoir = (1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))
        o = Wout @ reservoir

        Hx = 1 - Ha*old_Hx*old_Hx + old_Hy
        Hy = Hb*old_Hx
        old_Hx = Hx
        old_Hy = Hy

        Hx_plot = np.append(Hx_plot, Hx)
        RC_plot = np.append(RC_plot, o)
        t = np.append(t, a)
    
        old_reservoir = reservoir
    
    fig = plt.figure(figsize = (7, 5), dpi = 100)
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(t, Hx_plot, lw = 2, zorder=1, c = 'green', label="")
    ax1.plot(t, RC_plot, lw = 2, zorder=2, c = 'red', label="")

    ax1.set_xlabel('t', fontsize=20)
    ax1.set_ylabel('x', fontsize=20, labelpad=-2)
    ax1.set_xlim(0, predict_T)
    #ax1.set_xlim(predict_T-100, predict_T)
    #ax1.set_ylim(-1.6,1.6)

    ax1.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)

    fig.savefig('recursive_pre.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig('recursive_pre.png', bbox_inches='tight', pad_inches=0.05)

    fig = plt.figure(figsize = (5, 5), dpi = 100)
    ax2 = fig.add_subplot(1,1,1)
    ax2.scatter(Hx_plot[:-1], Hx_plot[1:], s=10, zorder=1, c = 'green', label="")
    ax2.scatter(RC_plot[:-1], RC_plot[1:], s=3, zorder=3, c = 'red', label="")
    ax2.scatter(fix_Hx, fix_Hx, s=20, zorder=2, c='yellow', label="")

    ax2.set_xlabel('x(t)', fontsize=20)
    ax2.set_ylabel('x(t+1)', fontsize=20, labelpad=-2)
    #ax2.set_xlim(-1.5,1.5)
    #ax2.set_ylim(-1.5,1.5)

    ax1.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)

    fig.savefig('delay_coordinate.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig('delay_coordinate.png', bbox_inches='tight', pad_inches=0.05)

    print("o", o)

# 固定点探索 ------------------------------------------------------------------------------
def fixpoint_search():
    global u, o, Hx, old_Hx, Hy, old_Hy,reservoir, old_reservoir, Win, W, Wout

    Hx_plot = np.array([])
    RC_plot = np.array([])
    t = np.array([])
    reservoir_plot = np.array([])
    tt = 0
    u = -1
    while standard_value < abs(u - o):
        # x1の予測
        u = o
        reservoir = (1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))
        o = Wout @ reservoir

        Hx = 1 - Ha*old_Hx*old_Hx + old_Hy
        Hy = Hb*old_Hx
        old_Hx = Hx
        old_Hy = Hy

        Hx_plot = np.append(Hx_plot, Hx)
        RC_plot = np.append(RC_plot, o)
        t = np.append(t, tt)

        tt = tt + 1
    
        old_reservoir = reservoir

        if  (o<-10.0) or (o>10.0):
            print("予測値が-10~10の範囲外です")
            break
    print("固定点", o)
    
    result = np.stack([t, Hx_plot, RC_plot], axis=1)
    np.savetxt('fixpoint_search_result.csv', result, delimiter=',', header='t,original,RCoutput', comments='')

    fig = plt.figure(figsize = (12, 5), dpi = 100)
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(t, RC_plot, lw = 2, zorder=2, c = 'blue', label="")

    ax1.set_xlabel('t', fontsize=20)
    ax1.set_ylabel('x', fontsize=20, labelpad=-2)
    #ax1.set_xlim(0, tt-1)
    ax1.set_xlim(tt-200, tt-1)

    ax1.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)

    fig.savefig('fixpoint_search.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig('fixpoint_search.png', bbox_inches='tight', pad_inches=0.05)

# 固定点安定化 ------------------------------------------------------------------------------
def control():
    global u, o, Hx, old_Hx, Hy, old_Hy,reservoir, old_reservoir, Win, W, Wout

    # 固定点探索
    RC_plot = np.array([o])
    t = np.array([0])
    tt = 1
    reservoir_dif = 100
    print("koteiten tansaku u", o)
    while reservoir_value < reservoir_dif:
        u = o
        reservoir = (1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))
        o = Wout @ reservoir

        reservoir_plot = np.hstack([old_reservoir, reservoir, abs(old_reservoir-reservoir)])
        reservoir_dif = np.sum(abs(old_reservoir-reservoir))
        if reservoir_dif < 10:
            print(tt, reservoir_dif, abs(RC_fix-o))
        old_reservoir = reservoir

        RC_plot = np.append(RC_plot, o)
        t = np.append(t, tt)
        tt = tt + 1

        if tt>500000:
            print("探索ステップ数が500000を超えました")
            RC_fix = None
            sys.exit()
            #break
        RC_fix = o
        
    print("固定点", o)

    result = np.stack([t, RC_plot], axis=1)
    np.savetxt('fixpoint_search_result.csv', result, delimiter=',', header='t,RCoutput', comments='')
    np.savetxt('reservoir_watch.csv', reservoir_plot, delimiter=',', comments='')

    fig = plt.figure(figsize = (12, 5), dpi = 100)
    ax1 = fig.add_subplot(1,1,1)
    ax1.axhline(y=fix_Hx, lw=1, zorder=1, c = 'yellow')
    ax1.plot(t, RC_plot, lw = 2, zorder=2, c = 'red', label="")

    ax1.set_xlabel('t', fontsize=20)
    ax1.set_ylabel('x', fontsize=20, labelpad=-2)
    ax1.set_xlim(tt-400, tt-1)
    #ax1.set_xlim(0, tt-1)
    ax1.set_ylim(-2,2)

    ax1.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)

    fig.savefig('fixpoint_search.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig('fixpoint_search.png', bbox_inches='tight', pad_inches=0.05)

    fig = plt.figure(figsize = (5, 5), dpi = 100)
    ax2 = fig.add_subplot(1,1,1)
    ax2.scatter(original_Hx[:-1], original_Hx[1:], s=10, zorder=1, c = 'green', label="")
    ax2.scatter(RC_plot[:-1], RC_plot[1:], s=3, zorder=2, c = 'red', label="")
    ax2.scatter(fix_Hx, fix_Hx, s=20, zorder=3, c='yellow', label="")
    ax2.scatter(RC_fix, RC_fix, s=20, zorder=3, c='black', label="")

    ax2.set_xlabel('x(t)', fontsize=20)
    ax2.set_ylabel('x(t+1)', fontsize=20, labelpad=-2)
    ax2.set_xlim(-1.5,1.5)
    ax2.set_ylim(-1.5,1.5)

    ax1.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)

    fig.savefig('delay_coordinate.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig('delay_coordinate.png', bbox_inches='tight', pad_inches=0.05)

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
                    A[j,k] = alpha * W[j,k] / pow(np.cosh(Win[j]*o + aaa), 2)
            k = k+1
        j = j+1
    A = A @ (I + Win@Wout)
    with open('Jacobi.csv','w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(A)
    K = -(np.linalg.inv(I-A)) @ A
    with open('K.csv','w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(K)

    # 安定化
    ctrl = np.array([])
    t = np.array([])
    tt = 0
    for a in range (0,ctrl_T+1):
        # x1の予測
        u = o
        reservoir = (I-K)@((1-alpha)*old_reservoir + alpha*(np.tanh(Win*u + W@old_reservoir))) + K@old_reservoir
        o = Wout @ reservoir

        old_reservoir = reservoir

        # 予測結果plot
        ctrl = np.append(ctrl, o)
        t = np.append(t, tt)
        tt = tt + 1
    print("制御後", ctrl)
    np.savetxt('ctrl_result.csv', ctrl, delimiter=',')

    fig = plt.figure(figsize = (5, 5), dpi = 100)
    ax1 = fig.add_subplot(1,1,1)
    Hx_plot = np.loadtxt('learn_result.csv', delimiter=',', dtype='float', skiprows=1, usecols=1)
    ax1.scatter(Hx_plot[:-1], Hx_plot[1:], s = 1, zorder=1, c = 'green', label="")
    ax1.plot(ctrl[:-1], ctrl[1:], lw = 2, zorder=2, c = 'red', label="")
    ax1.scatter(fix_Hx, fix_Hx, s = 25, zorder=3, c = 'yellow', label="")
    ax1.scatter(RC_fix, RC_fix, s = 25, zorder=3, c = 'black', label="")
    ax1.set_xlabel('x_t', fontsize=20)
    ax1.set_ylabel('x_t+1', fontsize=20, labelpad=-2)
    ax1.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)

    fig.savefig('ctrl.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig('ctrl.png', bbox_inches='tight', pad_inches=0.05)

    fig = plt.figure(figsize = (5, 5), dpi = 100)
    ax2 = fig.add_subplot(1,1,1)
    ax2.axhline(y=fix_Hx, lw=1, zorder=1, c = 'yellow')
    ax2.axhline(y=RC_fix, lw=1, zorder=1, c = 'black')
    ax2.plot(t, ctrl, lw = 2, zorder=2, c = 'red', label="")
    ax2.set_xlabel('t', fontsize=20)
    ax2.set_ylabel('x', fontsize=20, labelpad=-2)
    ax2.set_xlim(0, ctrl_T)
    #ax2.set_ylim(-1.5, 1.5)

    ax2.tick_params(direction = "in", labelsize=17, pad = 4, length = 6)

    fig.savefig('ctrl_time.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig('ctrl_time.png', bbox_inches='tight', pad_inches=0.05)

# W 読み込み
#make_W()
W = np.loadtxt('W.csv', delimiter=',', dtype='float')
spectral_radius = max(abs(np.linalg.eigvals(W)))
print('スペクトル半径', spectral_radius)

# Win 読み込み
#make_Win()
Win = np.loadtxt('W_in.csv', dtype='float')
Win = np.reshape(Win, (reservoir_node, 1))

# Wout 学習
#make_Wout()
reservoir = np.loadtxt('reservoir.csv', delimiter=',', dtype='float')
reservoir = np.reshape(reservoir, (reservoir_node, 1))
old_reservoir = reservoir
Wout = np.loadtxt('W_out.csv', delimiter=',', dtype='float')
Wout = Wout.reshape([1,reservoir_node])
old_logix = np.loadtxt('learn_result.csv', delimiter=',', dtype='float', skiprows=T+1, usecols=1)
o = Wout @ reservoir

# 再帰予測 (RCの出力を次の入力に)
make_reprediction()

# 固定点探索
#fixpoint_search()

# stability tranceformation
control()