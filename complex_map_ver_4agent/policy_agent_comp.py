#from dezero import Model
#from dezero import optimizers
#import dezero.functions as F
#import dezero.layers as L
import numpy as np


import random
#from gym import spaces
#from gym.spaces import Discrete
from dezero import Model



class Policy1:
    def __init__(self, grid, eta_size=2):
        """

        """
        self.action_size = 4
        self.eta_size = eta_size
        self.state_size = grid.shape[0] * grid.shape[1]#これはmapを変更したらその都度変えないといけない
        self.beta = 0.5#softmaxの逆温度(これが大きいほど、方策は決定的になりやすい)
        theta1_a = [1] * self.action_size
        theta1_nexteta = []
        for i in range(self.eta_size):
            theta1_nexteta.append(theta1_a)
        theta1 = []
        for i in range(self.state_size):
            theta1.append(theta1_nexteta)

        theta1 = np.array(theta1, dtype=np.float64)
        for nexteta_index in range(self.eta_size):
            theta1[0][nexteta_index] = np.array([np.nan, 1, 1, np.nan])
        for nexteta_index in range(self.eta_size):
            theta1[1][nexteta_index] = np.array([np.nan, 1, 1, 1])
            theta1[2][nexteta_index] = np.array([np.nan, 1, 1, 1])
        for nexteta_index in range(self.eta_size):
            theta1[3][nexteta_index] = np.array([np.nan, np.nan, 1, 1])
        for nexteta_index in range(self.eta_size):
            theta1[4][nexteta_index] = np.array([1, 1, 1, np.nan])
            theta1[8][nexteta_index] = np.array([1, 1, 1, np.nan])
            theta1[12][nexteta_index] = np.array([1, 1, 1, np.nan])
            theta1[16][nexteta_index] = np.array([1, 1, 1, np.nan])
            theta1[20][nexteta_index] = np.array([1, 1, 1, np.nan])
        for nexteta_index in range(self.eta_size):
            theta1[7][nexteta_index] = np.array([1, np.nan, 1, 1])
            theta1[11][nexteta_index] = np.array([1, np.nan, 1, 1])
            theta1[15][nexteta_index] = np.array([1, np.nan, 1, 1])
            theta1[19][nexteta_index] = np.array([1, np.nan, 1, 1])
            theta1[23][nexteta_index] = np.array([1, np.nan, 1, 1])
        for nexteta_index in range(self.eta_size):
            theta1[24][nexteta_index] = np.array([1, 1, np.nan, np.nan])
        for nexteta_index in range(self.eta_size):
            theta1[27][nexteta_index] = np.array([1, np.nan, np.nan, 1])
        for nexteta_index in range(self.eta_size):
            theta1[25][nexteta_index] = np.array([1, 1, np.nan, 1])
            theta1[26][nexteta_index] = np.array([1, 1, np.nan, 1])

        """
        theta1は(self.state_size, self.eta_size, self.action_size)の3次元配列
        self.theta1[state_index][eta_index][action_index]でθ(s1,a1,η2)を参照できる
        """
        self.theta1 = theta1
    def softmax_probs(self):
        beta = self.beta
        pi = np.zeros((self.state_size, self.eta_size, self.action_size))
        pi = np.array(pi, dtype=np.float64)

        max_theta = np.nanmax(self.theta1, axis=(1,2), keepdims=True)
        exp_theta = np.exp(beta * (self.theta1 - max_theta))
        #print(f'exp_theta1:{exp_theta}')
        for state_index in range(self.state_size):
            pi[state_index, :] = exp_theta[state_index, :] / np.nansum(exp_theta[state_index, :])
        pi = np.nan_to_num(pi)#nanを0に変換
        return pi

class Policy2:
    def __init__(self, grid, eta_size=2):
        """

        """
        self.action_size = 4
        self.eta_size = eta_size
        self.state_size = grid.shape[0] * grid.shape[1]
        self.beta = 1.0#softmaxの逆温度(これが大きいほど、方策は決定的になりやすい)

        theta2_a = [1] * self.action_size
        theta2_nexteta = []
        for i in range(self.eta_size):
            theta2_nexteta.append(theta2_a)
        theta2_eta = []
        for i in range(self.eta_size):
            theta2_eta.append(theta2_nexteta)
        theta2 = []
        for i in range(self.state_size):
            theta2.append(theta2_eta)
        theta2 = np.array(theta2, dtype=np.float64)

        for eta_index in range(self.eta_size):
            for nexteta_index in range(self.eta_size):
                theta2[0][eta_index][nexteta_index] = np.array([np.nan, 1, 1, np.nan])
        for eta_index in range(self.eta_size):
            for nexteta_index in range(self.eta_size):
                theta2[1][eta_index][nexteta_index] = np.array([np.nan, 1, 1, 1])
                theta2[2][eta_index][nexteta_index] = np.array([np.nan, 1, 1, 1])
        for eta_index in range(self.eta_size):
            for nexteta_index in range(self.eta_size):
                theta2[3][eta_index][nexteta_index] = np.array([np.nan, np.nan, 1, 1])
        for eta_index in range(self.eta_size):
            for nexteta_index in range(self.eta_size):
                theta2[4][eta_index][nexteta_index] = np.array([1, 1, 1, np.nan])
                theta2[8][eta_index][nexteta_index] = np.array([1, 1, 1, np.nan])
                theta2[12][eta_index][nexteta_index] = np.array([1, 1, 1, np.nan])
                theta2[16][eta_index][nexteta_index] = np.array([1, 1, 1, np.nan])
                theta2[20][eta_index][nexteta_index] = np.array([1, 1, 1, np.nan])
        for eta_index in range(self.eta_size):
            for nexteta_index in range(self.eta_size):
                theta2[7][eta_index][nexteta_index] = np.array([1, np.nan, 1, 1])
                theta2[11][eta_index][nexteta_index] = np.array([1, np.nan, 1, 1])
                theta2[15][eta_index][nexteta_index] = np.array([1, np.nan, 1, 1])
                theta2[19][eta_index][nexteta_index] = np.array([1, np.nan, 1, 1])
                theta2[23][eta_index][nexteta_index] = np.array([1, np.nan, 1, 1])
        for eta_index in range(self.eta_size):
            for nexteta_index in range(self.eta_size):
                theta2[24][eta_index][nexteta_index] = np.array([1, 1, np.nan, np.nan])
        for eta_index in range(self.eta_size):
            for nexteta_index in range(self.eta_size):
                theta2[27][eta_index][nexteta_index] = np.array([1, np.nan, np.nan, 1])
        for eta_index in range(self.eta_size):
            for nexteta_index in range(self.eta_size):
                theta2[25][eta_index][nexteta_index] = np.array([1, 1, np.nan, 1])
                theta2[26][eta_index][nexteta_index] = np.array([1, 1, np.nan, 1])
        
        
        """
        theta2は(self.state_size, self.eta_size, self.eta_size, self.action_size)の4次元配列
        self.theta1[state_index][eta_index][nexteta_index][action_index]でθ(st, ηt, at, η(t+1))を参照できる
        """
        self.theta2 = theta2

    
    def softmax_probs(self):
        beta = self.beta
        pi = np.zeros((self.state_size, self.eta_size, self.eta_size, self.action_size))
        pi = np.array(pi, dtype=np.float64)
        # 数値安定化のために最大値を引く
        max_theta = np.nanmax(self.theta2, axis=(2, 3), keepdims=True)  # (nexteta, action) に渡る最大値
        exp_theta = np.exp(beta * (self.theta2 - max_theta))  # 数値安定化した theta

        #print(f'exp_theta2:{exp_theta}')
        for state_index in range(self.state_size):
            for eta_index in range(self.eta_size):
                numerator = exp_theta[state_index][eta_index]  # 2次元部分を抜き出す
                denominator = np.nansum(numerator)  # 分母：全要素の合計
                if denominator > 0:  # 分母が 0 の場合を避ける
                    pi[state_index][eta_index] = numerator / denominator
                elif denominator == 0:
                    raise ValueError('softmax_probs in Policy2でオーバーフローが発生')
        pi = np.nan_to_num(pi)# nanを0に変換
        return pi





class Agent:
    def __init__(self, kappa, eta_size, Lambda, alpha_risk, grid, grid_shape, eta_space, action_size=4, gamma=0.9999, lr=0.002, epsilon=1e-8):
        """
        エージェントの初期化
        :param action_size: アクション空間のサイズ
        :param state_size: 状態空間の次元数
        :param gamma: 割引率
        :param lr: 初期学習率
        :param epsilon: 数値安定化のための定数
        """
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.action_size = action_size
        self.state_size = grid.shape[0] * grid.shape[1]
        self.eta_size = eta_size
        self.Lambda = Lambda
        self.alpha_risk = alpha_risk
        self.kappa = kappa
        self.grid = grid 
        self.grid_shape = grid_shape
        self.eta_space = eta_space
        self.memory = []
        self.cost_memory = []
        self.prob_memory = []
        self.pi1 = Policy1(eta_size=self.eta_size, grid=grid)
        self.pi2 = Policy2(eta_size=self.eta_size, grid=grid)


    def get_action_1(self, state_index):
        """
        Policy_1に基づいてactionとIetaを取得
        :param state: 現在の状態
        :return: (選択されたアクション, その確率)
        """
        probs = self.pi1.softmax_probs()[int(state_index)]#eta_size*action_sizeの二次元配列
        # 確率が1になるように正規化 (念のため)
        #print(f'np.sum(probs) in pi1:{np.sum(probs)}')
        ##print(f'probs before divide in pi1:{probs}')
        probs = probs / np.sum(probs)

        #デバッグ用
        #print("Probs:", probs.data)
        if np.any(np.isnan(probs.data)):
            print(f'probs in pi1:{probs}')
            raise ValueError("NaN detected in probability distribution1!")

        action_and_nexteta_index = np.random.choice(len(probs.flatten()), p=probs.flatten())  # 確率に基づいてアクションを選択(action_and_nexteta_indexは0,...,eta_size*action_size-1)
        nexteta_index = action_and_nexteta_index // self.action_size#現状nexteta_indexは0,...,eta_size-1の値を取る
        action = action_and_nexteta_index % self.action_size
        return action, nexteta_index#, probs[action_and_nexteta_index]#action,nexteta_indexはint, probs[action_and_nexteta_index]はvariable(float)

    def get_action_2(self, state_index, eta_index):#state_index, etaはint
        """
        Policy_2に基づいてactionとIetaを取得
        :param state: 現在の状態
        :return: (選択されたアクション, その確率)
        """
        #print(f'self.pi2.softmax_probs() : {self.pi2.softmax_probs()}')
        probs = self.pi2.softmax_probs()[int(state_index)][eta_index]
        # 確率が1になるように正規化 (念のため)
        #print("Probs:", probs.data)
        if np.any(np.isnan(probs.data)):
            print(f'before probs:{probs}')
            raise ValueError("NaN detected in before probability distribution2!")
        #print(f'np.sum(probs) in pi2:{np.sum(probs)}')
        #print(f'probs before divide in pi2:{probs}')
        #print(f'\nstate : {state_index}')
        #print(f'eta_index : {eta_index}')
        #print(f'正規化される前のprobs : {probs}')
        probs = probs / np.sum(probs)
        #print(f'正規化された probs : {probs}\n')
        #デバッグ用
        #print("Probs:", probs.data)
        if np.any(np.isnan(probs.data)):
            print(f'theta2 : {self.pi2.theta2[int(state_index)][eta_index]}, state : {state_index}, eta : {eta_index}')
            print(f'probs: in pi2{probs}')
            raise ValueError("NaN detected in probability distribution2!")

        action_and_nexteta_index = np.random.choice(len(probs.flatten()), p=probs.flatten())  # 確率に基づいてアクションを選択(action_and_nexteta_indexは0,...,eta_size*action_size-1)
        #print(f'action_and_nexteta_index : {action_and_nexteta_index}')#これは正しそう
        #print(f'state:{state}, eta:{eta}')
        #print(type(state))
        #print(type(eta))
        #print('状態とeta')
        #デバッグ用
        #print("Probs:", probs.data)
        if np.any(np.isnan(probs.data)):
            print(f'probs:{probs}')
            raise ValueError("NaN detected in probability distribution2!")

        nexteta_index = action_and_nexteta_index // self.action_size#現状etaは0,...,eta_size-1の値を取るのでのちに修正
        action = action_and_nexteta_index % self.action_size 
        return action, nexteta_index#, probs[action_and_Ieta]#actionはint, probs[action]はvariable(float)


    def add(self, cost, eta_index, nexteta_index, state_index, action, time):
        """
        報酬とアクション確率をメモリに追加
        :param cost: 報酬
        :param prob: アクション確率
        """
        """time=1とtime>1で新しく定義した損失の形が異なるのでそれに伴って新しい損失を定義"""
        if time == 1:
            new_cost = cost + self.gamma * self.Lambda * self.eta_space[nexteta_index]
        elif time > 1:
            new_cost = (self.Lambda / self.alpha_risk) * max(0, cost - self.eta_space[eta_index]) + (1 - self.Lambda) * cost + self.Lambda * self.gamma * self.eta_space[nexteta_index]
        data = {}
        data['eta_index'] = eta_index
        data['nexteta_index'] = nexteta_index
        data['state_index'] = state_index
        data['action'] = action
        data['new_cost'] = new_cost
        data['old_cost'] = cost
        self.memory.append(data)
        #print(f'time={time}, old_cost = {cost}, (new_cost, prob) = {data}')
        #self.cost_memory.append(new_cost)
        #self.prob_memory.append(prob)


    def update_pi1_SGD(self, learning_rate=None, kappa=None):
        """
        方策Policy1のネットワークの更新を行う（動的学習率を適用）
        :param learning_rate: 新しい学習率（指定がなければデフォルトを使用）
        """
        if learning_rate is None:
            learning_rate = self.lr

        if kappa is None:
            kappa = self.kappa

        new_theta1 = self.pi1.theta1
        G = 0
        #print(f'係数:{kappa / (self.state_size * self.action_size * self.eta_size * self.eta_size)}')
        #print(f'探索項 in update_pi1:{loss}')
        #print('new_cost')
        total_cost = 0
        for data in reversed(self.memory):
            new_cost = data['new_cost']
            #print(f' {new_cost},', end='')
            G = new_cost + self.gamma * G
            total_cost += data['old_cost']
            #print(f'updated G:{G}')
        adjustment = 0
        if total_cost > 10:
            adjustment = 0.005
        #目的関数の勾配の更新
        for state_index in range(self.state_size):
            for nexteta_index in range(self.eta_size):
                for action in range(self.action_size):
                    if not(np.isnan(self.pi1.theta1[state_index][nexteta_index][action])):#theta1がnanじゃないところを更新
                        #定義関数の定義
                        i_s = 0
                        i_nexteta = 0
                        i_a = 0
                        data = self.memory[0]
                        if state_index == data['state_index']:
                            i_s = 1
                        if nexteta_index == data['nexteta_index']:
                            i_nexteta = 1
                        if action == data['action']:
                            i_a = 1
                        grad = (i_s * (i_nexteta * i_a + adjustment -self.pi1.softmax_probs()[state_index][nexteta_index][action])) * G
                        #theta１の更新
                        new_theta1[state_index][nexteta_index][action] -= learning_rate * grad
        max_theta = np.nanmax(new_theta1, axis=(1,2), keepdims=True)
        new_theta1 = new_theta1 - max_theta
        self.pi1.theta1 = new_theta1
        """for i in range(self.state_size):
            print(f'self.pi1.theta1:{self.pi1.softmax_probs()[i]}')
"""
    def update_pi2_SGD(self, learning_rate=None, kappa=None):
        """
        方策Policy1のネットワークの更新を行う（動的学習率を適用）
        :param learning_rate: 新しい学習率（指定がなければデフォルトを使用）
        """
        if learning_rate is None:
            learning_rate = self.lr

        if kappa is None:
            kappa = self.kappa


        new_theta2 = self.pi2.theta2
        total_cost = 0
        for data in reversed(self.memory):
            total_cost += data['old_cost']
        adjustment = 0
        if total_cost > 10:
            adjustment = 0.005
        #勾配項の追加
        for step in range(1,len(self.memory)):
            G = 0
            if step != len(self.memory)-1:
                for data in reversed(self.memory[step:]):
                    new_cost = data['new_cost']
                    G = new_cost + self.gamma * G
                G /= self.gamma**(-1*step)
            elif step == len(self.memory)-1:
                data = self.memory[step]
                new_cost = data['new_cost']
                G = new_cost
            #print(f'step:{step}, G:{G}, ')

            #目的関数の勾配の更新
            for state_index in range(self.state_size):
                for eta_index in range(self.eta_size):
                    for nexteta_index in range(self.eta_size):
                        for action in range(self.action_size):
                            if not(np.isnan(self.pi2.theta2[state_index][eta_index][nexteta_index][action])):#theta2がnanじゃないところを更新
                                #定義関数の定義
                                i_s = 0
                                i_eta = 0
                                i_nexteta = 0
                                i_a = 0
                                data = self.memory[step]
                                if state_index == data['state_index']:
                                    i_s = 1
                                if eta_index == data['eta_index']:
                                    i_eta = 1
                                if nexteta_index == data['nexteta_index']:
                                    i_nexteta = 1
                                if action == data['action']:
                                    i_a = 1
                                grad = (i_s * i_eta * (i_nexteta * i_a + adjustment - self.pi2.softmax_probs()[state_index][eta_index][nexteta_index][action])) * G
                                #theta１の更新
                                new_theta2[state_index][eta_index][nexteta_index][action] -= learning_rate * grad
        max_theta = np.nanmax(new_theta2, axis=(2, 3), keepdims=True)  # (nexteta, action) に渡る最大値
        new_theta2 = new_theta2 - max_theta  # 数値安定化した theta
        self.pi2.theta2 = new_theta2
        """for i in range(self.state_size):
            print(f'self.pi2.theta2:{self.pi2.softmax_probs()[i]}')"""

    def update_pi1_SGD_distributed(self, agent_index, P, z, other_agent1, other_agent2, other_agent3, other_agent4, learning_rate=None, kappa=None):
        """
        分散最適化用の推定値の更新
        方策Policy1のネットワークの更新を行う（動的学習率を適用）
        :param learning_rate: 新しい学習率（指定がなければデフォルトを使用
        :P: 重み行列
        :z: 重み行列の固有値１の左固有ベクトルの推定値
        :agent_index: 自身のエージェント番号(1,2,3のどれか)
        :other_agent1,2: agentのclassとエージェント番号(1,2,3)のタプルを渡す
        
        """
        if learning_rate is None:
            learning_rate = self.lr

        if kappa is None:
            kappa = self.kappa

        new_theta1 = np.zeros((self.state_size, self.eta_size, self.action_size))
        G = 0
        #print(f'探索項 in update_pi1:{loss}')
        #print('new_cost')
        total_cost = 0
        for data in reversed(self.memory):
            new_cost = data['new_cost']
            #print(f' {new_cost},', end='')
            G = new_cost + self.gamma * G
            total_cost += data['old_cost']
            #print(f'updated G:{G}')
        adjustment = 0
        if total_cost > 10:
            adjustment = 0.005
        #目的関数の勾配の更新
        for state_index in range(self.state_size):
            for nexteta_index in range(self.eta_size):
                for action in range(self.action_size):
                    if not(np.isnan(self.pi1.theta1[state_index][nexteta_index][action])):#theta1がnanじゃないところを更新
                        #定義関数の定義
                        i_s = 0
                        i_nexteta = 0
                        i_a = 0
                        data = self.memory[0]
                        if state_index == data['state_index']:
                            i_s = 1
                        if nexteta_index == data['nexteta_index']:
                            i_nexteta = 1
                        if action == data['action']:
                            i_a = 1
                        grad = (i_s * (i_nexteta * i_a + adjustment -self.pi1.softmax_probs()[state_index][nexteta_index][action])) * G
                        #theta１の更新
                        new_theta1[state_index][nexteta_index][action] = P[agent_index - 1][agent_index - 1] * self.pi1.theta1[state_index][nexteta_index][action] + P[agent_index - 1][other_agent1[0] - 1] * other_agent1[1].pi1.theta1[state_index][nexteta_index][action] + P[agent_index - 1][other_agent2[0] - 1] * other_agent2[1].pi1.theta1[state_index][nexteta_index][action] + P[agent_index - 1][other_agent3[0] - 1] * other_agent3[1].pi1.theta1[state_index][nexteta_index][action] + P[agent_index - 1][other_agent4[0] - 1] * other_agent4[1].pi1.theta1[state_index][nexteta_index][action]- (learning_rate / z[agent_index - 1]) * grad
                    else:
                        new_theta1[state_index][nexteta_index][action] = np.nan
        max_theta = np.nanmax(new_theta1, axis=(1,2), keepdims=True)
        new_theta1 = new_theta1 - max_theta
        self.pi1.theta1 = new_theta1
        """for i in range(self.state_size):
            print(f'self.pi1.theta1:{self.pi1.softmax_probs()[i]}')
"""
    def update_pi2_SGD_distributed(self, agent_index, P, z, other_agent1, other_agent2, other_agent3, other_agent4, learning_rate=None, kappa=None):
        """
        方策Policy1のネットワークの更新を行う（動的学習率を適用）
        :param learning_rate: 新しい学習率（指定がなければデフォルトを使用）
        """
        if learning_rate is None:
            learning_rate = self.lr

        if kappa is None:
            kappa = self.kappa


        new_theta2 = np.zeros((self.state_size, self.eta_size, self.eta_size, self.action_size))
        total_cost = 0
        for data in reversed(self.memory):
            total_cost += data['old_cost']
        adjustment = 0
        if total_cost > 10:
            adjustment = 0.005
        #勾配項の追加
        for step in range(1,len(self.memory)):
            G = 0
            if step != len(self.memory)-1:
                for data in reversed(self.memory[step:]):
                    new_cost = data['new_cost']
                    G = new_cost + self.gamma * G
                G /= self.gamma**(-1*step)
            elif step == len(self.memory)-1:
                data = self.memory[step]
                new_cost = data['new_cost']
                G = new_cost
            #print(f'step:{step}, G:{G}, ')
            #目的関数の勾配の更新
            for state_index in range(self.state_size):
                for eta_index in range(self.eta_size):
                    for nexteta_index in range(self.eta_size):
                        for action in range(self.action_size):
                            if not(np.isnan(self.pi2.theta2[state_index][eta_index][nexteta_index][action])):#theta2がnanじゃないところを更新
                                #定義関数の定義
                                i_s = 0
                                i_eta = 0
                                i_nexteta = 0
                                i_a = 0
                                data = self.memory[step]
                                if state_index == data['state_index']:
                                    i_s = 1
                                if eta_index == data['eta_index']:
                                    i_eta = 1
                                if nexteta_index == data['nexteta_index']:
                                    i_nexteta = 1
                                if action == data['action']:
                                    i_a = 1
                                grad = (i_s * i_eta * (i_nexteta * i_a + adjustment - self.pi2.softmax_probs()[state_index][eta_index][nexteta_index][action])) * G
                                #theta１の更新
                                new_theta2[state_index][eta_index][nexteta_index][action] -= (learning_rate / z[agent_index - 1]) * grad
                                
                                if step == 1:
                                    new_theta2[state_index][eta_index][nexteta_index][action] += P[agent_index - 1][agent_index - 1] * self.pi2.theta2[state_index][eta_index][nexteta_index][action] + P[agent_index - 1][other_agent1[0] - 1] * other_agent1[1].pi2.theta2[state_index][eta_index][nexteta_index][action] + P[agent_index - 1][other_agent2[0] - 1] * other_agent2[1].pi2.theta2[state_index][eta_index][nexteta_index][action] + P[agent_index - 1][other_agent3[0] - 1] * other_agent3[1].pi2.theta2[state_index][eta_index][nexteta_index][action] + P[agent_index - 1][other_agent4[0] - 1] * other_agent4[1].pi2.theta2[state_index][eta_index][nexteta_index][action]
                            else:
                                new_theta2[state_index][eta_index][nexteta_index][action] = np.nan
        max_theta = np.nanmax(new_theta2, axis=(2, 3), keepdims=True)  # (nexteta, action) に渡る最大値
        new_theta2 = new_theta2 - max_theta  # 数値安定化した theta
        self.pi2.theta2 = new_theta2
        """for i in range(self.state_size):
            print(f'self.pi2.theta2:{self.pi2.softmax_probs()[i]}')"""