from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
import numpy as np


import random
#from gym import spaces
#from gym.spaces import Discrete

import dezero.functions as F
import dezero.layers as L
from dezero import Model



class Policy():
    def __init__(self, grid_shape):
        """

        """
        self.action_size = 4
        self.state_size = grid_shape[0] * grid_shape[1]
        self.beta = 0.5#softmaxの逆温度(これが大きいほど、方策は決定的になりやすい)
        theta_a = [1] * self.action_size
        theta = []
        for i in range(self.state_size):
            theta.append(theta_a)

        theta = np.array(theta, dtype=np.float64)
        if self.state_size == 16:
            theta[0] = np.array([np.nan, 1, 1, np.nan])
            theta[1] = np.array([np.nan, 1, 1, 1])
            theta[2] = np.array([np.nan, 1, 1, 1])
            theta[3] = np.array([np.nan, np.nan, 1, 1])
            theta[4] = np.array([1, 1, 1, np.nan])
            theta[8] = np.array([1, 1, 1, np.nan])
            theta[7] = np.array([1, np.nan, 1, 1])
            theta[11] = np.array([1, np.nan, 1, 1])
            theta[12] = np.array([1, 1, np.nan, np.nan])
            theta[15] = np.array([1, np.nan, np.nan, 1])
            theta[13] = np.array([1, 1, np.nan, 1])
            theta[14] = np.array([1, 1, np.nan, 1])
        elif self.state_size == 48:
            theta[0] = np.array([np.nan, 1, 1, np.nan])
            theta[1] = np.array([np.nan, 1, 1, 1])
            theta[2] = np.array([np.nan, 1, 1, 1])
            theta[3] = np.array([np.nan, 1, 1, 1])
            theta[4] = np.array([np.nan, 1, 1, 1])
            theta[5] = np.array([np.nan, np.nan, 1, 1])
            theta[6] = np.array([1, 1, 1, np.nan])
            theta[12] = np.array([1, 1, 1, np.nan])
            theta[18] = np.array([1, 1, 1, np.nan])
            theta[24] = np.array([1, 1, 1, np.nan])
            theta[30] = np.array([1, 1, 1, np.nan])
            theta[36] = np.array([1, 1, 1, np.nan])
            theta[11] = np.array([1, np.nan, 1, 1])
            theta[17] = np.array([1, np.nan, 1, 1])
            theta[23] = np.array([1, np.nan, 1, 1])
            theta[29] = np.array([1, np.nan, 1, 1])
            theta[35] = np.array([1, np.nan, 1, 1])
            theta[41] = np.array([1, np.nan, 1, 1])
            theta[42] = np.array([1, 1, np.nan, np.nan])
            theta[47] = np.array([1, np.nan, np.nan, 1])
            theta[43] = np.array([1, 1, np.nan, 1])
            theta[44] = np.array([1, 1, np.nan, 1])
            theta[45] = np.array([1, 1, np.nan, 1])
            theta[46] = np.array([1, 1, np.nan, 1])

        """
        theta1は(self.state_size, self.eta_size, self.action_size)の3次元配列
        self.theta1[state_index][eta_index][action_index]でθ(s1,a1,η2)を参照できる
        """
        self.theta = theta
    def softmax_probs(self):
        beta = self.beta
        pi = np.zeros((self.state_size, self.action_size))
        pi = np.array(pi, dtype=np.float64)

        max_theta = np.nanmax(self.theta, axis=1, keepdims=True)

        
        exp_theta = np.exp(beta * (self.theta - max_theta))


        #print(f'exp_theta1:{exp_theta}')
        for state_index in range(self.state_size):
            pi[state_index, :] = exp_theta[state_index, :] / np.nansum(exp_theta[state_index, :])
        pi = np.nan_to_num(pi)#nanを0に変換
        return pi



class Agent:
    def __init__(self,grid_shape, action_size=4, gamma=0.9999, lr=0.002, epsilon=1e-8):
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
        self.state_size = grid_shape[0] * grid_shape[1]
        self.grid_shape = grid_shape
        self.memory = []
        self.cost_memory = []
        self.prob_memory = []
        self.pi = Policy(grid_shape=grid_shape)


    def get_action(self, state_index):
        """
        Policy_1に基づいてactionとIetaを取得
        :param state: 現在の状態
        :return: (選択されたアクション, その確率)
        """
        probs = self.pi.softmax_probs()[int(state_index)]#eta_size*action_sizeの二次元配列
        # 確率が1になるように正規化 (念のため)
        probs = probs / np.sum(probs)

        #デバッグ用
        #print("Probs:", probs.data)
        if np.any(np.isnan(probs.data)):
            print(f'probs in pi1:{probs}')
            raise ValueError("NaN detected in probability distribution1!")

        action = np.random.choice(len(probs), p=probs)  # 確率に基づいてアクションを選択(actionは0,...,action_size-1)
        return action



    def add(self, cost, state_index, action, time):
        """
        報酬とアクション確率をメモリに追加
        :param cost: 報酬
        :param prob: アクション確率
        """
        """time=1とtime>1で新しく定義した損失の形が異なるのでそれに伴って新しい損失を定義"""
        data = {}
        data['state_index'] = state_index
        data['action'] = action
        data['cost'] = cost
        self.memory.append(data)
        #print(f'time={time}, old_cost = {cost}, (new_cost, prob) = {data}')
        #self.cost_memory.append(new_cost)
        #self.prob_memory.append(prob)


    def update_pi_SGD(self, learning_rate=None, kappa=None):
        """
        方策Policyのネットワークの更新を行う（動的学習率を適用）
        :param learning_rate: 新しい学習率（指定がなければデフォルトを使用）
        """
        if learning_rate is None:
            learning_rate = self.lr


        new_theta = self.pi.theta
        total_cost = 0
        for data in reversed(self.memory):
            total_cost += data['cost']
        adjustment = 0
        if total_cost > 10:
            adjustment = 0.005
        #勾配項の追加
        for step in range(0,len(self.memory)):
            G = 0
            if step != len(self.memory)-1:
                for data in reversed(self.memory[step:]):
                    cost = data['cost']
                    G = cost + self.gamma * G
                G /= self.gamma**(-1*step)
            elif step == len(self.memory)-1:
                data = self.memory[step]
                cost = data['cost']
                G = cost
            #print(f'step:{step}, G:{G}, ')

            #目的関数の勾配の更新
            for state_index in range(self.state_size):
                for action in range(self.action_size):
                    if not(np.isnan(self.pi.theta[state_index][action])):#theta2がnanじゃないところを更新
                        #定義関数の定義
                        i_s = 0
                        i_a = 0
                        data = self.memory[step]
                        if state_index == data['state_index']:
                            i_s = 1
                        if action == data['action']:
                            i_a = 1
                        grad = (i_s * (i_a + adjustment - self.pi.softmax_probs()[state_index][action])) * G
                        #theta１の更新
                        new_theta[state_index][action] -= learning_rate * grad
        self.pi.theta = new_theta
  
        

    def update_pi_SGD_distributed(self, agent_index, P, z, other_agent1, other_agent2, learning_rate=None):
        """
        方策Policy1のネットワークの更新を行う（動的学習率を適用）
        :param learning_rate: 新しい学習率（指定がなければデフォルトを使用）
        """
        if learning_rate is None:
            learning_rate = self.lr
        new_theta = np.zeros((self.state_size, self.action_size))
        
        total_cost = 0
        for data in reversed(self.memory):
            total_cost += data['cost']
        adjustment = 0
        if total_cost > 10:
            adjustment = 0.005
        #勾配項の追加
        for step in range(0,len(self.memory)):
            G = 0
            if step != len(self.memory)-1:
                for data in reversed(self.memory[step:]):
                    cost = data['cost']
                    G = cost + self.gamma * G
                G /= self.gamma**(-1*step)
            elif step == len(self.memory)-1:
                data = self.memory[step]
                cost = data['cost']
                G = cost
            #print(f'step:{step}, G:{G}, ')

            #目的関数の勾配の更新
            for state_index in range(self.state_size):
                for action in range(self.action_size):
                    if not(np.isnan(self.pi.theta[state_index][action])):#thetaがnanじゃないところを更新
                        #定義関数の定義
                        i_s = 0
                        i_a = 0
                        data = self.memory[step]
                        if state_index == data['state_index']:
                            i_s = 1
                        if action == data['action']:
                            i_a = 1
                        grad = (i_s * (i_a + adjustment - self.pi.softmax_probs()[state_index][action])) * G
                        #theta１の更新
                        new_theta[state_index][action] -= (learning_rate / z[agent_index - 1]) * grad
                        
                        if step == 1:
                            new_theta[state_index][action] += P[agent_index - 1][agent_index - 1] * self.pi.theta[state_index][action] + P[agent_index - 1][other_agent1[0] - 1] * other_agent1[1].pi.theta[state_index][action] + P[agent_index - 1][other_agent2[0] - 1] * other_agent2[1].pi.theta[state_index][action]
                    else:
                        new_theta[state_index][action] = np.nan
        self.pi.theta = new_theta