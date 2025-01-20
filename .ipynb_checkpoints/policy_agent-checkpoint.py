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


class Policy_test1(Model):
    def __init__(self, action_size, eta_size):
        super().__init__()
        # 畳み込み層（Conv）: out_channels と引数順序に注意
        self.conv1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        # 全結合層（FC）
        self.fc1 = L.Linear(136)
        self.fc2 = L.Linear(action_size*eta_size)
        self.scaling = 3

    def forward(self, x):
        x = x.astype(np.float64)  # 必要に応じて64ビットに変更
        # 状態を1次元から4次元テンソルに変換（バッチサイズ, チャンネル, 高さ, 幅）
        x = x.reshape(x.shape[0], 1, 1, -1)  # バッチサイズ, チャンネル=1, 高さ=1, 幅=任意
        if x.ndim == 1:
            x = x[np.newaxis, :]  # xを2次元に変換（バッチサイズを追加）
        elif x.ndim == 0:
            x = x[np.newaxis, np.newaxis]  # スカラーを2次元に変換
        # 畳み込み＋ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))

        # Flattenして全結合層へ
        x = x.reshape(x.shape[0], -1)  # 平坦化
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x - np.max(x.data)  # 最大値を引いてオーバーフローを防止
        x /= np.linalg.norm(x.data) + 1e-8#L2ノルムによるスケーリング
        x = F.softmax(x, axis=-1)  # 最終出力
        return x

class Policy_test2(Model):
    def __init__(self, action_size, eta_size):
        super().__init__()
        # 畳み込み層（Conv）: out_channels と引数順序に注意
        self.conv1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        # 全結合層（FC）
        self.fc1 = L.Linear(136)
        self.fc2 = L.Linear(action_size*eta_size)

        self.scaling = 3

    def forward(self, x):
        x = x.astype(np.float64)  # 必要に応じて64ビットに変更
        # 状態を1次元から4次元テンソルに変換（バッチサイズ, チャンネル, 高さ, 幅）
        x = x.reshape(x.shape[0], 1, 1, -1)  # バッチサイズ, チャンネル=1, 高さ=1, 幅=任意
        if x.ndim == 1:
            x = x[np.newaxis, :]  # xを2次元に変換（バッチサイズを追加）
        elif x.ndim == 0:
            x = x[np.newaxis, np.newaxis]  # スカラーを2次元に変換
        # 畳み込み＋ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))

        # Flattenして全結合層へ
        x = x.reshape(x.shape[0], -1)  # 平坦化
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x - np.max(x.data)  # 最大値を引いてオーバーフローを防止
        x /= np.linalg.norm(x.data) + 1e-8#L2ノルムによるスケーリング
        x = F.softmax(x, axis=-1)  # 最終出力
        return x




class Policy1(Model):
    def __init__(self, action_size, eta_size):
        super().__init__()
        self.l1 = L.Linear(64)
        self.l2 = L.Linear(64)
        self.l3 = L.Linear(128)
        self.l4 = L.Linear(action_size*eta_size)

        self.scaling = 100


    def forward(self, x):#np.ndarray([[np.int64, np.int64]])を渡す
        x = x.astype(np.float64)  # 必要に応じて64ビットに変更
        x = x.reshape(x.shape[0], -1)  # フラット化して 2次元に変換
        #print(f'Before l1:{x}')
        x = F.relu(self.l1(x))
        #print('Before l2:',x.data)
        x = F.relu(self.l2(x))
        #print('After l2:',x.data)
        x = self.l4(x)
        x = x - np.max(x.data)  # 最大値を引いてオーバーフローを防止
        #x /= self.scaling
        x /= np.linalg.norm(x.data) + 1e-8#L2ノルムによるスケーリング(割る値が大きくなるほど、出力が確率的になることに注意)
        x *= 100
        #print("Before softmax:", x.data)  # 確認
        x = F.softmax(x, axis=-1)#axis=-1は数値安定化のために使用
        #print("After softmax:", x.data)  # 確認
        softmax_denominator = np.sum(np.exp(x.data))  # ソフトマックスの分母
        #print("Softmax denominator:", softmax_denominator)
        if softmax_denominator == 0:
            print("Warning: Softmax denominator is zero.")


        return x#xの形はvariable([[x1, x2, x3, ...]])中身の個数はaction_size*eta_size個

class Policy2(Model):
    def __init__(self, action_size, eta_size):
        super().__init__()
        self.l1 = L.Linear(64)
        self.l2 = L.Linear(64)
        self.l3 = L.Linear(128)
        self.l4 = L.Linear(action_size*eta_size)

        self.scaling = 100


    def forward(self, x):#np.ndarray([[np.int64]])を渡す
        x = x.astype(np.float64)  # 必要に応じて64ビットに変更

        x = x.reshape(x.shape[0], -1)
        #print(f'Before l1:{x}')
        x = F.relu(self.l1(x))
        #print(f'Before l2:{x}')
        x = F.relu(self.l2(x))
        #print(f'After l2:{x}')
        x = self.l4(x)
        x = x - np.max(x.data)  # 最大値を引いてオーバーフローを防止
        #x /= self.scaling
        x /= np.linalg.norm(x.data) + 1e-8#L2ノルムによるスケーリング(割る値が大きくなるほど、出力が確率的になることに注意)
        x *= 100
        #print("Before softmax:", x.data)  # 確認
        x = F.softmax(x, axis=-1)#axis=-1は数値安定化のために使用
        #print("After softmax:", x.data)  # 確認
        softmax_denominator = np.sum(np.exp(x.data))  # ソフトマックスの分母
        #print("Softmax denominator:", softmax_denominator)
        if softmax_denominator == 0:
            print("Warning: Softmax denominator is zero.")
        
        return x#xの形はvariable([[x1, x2, x3, x4]])

class Theta1:
    def __init__(self, action_size, eta_size, state_size):
        """

        """
        self.action_size = action_size
        self.eta_size = eta_size
        self.state_size = state_size
        self.beta = 1.0#softmaxの逆温度(これが大きいほど、方策は決定的になりやすい)
        
        theta1 = [1] * self.action_size
        theta1 = theta1 * self.eta_size
        theta1 = theta1 * self.state_size
        self.theta1 = np.array(theta1)

    def softmax_probs(self):
        beta = self.beta
        ,  ,= self.theta1.shape
        pi = np.zeros(())





class Agent:
    def __init__(self, action_size, state_size, kappa, eta_size, Lambda, alpha_risk, grid_shape, eta_space, gamma=0.9999, lr=0.002, alpha=0.001,beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        エージェントの初期化
        :param action_size: アクション空間のサイズ
        :param state_size: 状態空間の次元数
        :param gamma: 割引率
        :param lr: 初期学習率
        :param beta1: Adamの1次モーメント係数
        :param beta2: Adamの2次モーメント係数
        :param epsilon: 数値安定化のための定数
        """
        self.gamma = gamma
        self.lr = lr
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.action_size = action_size
        self.state_size = state_size
        self.eta_size = eta_size
        self.Lambda = Lambda
        self.alpha_risk = alpha_risk
        self.kappa = kappa
        self.grid_shape = grid_shape
        self.eta_space = eta_space
        #self.memory = []
        self.cost_memory = []
        self.prob_memory = []
        self.pi1 = Policy1(action_size=self.action_size, eta_size=self.eta_size)
        self.pi2 = Policy2(action_size=self.action_size, eta_size=self.eta_size)

        #self.optimizer = optimizers.Adam(self.lr)
        #self.optimizer.setup(self.pi)

        """# Adam用のモーメント初期化
        self.m = {param: np.zeros_like(param.data) for param in self.pi.params()}  # 1次モーメント
        self.v = {param: np.zeros_like(param.data) for param in self.pi.params()}  # 2次モーメント
        self.t = 0  # 時間ステップ
        """


    def get_action_1(self, state):
        """
        Policy_1に基づいてactionとIetaを取得
        :param state: 現在の状態
        :return: (選択されたアクション, その確率)
        """
        state = np.array(state, dtype=np.int64)
        state = state[np.newaxis, :]
        #state = np.array([state])[np.newaxis, :]  # 状態を NumPy 配列に変換(stateがnp.int64単体であるときに適用)
        probs = self.pi1(state)  # 方策ネットワークからアクション確率を取得
        probs = probs[0]#probs[0]はvariable([p1, p2, p3, ...])の形(個数はeta_size*action_size)
        #デバッグ用
        #print("Probs:", probs.data)
        if np.any(np.isnan(probs.data)):
            raise ValueError("NaN detected in probability distribution1!")

        action_and_Ieta = np.random.choice(len(probs), p=probs.data)  # 確率に基づいてアクションを選択(action_and_Ietaは0,...,eta_size*action_size-1)
        Ieta = action_and_Ieta // self.action_size#現状Ietaは0,...,eta_size-1の値を取る
        action = action_and_Ieta % self.action_size 
        return action, Ieta, probs[action_and_Ieta]#actionはint, probs[action]はvariable(float)

    def get_action_2(self, state, eta):#stateはnp.int64, etaはint
        """
        Policy_2に基づいてactionとIetaを取得
        :param state: 現在の状態
        :return: (選択されたアクション, その確率)
        """
        #print(f'state:{state}, eta:{eta}')
        #print(type(state))
        #print(type(eta))
        state_and_eta = state + [eta]
        #print('状態とeta')
        #print(state.append(eta))
        #print(state_and_eta)
        state_and_eta = np.array(state_and_eta, dtype=np.int64)
        state_and_eta =  state_and_eta[np.newaxis, :]
        """state_and_Ieta = state * np.int64(Ieta)
        state_and_Ieta = np.array([state_and_Ieta])[np.newaxis, :]  # 状態を NumPy 配列に変換"""
        probs = self.pi2(state_and_eta)  # 方策ネットワークからアクション確率を取得
        probs = probs[0]#probs[0]はvariable([p1, p2, p3, ...])の形(個数はeta_size*action_size)

        #デバッグ用
        #print("Probs:", probs.data)
        if np.any(np.isnan(probs.data)):
            print(f'probs:{probs}')
            raise ValueError("NaN detected in probability distribution2!")

        action_and_Ieta = np.random.choice(len(probs), p=probs.data)  # 確率に基づいてアクションを選択(action_and_etaは0,...,eta_size*action_size-1)
        Ieta = action_and_Ieta // self.action_size#現状etaは0,...,eta_size-1の値を取るのでのちに修正
        action = action_and_Ieta % self.action_size 
        return action, Ieta, probs[action_and_Ieta]#actionはint, probs[action]はvariable(float)


    def add(self, cost, eta, eta_next, prob, time):
        """
        報酬とアクション確率をメモリに追加
        :param cost: 報酬
        :param prob: アクション確率
        """
        """time=1とtime>1で新しく定義した損失の形が異なるのでそれに伴って新しい損失を定義"""
        if time == 1:
            new_cost = cost + self.gamma * self.Lambda * eta_next
            #print(f'gamma:{self.gamma}, Lambda:{self.Lambda}, eta_next:{eta_next}')
        elif time > 1:
            #print(f'gamma:{self.gamma}, Lambda:{self.Lambda}, eta_next:{eta_next}')
            new_cost = (self.Lambda / self.alpha_risk) * max(0, cost-eta) + (1 - self.Lambda) * cost + self.Lambda * self.gamma * eta_next
        data = (new_cost, prob)
        #print(f'time={time}, old_cost = {cost}, (new_cost, prob) = {data}')
        self.cost_memory.append(new_cost)
        self.prob_memory.append(prob)


    def update_pi1_SGD(self, learning_rate=None, kappa=None):
        """
        方策Policy1のネットワークの更新を行う（動的学習率を適用）
        :param learning_rate: 新しい学習率（指定がなければデフォルトを使用）
        """
        if learning_rate is None:
            learning_rate = self.lr

        if kappa is None:
            kappa = self.kappa

        self.pi1.cleargrads()
        G, loss = 0, 0
        #探索項の追加
        for state_index in range(self.state_size):
            zelo_list = [0] * self.state_size
            zelo_list[state_index] = 1
            state = zelo_list
            state = np.array(state, dtype=np.int64)
            state = state[np.newaxis, :]
            probs = self.pi1(state)[0]
            for prob in probs:
                #if prob.data <=0:
                    #print('update_pi1_SGD内でprob=0がF.log()に代入されました。')#デバッグ用（この警告が出たらpi1のsoftmax関数に渡す値をスケーリングし直す。）
                    #raise ValueError(f"Invalid probability value detected: prob={prob.data}")
                loss -= F.log(prob + 1e-250)
        loss *= kappa / (self.state_size * self.action_size * self.eta_size)
        #print(f'係数:{kappa / (self.state_size * self.action_size * self.eta_size * self.eta_size)}')
        #print(f'探索項 in update_pi1:{loss}')
        #print('new_cost')
        for new_cost in reversed(self.cost_memory):
            #print(f' {new_cost},', end='')
            G = new_cost + self.gamma * G
            #print(f'updated G:{G}')
        prob1 = self.prob_memory[0]
        #print(f'prob1:{prob1}')
        #print(f'G:{G}')
        loss += F.log(prob1) * G
        #print(f'loss in update_pi1_SGD: {loss}')
        #print(f'目的関数 in update_pi1:{loss}')
        loss.backward()#勾配の計算
        # 手動SGD: 各パラメータを更新
        for param in self.pi1.params():
            if param.grad is not None:
                param.data -= learning_rate * param.grad.data  # パラメータ更新

        

    def update_pi2_SGD(self, learning_rate=None, kappa=None):
        """
        方策Policy1のネットワークの更新を行う（動的学習率を適用）
        :param learning_rate: 新しい学習率（指定がなければデフォルトを使用）
        """
        if learning_rate is None:
            learning_rate = self.lr

        if kappa is None:
            kappa = self.kappa

        self.pi2.cleargrads()

        loss = 0


        #探索項の追加
        for state_index in range(self.state_size):
            zelo_list = [0] * self.state_size
            zelo_list[state_index] = 1
            state = zelo_list
            for Ieta in range(self.eta_size):
                state_and_eta = state + [self.eta_space[Ieta]]
                state_and_eta = np.array(state_and_eta, dtype=np.int64)
                state_and_eta = state_and_eta[np.newaxis, :]
                probs = self.pi2(state_and_eta)  # 方策ネットワークからアクション確率を取得
                probs = probs[0]#probs[0]はvariable([p1, p2, p3, ...])の形(個数はeta_size*action_size)
                #print(f'probs:{probs}')
                for prob in probs:
                    #if prob.data <=0:
                        #print('update_pi2_SGD内でprob=0がF.log()に代入されました。')#デバッグ用（この警告が出たらpi2のsoftmax関数に渡す値をスケーリングし直す。）
                        #raise ValueError(f"Invalid probability value detected: prob={prob.data}")
                    loss -= F.log(prob + 1e-250)
        loss *= kappa / (self.state_size * self.action_size * self.eta_size * self.eta_size)
        #print(f'係数:{kappa / (self.state_size * self.action_size * self.eta_size * self.eta_size)}')
        #print(f'探索項 in update_pi2:{loss}')
        #勾配項の追加
        for step in range(1,len(self.cost_memory)):
            G = 0
            if step != len(self.cost_memory)-1:
                for new_cost in reversed(self.cost_memory[step:]):
                    G = new_cost + self.gamma * G
                G /= self.gamma**(-1*step)
            elif step == len(self.cost_memory)-1:
                G = self.cost_memory[step]
            #print(f'step:{step}, G:{G}, ')
            loss += G * F.log(self.prob_memory[step])
        
        #print(f'loss in update_pi2_SGD: {loss}')




        loss.backward()#勾配の計算
        # 手動SGD: 各パラメータを更新
        for param in self.pi2.params():
            if param.grad is not None:
                param.data -= learning_rate * param.grad.data  # パラメータ更新
        


    """def update_Adam(self, learning_rate=None):
        
        #方策ネットワークの更新を行う（Adamを手動で実装）
        #:param learning_rate: 新しい学習率（指定がなければデフォルトを使用）
        
        if learning_rate is None:
            learning_rate = self.lr

        self.pi.cleargrads()

        # 割引報酬 G と損失の計算
        G, loss = 0, 0
        for cost, prob in reversed(self.memory):
            G = cost + self.gamma * G

        for cost, prob in self.memory:
            loss += F.log(prob) * G

        loss.backward()  # 勾配の計算

        # 時間ステップの更新
        self.t += 1

        # パラメータ更新 (Adam)
        for param in self.pi.params():
            if param.grad is not None:
                grad = param.grad.data

                # 1次モーメントと2次モーメントの更新
                self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
                self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)

                # バイアス補正
                m_hat = self.m[param] / (1 - self.beta1 ** self.t)
                v_hat = self.v[param] / (1 - self.beta2 ** self.t)

                # パラメータの更新
                param.data -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # メモリをクリア
        self.memory = []
        """
