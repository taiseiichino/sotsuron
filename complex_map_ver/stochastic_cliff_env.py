#オリジナルのcliffwalk environmentのクラスを定義する

import gym
from gym import spaces
import numpy as np
import random

class StochasticCliffWalkingEnv(gym.Env):
    def __init__(self, grid, slip_prob=0.0):
        super().__init__()
        self.grid = grid
        self.n_rows, self.n_cols = grid.shape
        self.start = np.argwhere(grid == 'S')[0]#スタート位置の座標をNumpyの配列にしてself.startに代入
        self.goal = np.argwhere(grid == 'G')[0]#ゴール位置も同様
        self.current_pos = self.start
        self.slip_prob = slip_prob  # 崖に滑る確率
        self.max_steps = 200  # 最大ステップ数を設定
        self.elapsed_steps = 0  # 経過ステップ数を初期化

        # 観測空間とアクション空間を定義
        self.observation_space = spaces.Discrete(self.n_rows * self.n_cols)#spaces.Discreteクラスで状態を[0,1,2,3]のように管理する。.nで状態総数を参照できる
        self.action_space = spaces.Discrete(4)  # 上, 右, 下, 左
        self.cost_space = [1, 5]#通常マスは1,崖マスは100
        self.eta_space = spaces.Discrete(len(self.cost_space)) #コストの種類の数と同じ。

    def reset(self, seed=None, options=None, start_pos=None):#start_posはgridのindexをリストで渡す
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        #start_posが何も受け取っていない場合は'S'を初期位置とする
        if not start_pos:
            self.current_pos = self.start
        else:
            self.current_pos = np.array(start_pos, dtype=np.int64)
        """
        初期位置をランダムにしたい場合は以下を使う
        """
        """start_pos = [np.random.choice(range(self.n_rows)), np.random.choice(range(self.n_cols))]
        self.current_pos = np.array(start_pos, dtype=np.int64)
        #print(f'start_pos:{self.current_pos}')
        #print(f'type(self.start):{type(self.start)}')
        self.elapsed_steps = 0#ステップカウントをリセット
        info = {}
        return self._get_state(), info"""
        """
        初期位置を決定的にしたい場合は以下を使う
        """
        self.elapsed_steps = 0#ステップカウントをリセット
        info = {}
        return self._get_state(), info

    def constant_reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.current_pos = self.start
        self.elapsed_steps = 0#ステップカウントをリセット
        info = {}
        return self._get_state(), info

    def step(self, action):
        moves = {
            0: (-1, 0),  # 上
            1: (0, 1),   # 右
            2: (1, 0),   # 下
            3: (0, -1),  # 左
        }

        intended_move = np.array(moves[action])
        new_pos = self.current_pos + intended_move

        # **確率的な移動の追加**
        # 現在地が崖の上のセルの場合、`slip_prob`の確率で崖に滑る
        if self.is_above_cliff():
            if random.uniform(0, 1) < self.slip_prob:
                new_pos = self.current_pos + np.array([1, 0])  # 崖（下のセル）に移動
        cost = 0
        # 境界チェック
        if (
            0 <= new_pos[0] < self.n_rows and
            0 <= new_pos[1] < self.n_cols
        ):
            self.current_pos = new_pos
        else:
            cost += 1

        # 報酬と終了条件
        cell = self.grid[tuple(self.current_pos)]
        if cell == 'C':  # 崖
            cost += self.cost_space[1]
            terminated = False#あとでTrueに変える
            self.current_pos = self.start

        elif np.array_equal(self.current_pos, self.goal):  # ゴール
            cost += 0
            terminated = True#ゴールして終了したい場合はTrue
            #terminated = False
            #self.current_pos = self.start

        else:
            cost += self.cost_space[0]  # 通常マス
            terminated = False

        self.elapsed_steps += 1
        truncated = self.elapsed_steps >=self.max_steps
        info = {}

        return self._get_state(), cost, terminated, truncated, info

    def render(self):
        env = np.copy(self.grid)
        env[tuple(self.current_pos)] = 'A'
        print("\n".join([" ".join(row) for row in env]))
        print()

    def _get_state(self):
        return self.current_pos[0] * self.n_cols + self.current_pos[1]#np.int64型

    def is_above_cliff(self):
        """
        現在地が崖'C'の上のセルかを判定
        """
        below_pos = self.current_pos + np.array([1, 0])
        if (
            0 <= below_pos[0] < self.n_rows and
            0 <= below_pos[1] < self.n_cols
        ):
            return self.grid[tuple(below_pos)] == 'C'
        return False
