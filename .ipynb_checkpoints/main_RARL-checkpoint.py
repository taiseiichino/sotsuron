
#他ファイルからのimport
from gym.envs.registration import register#環境の登録
from stochastic_cliff_env import StochasticCliffWalkingEnv  # カスタム環境をインポート
from custom_maps import get_custom_map, get_custom_map_test#カスタムマップをインポート
from policy_agent import Policy1, Policy2, Agent
import math
import gym#カスタム環境をロードするために必要
import numpy as np
import os
from datetime import datetime
#import dezero.functions as F
#import dezero.layers as L
#from dezero import Model
#from dezero import optimizers






# カスタムマップの定義
custom_grid = get_custom_map()

# 環境を登録
register(
    id='StochasticCliffWalking-v0',  # 環境ID
    entry_point='stochastic_cliff_env:StochasticCliffWalkingEnv',  # 環境クラスへのパス
    kwargs={'grid': custom_grid, 'slip_prob': 0.5},  # 環境に渡すパラメータ
)

# 登録済みのカスタム環境をロード
env = gym.make('StochasticCliffWalking-v0')

# 環境をテスト
# 環境を初期化
#obs, info = env.reset()

# 結果を表示（テスト用）
#print("観測:", obs)#obs(状態を表すインスタンス)はnp.int64
#print("情報:", info)
#env.render()




episodes = 20000
agent = Agent(kappa=0.5, eta_size=env.eta_space.n, Lambda=0.5, alpha_risk=0.05, grid_shape=[env.n_rows, env.n_cols], eta_space=env.cost_space)#action_size,state_sizeはint型で渡している
cost_history = []

"""print('初期値')
for state_row in range(agent.grid_shape[0]):
    for state_col in range(agent.grid_shape[1]):
        for Ieta in range(agent.eta_size):
            state_and_eta = np.array([state_row, state_col, agent.eta_space[Ieta]], dtype=np.int64)
            state_and_eta = state_and_eta[np.newaxis, :]
            probs = agent.pi2(state_and_eta)  # 方策ネットワークからアクション確率を取得
            probs = probs[0]#probs[0]はvariable([p1, p2, p3, ...])の形(個数はeta_size*action_size)
            print(f'probs:{probs}')
"""
for episode in range(episodes):
    state_index, info = env.reset()#stateはnumpy.int64型
    start_position = state_index
    done = False#done == Trueの時episode終了とする
    total_cost = 0
    t = 1
    while not done:
        if t == 1:
            #print(f'get_action_1前のstate:{state}')#デバッグ用
            action, nexteta_index= agent.get_action_1(state_index)#a_1,Ieta_2を取得(actionはint, probはvariable(float))
            eta_index = 0
        elif t > 1:
            #print(f'get_action_2前のstate:{state}')#デバッグ用
            action, nexteta_index= agent.get_action_2(state_index, eta_index)#a_t,Ieta_t+1を取得(actionはint, probはvariable(float))
        next_state, cost, terminated, truncated, info_nan = env.step(action)#s_t+1,c_tを取得(next_stateはnp.int64)

        agent.add(cost=cost, eta_index=eta_index, nexteta_index=nexteta_index, state_index=state_index, action=action, time=t)
        state = next_state
        eta_index = nexteta_index
        total_cost += cost
        done = terminated or truncated


        t += 1
        #print('episode:' + str(episode) + "done")
    #step_sizeを変動型として計算
    #step_size = 0.02236/math.sqrt(episode+0.2)
    step_size = 0.02
    kappa = 0/math.sqrt(episode+0.2)


    #print('update pi1')
    agent.update_pi1_SGD(learning_rate=step_size, kappa=kappa)
    #print('update pi2')
    agent.update_pi2_SGD(learning_rate=step_size, kappa=kappa)
    #メモリーをクリア
    agent.memory = []

    cost_history.append(total_cost)
    if episode % 1 == 0:
        print("episode :{}, total cost : {:.1f}, start_position : {}, step_size : {}, kappa : {}, time : {}".format(episode, total_cost, start_position, step_size, kappa, t))
    if episode % 100 == 0:
        for s in range(agent.state_size):
            print(f'state : {s}\npi1 : {agent.pi1.softmax_probs()[s]}')
        for s in range(agent.state_size):
            print(f'state : {s}\npi2 : {agent.pi2.softmax_probs()[s]}')
        

# ベースフォルダ
base_folder = "数値データ"
# 現在の日付時刻を取得してフォルダ名を作成
current_time = datetime.now().strftime("%Y年%m月%d日%H時%M分")
output_folder_name = current_time
# ベースフォルダの中に作成するサブフォルダのパスを構築
output_folder = os.path.join(base_folder, output_folder_name)

# フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#プロット
import matplotlib.pyplot as plt

plt.plot(cost_history)
plt.xlabel('Episode')
plt.ylabel('Total Cost')
plt.title('Cost History')



# 保存するファイルパス（フォルダ内に "data.txt" を作成）
file_path = os.path.join(base_folder, "設定変数.txt")

# データを書き込む
with open(file_path, "w") as file:
    file.write(f'episodes:{episodes}\nkappa:{agent.kappa}\ngamma:{agent.gamma}\nlambda:{agent.Lambda}\nalpha_risk:{agent.alpha_risk}\n')  # 数値データを記載

# サブフォルダ内に保存
output_path = os.path.join(output_folder, 'cost_history.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 解像度を指定して保存
plt.show()


