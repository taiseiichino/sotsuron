
#他ファイルからのimport
from cmath import cos
from gym.envs.registration import register#環境の登録
from stochastic_cliff_env import StochasticCliffWalkingEnv  # カスタム環境をインポート
from custom_maps import get_custom_map#カスタムマップをインポート
from policy_agent import Policy, Policy, Agent
import math
import gym#カスタム環境をロードするために必要
import numpy as np
import os
from datetime import datetime
import pandas as pd
#プロット
import matplotlib.pyplot as plt
import logging








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




episodes = 3
agent = Agent(grid_shape=[env.n_rows, env.n_cols])#action_size,state_sizeはint型で渡している
cost_history = []
cost_list = []

for episode in range(episodes):
    t = 1
    state_index, info = env.reset()#stateはnumpy.int64型
    #print(f'first state : {state_index}')
    #print(f't : {t}')
    #env.render()
    start_position = state_index
    done = False#done == Trueの時episode終了とする
    total_cost = 0
    while not done:
        #print(f'get_action_1前のstate:{state}')#デバッグ用
        action = agent.get_action(state_index)#a_1,Ieta_2を取得(actionはint, probはvariable(float))
        next_state, cost, terminated, truncated, info_nan = env.step(action)#s_t+1,c_tを取得(next_stateはnp.int64)
        agent.add(cost=cost, state_index=state_index, action=action, time=t)
        state_index = next_state
        total_cost += cost
        done = terminated or truncated
        
        #env.render()


        t += 1
        #print('episode:' + str(episode) + "done")
    step_size = 0.001


    #print('update pi')
    agent.update_pi_SGD(learning_rate=step_size)
    
    #メモリーをクリア
    agent.memory = []
    data = {}
    cost_history.append(total_cost)
    data['cost'] = cost
    data['episode'] = episode
    cost_list.append(data)
    if episode % 10 == 0:
        print("episode :{}, total cost : {:.1f}, start_position : {}, step_size : {}, time : {}".format(episode, total_cost, start_position, step_size, t))
    if episode % 100 == 0:
        for s in range(agent.state_size):
            print(f'state : {s}\npi : {agent.pi.softmax_probs()[s]}')


# DataFrame
df = pd.DataFrame(cost_list)

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

# DataFrame を保存するファイルパスを指定
csv_file_path = os.path.join(output_folder, "cost_history.csv")
df.to_csv(csv_file_path, index=False)

# プロット
plt.plot(cost_history)
plt.xlabel('Episode')
plt.ylabel('Total Cost')
plt.title('Cost History')

# 保存するファイルパス（フォルダ内に "設定変数.txt" を作成）
file_path = os.path.join(output_folder, "設定変数.txt")

# データを書き込む
with open(file_path, "w") as file:
    file.write(f'episodes:{episodes}\ngamma:{agent.gamma}\nmax_step:{200}')  # 数値データを記載

# プロット画像をサブフォルダ内に保存
output_path = os.path.join(output_folder, 'cost_history.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 解像度を指定して保存
plt.show()


pi_file_path = os.path.join(output_folder, "pi.txt")

# pi1.txt の内容
with open(pi_file_path, "w") as file:
    for s in range(agent.state_size):
            file.write(f'state : {s}\npi : {agent.pi.softmax_probs()[s]}\n')

print(f"データフレームを保存しました: {csv_file_path}")
print(f"設定変数を保存しました: {file_path}")
print(f"プロット画像を保存しました: {output_path}")
