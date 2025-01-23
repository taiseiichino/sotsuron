
#他ファイルからのimport
from cmath import cos
from gym.envs.registration import register#環境の登録
from stochastic_cliff_env import StochasticCliffWalkingEnv  # カスタム環境をインポート
from custom_maps import get_custom_map, get_custom_map_comp#カスタムマップをインポート
from policy_agent import Policy, Agent
import math
import gym#カスタム環境をロードするために必要
import numpy as np
import os
from datetime import datetime
import pandas as pd
#プロット
import matplotlib.pyplot as plt

    


def distributed_RARL(episodes=30001, step_size=0.0003):

    # カスタムマップの定義
    custom_grid = get_custom_map_comp()

    # 登録済みのカスタム環境をロード
    env = StochasticCliffWalkingEnv(grid=custom_grid, slip_prob=0.0)

    agent1 = Agent(grid_shape=[env.n_rows, env.n_cols])#action_size,state_sizeはint型で渡している
    agent2 = Agent(grid_shape=[env.n_rows, env.n_cols])
    agent3 = Agent(grid_shape=[env.n_rows, env.n_cols])

    a1_cost_history = []
    a1_cost_list = []
    a2_cost_history = []
    a2_cost_list = []
    a3_cost_history = []
    a3_cost_list = []

    P = [[1/3, 1/3, 1/3], [1/4, 3/8, 3/8], [1/4, 3/8, 3/8]]
    z1 = [1, 0, 0]
    z2 = [0, 1, 0]
    z3 = [0, 0, 1]

    for episode in range(episodes):

        """agent1の軌跡の生成"""
        a1_t = 1
        a1_state_index, a1_info = env.reset()
        #print(f'first state : {state_index}')
        #print(f't : {t}')
        #env.render()
        a1_start_position = a1_state_index
        a1_done = False#a1_done == Trueの時episode終了とする
        a1_total_cost = 0
        while not a1_done:
            #print(f'get_action_1前のstate:{state}')#デバッグ用
            a1_action = agent1.get_action(a1_state_index)
            a1_next_state, a1_cost, a1_terminated, a1_truncated, a1_info_nan = env.step(a1_action,1)#s_t+1,c_tを取得(next_stateはnp.int64)
            #print(f't : {t}')
            #print(f'とったaction : {action}')
            #print(f'現在位置state : {next_state}')
            agent1.add(cost=a1_cost, state_index=a1_state_index, action=a1_action, time=a1_t)
            a1_state_index = a1_next_state
            a1_total_cost += a1_cost
            a1_done = a1_terminated or a1_truncated
            
            #env.render()
            a1_t += 1
            #print('episode:' + str(episode) + "done")

        """agent2の奇跡の生成"""
        a2_t = 1
        if custom_grid.shape[0]*custom_grid.shape[1] == 16:
            a2_state_index, a2_info = env.reset(start_pos=[2, 2])
        elif custom_grid.shape[0]*custom_grid.shape[1] == 48:
             a2_state_index, a2_info = env.reset(start_pos=[2, 1])
        #print(f'first state : {state_index}')
        #print(f't : {t}')
        #env.render()
        a2_start_position = a2_state_index
        a2_done = False#a2_done == Trueの時episode終了とする
        a2_total_cost = 0
        while not a2_done:
            #print(f'get_action_1前のstate:{state}')#デバッグ用
            a2_action = agent2.get_action(a2_state_index)#a_1,Ieta_2を取得(actionはint, probはvariable(float))
            a2_next_state, a2_cost, a2_terminated, a2_truncated, a2_info_nan = env.step(a2_action, 2)#s_t+1,c_tを取得(next_stateはnp.int64)
            #print(f't : {t}')
            #print(f'とったaction : {action}')
            #print(f'現在位置state : {next_state}')
            agent2.add(cost=a2_cost, state_index=a2_state_index, action=a2_action, time=a2_t)
            a2_state_index = a2_next_state
            a2_total_cost += a2_cost
            a2_done = a2_terminated or a2_truncated
            
            #env.render()


            a2_t += 1
            #print('episode:' + str(episode) + "done")

        """agent3の軌跡の生成"""
        a3_t = 1
        if custom_grid.shape[0]*custom_grid.shape[1] == 16:
            a3_state_index, a3_info = env.reset(start_pos=[1, 2])
        elif custom_grid.shape[0]*custom_grid.shape[1] == 48:
             a3_state_index, a3_info = env.reset(start_pos=[3, 4])
        #print(f'first state : {state_index}')
        #print(f't : {t}')
        #env.render()
        a3_start_position = a3_state_index
        a3_done = False#a3_done == Trueの時episode終了とする
        a3_total_cost = 0
        while not a3_done:
            #print(f'get_action_1前のstate:{state}')#デバッグ用
            a3_action = agent3.get_action(a3_state_index)#a_1,Ieta_2を取得(actionはint, probはvariable(float))
            a3_next_state, a3_cost, a3_terminated, a3_truncated, a3_info_nan = env.step(a3_action,3)#s_t+1,c_tを取得(next_stateはnp.int64)
            #print(f't : {t}')
            #print(f'とったaction : {action}')
            #print(f'現在位置state : {next_state}')
            agent3.add(cost=a3_cost, state_index=a3_state_index, action=a3_action, time=a3_t)
            a3_state_index = a3_next_state
            a3_total_cost += a3_cost
            a3_done = a3_terminated or a3_truncated
            #env.render()
            a3_t += 1
            #print('episode:' + str(episode) + "done")
        #step_sizeを変動型として計算
        #step_size = 0.02236/math.sqrt(episode+0.2

        """各エージェントの推定値の更新"""
        #print('update pi1')
        agent1.update_pi_SGD_distributed(agent_index=1, P=P, z=z1, other_agent1=(2, agent2), other_agent2=(3, agent3) ,learning_rate=step_size)

        #print('update pi1')
        agent2.update_pi_SGD_distributed(agent_index=2, P=P, z=z2, other_agent1=(1, agent1), other_agent2=(3, agent3) ,learning_rate=step_size)
        
        #print('update pi1')
        agent3.update_pi_SGD_distributed(agent_index=3, P=P, z=z3, other_agent1=(1, agent1), other_agent2=(2, agent2) ,learning_rate=step_size)

        new_z1 = z1
        new_z2 = z2
        new_z3 = z3
        """重み行列Pの固有値1の左固有ベクトルの推定値zの更新"""
        for i in range(3):
            new_z1[i] = P[0][0] * z1[i] + P[0][1] * z2[i] + P[0][2] * z3[i]
            new_z2[i] = P[1][0] * z1[i] + P[1][1] * z2[i] + P[1][2] * z3[i]
            new_z3[i] = P[2][0] * z1[i] + P[2][1] * z2[i] + P[2][2] * z3[i]
        z1 = new_z1
        z2 = new_z2
        z3 = new_z3
        #メモリーをクリア
        agent1.memory = []
        agent2.memory = []
        agent3.memory = []

        a1_data = {}
        a2_data = {}
        a3_data = {}

        a1_cost_history.append(a1_total_cost)
        a1_data['cost'] = a1_total_cost
        a1_data['episode'] = episode
        a1_cost_list.append(a1_data)

        a2_cost_history.append(a2_total_cost)
        a2_data['cost'] = a2_total_cost
        a2_data['episode'] = episode
        a2_cost_list.append(a2_data)

        a3_cost_history.append(a3_total_cost)
        a3_data['cost'] = a3_total_cost
        a3_data['episode'] = episode
        a3_cost_list.append(a3_data)

        if episode % 10 == 0:
            print("agent1 : episode :{}, total cost : {:.1f}, start_position : {}, step_size : {}, time : {}".format(episode, a1_total_cost, a1_start_position, step_size, a1_t))
            print("agent2 : episode :{}, total cost : {:.1f}, start_position : {}, step_size : {}, time : {}".format(episode, a2_total_cost, a2_start_position, step_size, a2_t))
            print("agent3 : episode :{}, total cost : {:.1f}, start_position : {}, step_size : {}, time : {}".format(episode, a3_total_cost, a3_start_position, step_size, a3_t))
        if episode % 100 == 0:
            print('agent1')
            for s in range(agent1.state_size):
                print(f'state : {s}\npi : {agent1.pi.softmax_probs()[s]}')

            print('\nagent2')
            for s in range(agent2.state_size):
                print(f'state : {s}\npi : {agent2.pi.softmax_probs()[s]}')
            
            print('\nagent3')
            for s in range(agent3.state_size):
                print(f'state : {s}\npi : {agent3.pi.softmax_probs()[s]}')


    # DataFrame
    df1 = pd.DataFrame(a1_cost_list)
    # DataFrame
    df2 = pd.DataFrame(a2_cost_list)
    # DataFrame
    df3 = pd.DataFrame(a3_cost_list)
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
    a1_csv_file_path = os.path.join(output_folder, "a1_cost_history.csv")
    df1.to_csv(a1_csv_file_path, index=False)

    # DataFrame を保存するファイルパスを指定
    a2_csv_file_path = os.path.join(output_folder, "a2_cost_history.csv")
    df2.to_csv(a2_csv_file_path, index=False)

    # DataFrame を保存するファイルパスを指定
    a3_csv_file_path = os.path.join(output_folder, "a3_cost_history.csv")
    df3.to_csv(a3_csv_file_path, index=False)

    # プロット
    plt.plot(a1_cost_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Cost')
    plt.title('Cost History')
    # プロット画像をサブフォルダ内に保存
    output_path = os.path.join(output_folder, 'a1_cost_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 解像度を指定して保存
    #plt.show()

    # プロット
    plt.plot(a2_cost_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Cost')
    plt.title('Cost History')
    # プロット画像をサブフォルダ内に保存
    output_path = os.path.join(output_folder, 'a2_cost_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 解像度を指定して保存
    #plt.show()

    # プロット
    plt.plot(a3_cost_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Cost')
    plt.title('Cost History')
    # プロット画像をサブフォルダ内に保存
    output_path = os.path.join(output_folder, 'a3_cost_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 解像度を指定して保存
    #plt.show()

    # 保存するファイルパス（フォルダ内に "設定変数.txt" を作成）
    file_path = os.path.join(output_folder, "設定変数.txt")

    # データを書き込む
    with open(file_path, "w") as file:
        file.write(f'cliffcost : 5\nslip_prob : {slip_p}\nepisodes : {episodes}\ngamma : {agent1.gamma}\nmax_step : {200}\nstep_size : {step_size}')  # 数値データを記載




    a1_pi_file_path = os.path.join(output_folder, "a1_pi.txt")
    with open(a1_pi_file_path, "w") as file:
        for s in range(agent1.state_size):
                file.write(f'state : {s}\npi1 : {agent1.pi.softmax_probs()[s]}\n')

    a2_pi_file_path = os.path.join(output_folder, "a2_pi.txt")
    with open(a2_pi_file_path, "w") as file:
        for s in range(agent2.state_size):
                file.write(f'state : {s}\npi1 : {agent2.pi.softmax_probs()[s]}\n')


    a3_pi_file_path = os.path.join(output_folder, "a3_pi.txt")
    with open(a3_pi_file_path, "w") as file:
        for s in range(agent3.state_size):
                file.write(f'state : {s}\npi1 : {agent3.pi.softmax_probs()[s]}\n')

