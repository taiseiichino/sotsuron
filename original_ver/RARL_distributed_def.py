
#他ファイルからのimport
from cmath import cos
from gym.envs.registration import register#環境の登録
from stochastic_cliff_env import StochasticCliffWalkingEnv  # カスタム環境をインポート
from custom_maps import get_custom_map#カスタムマップをインポート
from policy_agent import Policy1, Policy2, Agent
import math
import gym#カスタム環境をロードするために必要
import numpy as np
import os
from datetime import datetime
import pandas as pd
#プロット
import matplotlib.pyplot as plt

    


def distributed_RARL(Lambda, episodes=30001, step_size=0.0003):

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




    agent1 = Agent(kappa=0.5, eta_size=env.eta_space.n, Lambda=Lambda, alpha_risk=0.05, grid_shape=[env.n_rows, env.n_cols], eta_space=env.cost_space)#action_size,state_sizeはint型で渡している
    agent2 = Agent(kappa=0.5, eta_size=env.eta_space.n, Lambda=Lambda, alpha_risk=0.05, grid_shape=[env.n_rows, env.n_cols], eta_space=env.cost_space)
    agent3 = Agent(kappa=0.5, eta_size=env.eta_space.n, Lambda=Lambda, alpha_risk=0.05, grid_shape=[env.n_rows, env.n_cols], eta_space=env.cost_space)

    a1_cost_history = []
    a1_cost_list = []
    a2_cost_history = []
    a2_cost_list = []
    a3_cost_history = []
    a3_cost_list = []

    P = [[1/3, 1/3, 1/3], [1/4, 3/8, 3/8], [1/4, 3/8, 3/8]]
    z1 = [1/3, 1/3, 1/3]
    z2 = [1/3, 1/3, 1/3]
    z3 = [1/3, 1/3, 1/3]

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
            if a1_t == 1:
                #print(f'get_action_1前のstate:{state}')#デバッグ用
                a1_action, a1_nexteta_index= agent1.get_action_1(a1_state_index)#a_1,Ieta_2を取得(actionはint, probはvariable(float))
                a1_eta_index = 0
            elif a1_t > 1:
                #print(f'get_action_2前のstate:{state}')#デバッグ用
                a1_action, a1_nexteta_index= agent1.get_action_2(a1_state_index, a1_eta_index)#a_t,Ieta_t+1を取得(actionはint, probはvariable(float))
            a1_next_state, a1_cost, a1_terminated, a1_truncated, a1_info_nan = env.step(a1_action)#s_t+1,c_tを取得(next_stateはnp.int64)
            #print(f't : {t}')
            #print(f'とったaction : {action}')
            #print(f'現在位置state : {next_state}')
            agent1.add(cost=a1_cost, eta_index=a1_eta_index, nexteta_index=a1_nexteta_index, state_index=a1_state_index, action=a1_action, time=a1_t)
            a1_state_index = a1_next_state
            a1_eta_index = a1_nexteta_index
            a1_total_cost += a1_cost
            a1_done = a1_terminated or a1_truncated
            
            #env.render()


            a1_t += 1
            #print('episode:' + str(episode) + "done")

        """agent2の奇跡の生成"""
        a2_t = 1
        a2_state_index, a2_info = env.reset(start_pos=[2, 2])
        #print(f'first state : {state_index}')
        #print(f't : {t}')
        #env.render()
        a2_start_position = a2_state_index
        a2_done = False#a2_done == Trueの時episode終了とする
        a2_total_cost = 0
        while not a2_done:
            if a2_t == 1:
                #print(f'get_action_1前のstate:{state}')#デバッグ用
                a2_action, a2_nexteta_index= agent2.get_action_1(a2_state_index)#a_1,Ieta_2を取得(actionはint, probはvariable(float))
                a2_eta_index = 0
            elif a2_t > 1:
                #print(f'get_action_2前のstate:{state}')#デバッグ用
                a2_action, a2_nexteta_index= agent2.get_action_2(a2_state_index, a2_eta_index)#a_t,Ieta_t+1を取得(actionはint, probはvariable(float))
            a2_next_state, a2_cost, a2_terminated, a2_truncated, a2_info_nan = env.step(a2_action)#s_t+1,c_tを取得(next_stateはnp.int64)
            #print(f't : {t}')
            #print(f'とったaction : {action}')
            #print(f'現在位置state : {next_state}')
            agent2.add(cost=a2_cost, eta_index=a2_eta_index, nexteta_index=a2_nexteta_index, state_index=a2_state_index, action=a2_action, time=a2_t)
            a2_state_index = a2_next_state
            a2_eta_index = a2_nexteta_index
            a2_total_cost += a2_cost
            a2_done = a2_terminated or a2_truncated
            
            #env.render()


            a2_t += 1
            #print('episode:' + str(episode) + "done")

        """agent3の軌跡の生成"""
        a3_t = 1
        a3_state_index, a3_info = env.reset(start_pos=[1, 2])
        #print(f'first state : {state_index}')
        #print(f't : {t}')
        #env.render()
        a3_start_position = a3_state_index
        a3_done = False#a3_done == Trueの時episode終了とする
        a3_total_cost = 0
        while not a3_done:
            if a3_t == 1:
                #print(f'get_action_1前のstate:{state}')#デバッグ用
                a3_action, a3_nexteta_index= agent3.get_action_1(a3_state_index)#a_1,Ieta_2を取得(actionはint, probはvariable(float))
                a3_eta_index = 0
            elif a3_t > 1:
                #print(f'get_action_2前のstate:{state}')#デバッグ用
                a3_action, a3_nexteta_index= agent3.get_action_2(a3_state_index, a3_eta_index)#a_t,Ieta_t+1を取得(actionはint, probはvariable(float))
            a3_next_state, a3_cost, a3_terminated, a3_truncated, a3_info_nan = env.step(a3_action)#s_t+1,c_tを取得(next_stateはnp.int64)
            #print(f't : {t}')
            #print(f'とったaction : {action}')
            #print(f'現在位置state : {next_state}')
            agent3.add(cost=a3_cost, eta_index=a3_eta_index, nexteta_index=a3_nexteta_index, state_index=a3_state_index, action=a3_action, time=a3_t)
            a3_state_index = a3_next_state
            a3_eta_index = a3_nexteta_index
            a3_total_cost += a3_cost
            a3_done = a3_terminated or a3_truncated
            
            #env.render()
        


            a3_t += 1
            #print('episode:' + str(episode) + "done")
        #step_sizeを変動型として計算
        #step_size = 0.02236/math.sqrt(episode+0.2
        kappa = 0/math.sqrt(episode+0.2)

        """各エージェントの推定値の更新"""
        #print('update pi1')
        agent1.update_pi1_SGD_distributed(agent_index=1, P=P, z=z1, other_agent1=(2, agent2), other_agent2=(3, agent3) ,learning_rate=step_size, kappa=kappa)
        #print('update pi2')
        agent1.update_pi2_SGD_distributed(agent_index=1, P=P, z=z1, other_agent1=(2, agent2), other_agent2=(3, agent3), learning_rate=step_size, kappa=kappa)

        #print('update pi1')
        agent2.update_pi1_SGD_distributed(agent_index=2, P=P, z=z2, other_agent1=(1, agent1), other_agent2=(3, agent3) ,learning_rate=step_size, kappa=kappa)
        #print('update pi2')
        agent2.update_pi2_SGD_distributed(agent_index=2, P=P, z=z2, other_agent1=(1, agent1), other_agent2=(3, agent3), learning_rate=step_size, kappa=kappa)

        #print('update pi1')
        agent3.update_pi1_SGD_distributed(agent_index=3, P=P, z=z3, other_agent1=(1, agent1), other_agent2=(2, agent2) ,learning_rate=step_size, kappa=kappa)
        #print('update pi2')
        agent3.update_pi2_SGD_distributed(agent_index=3, P=P, z=z3, other_agent1=(1, agent1), other_agent2=(2, agent2), learning_rate=step_size, kappa=kappa)

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
            print("agent1 : episode :{}, total cost : {:.1f}, start_position : {}, step_size : {}, kappa : {}, time : {}".format(episode, a1_total_cost, a1_start_position, step_size, kappa, a1_t))
            print("agent2 : episode :{}, total cost : {:.1f}, start_position : {}, step_size : {}, kappa : {}, time : {}".format(episode, a2_total_cost, a2_start_position, step_size, kappa, a2_t))
            print("agent3 : episode :{}, total cost : {:.1f}, start_position : {}, step_size : {}, kappa : {}, time : {}".format(episode, a3_total_cost, a3_start_position, step_size, kappa, a3_t))
        if episode % 100 == 0:
            print('agent1')
            for s in range(agent1.state_size):
                print(f'state : {s}\npi1 : {agent1.pi1.softmax_probs()[s]}')
            for s in range(agent1.state_size):
                print(f'state : {s}\npi2 : {agent1.pi2.softmax_probs()[s]}')

            print('\nagent2')
            for s in range(agent2.state_size):
                print(f'state : {s}\npi1 : {agent2.pi1.softmax_probs()[s]}')
            for s in range(agent2.state_size):
                print(f'state : {s}\npi2 : {agent2.pi2.softmax_probs()[s]}')
            
            print('\nagent3')
            for s in range(agent3.state_size):
                print(f'state : {s}\npi1 : {agent3.pi1.softmax_probs()[s]}')
            for s in range(agent3.state_size):
                print(f'state : {s}\npi2 : {agent3.pi2.softmax_probs()[s]}')


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
        file.write(f'episodes : {episodes}\nkappa : {agent1.kappa}\ngamma : {agent1.gamma}\nlambda : {agent1.Lambda}\nalpha_risk : {agent1.alpha_risk}\nmax_step : {200}\nstep_size : {step_size}')  # 数値データを記載




    # pi1.txt と pi2.txt を cost_history.csv と同じディレクトリに作成
    a1_pi1_file_path = os.path.join(output_folder, "a1_pi1.txt")
    a1_pi2_file_path = os.path.join(output_folder, "a1_pi2.txt")

    # pi1.txt の内容
    with open(a1_pi1_file_path, "w") as file:
        for s in range(agent1.state_size):
                file.write(f'state : {s}\npi1 : {agent1.pi1.softmax_probs()[s]}\n')

    # pi2.txt の内容
    with open(a1_pi2_file_path, "w") as file:
        for s in range(agent1.state_size):
            for eta_index in range(agent1.eta_size):
                file.write(f'state : {s}\neta : {agent1.eta_space[eta_index]}\npi2 : {agent1.pi2.softmax_probs()[s]}\n')

    # pi1.txt と pi2.txt を cost_history.csv と同じディレクトリに作成
    a2_pi1_file_path = os.path.join(output_folder, "a2_pi1.txt")
    a2_pi2_file_path = os.path.join(output_folder, "a2_pi2.txt")

    # pi1.txt の内容
    with open(a2_pi1_file_path, "w") as file:
        for s in range(agent2.state_size):
                file.write(f'state : {s}\npi1 : {agent2.pi1.softmax_probs()[s]}\n')

    # pi2.txt の内容
    with open(a2_pi2_file_path, "w") as file:
        for s in range(agent2.state_size):
            for eta_index in range(agent2.eta_size):
                file.write(f'state : {s}\neta : {agent2.eta_space[eta_index]}\npi2 : {agent2.pi2.softmax_probs()[s]}\n')

    # pi1.txt と pi2.txt を cost_history.csv と同じディレクトリに作成
    a3_pi1_file_path = os.path.join(output_folder, "a3_pi1.txt")
    a3_pi2_file_path = os.path.join(output_folder, "a3_pi2.txt")

    # pi1.txt の内容
    with open(a3_pi1_file_path, "w") as file:
        for s in range(agent3.state_size):
                file.write(f'state : {s}\npi1 : {agent3.pi1.softmax_probs()[s]}\n')

    # pi2.txt の内容
    with open(a3_pi2_file_path, "w") as file:
        for s in range(agent3.state_size):
            for eta_index in range(agent3.eta_size):
                file.write(f'state : {s}\neta : {agent3.eta_space[eta_index]}\npi2 : {agent3.pi2.softmax_probs()[s]}\n')