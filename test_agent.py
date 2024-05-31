"""评估 Agent 性能
"""
import sys
import numpy as np
import xlsxwriter
import time
from StudentCode.BlackJackBattleEnv import BlackJack, SimplePolicy
from StudentCode import agents, agent_names


def battle(agent1, agent1_name, agent2, agent2_name, num_round=1000, logger=sys.stdout, seed=0):
    """对比两个智能体的性能，评估 num_round 轮，智能体轮流坐庄

    Parameters
    ----------
    agent1 : Policy
        智能体1
    agent1_name : str
        智能体1 名称
    agent2 : Policy
        智能体2
    agent2_name : str
        智能体2 名称
    num_rount : int, optional
        总轮数，其中一半智能体1坐庄, by default 100
    logger : File, optional
        日志文件对象, by default None
    seed : int, optional
        随机数种子

    Return
    ------
    win_rate : float
        智能体 1 总胜率
    win_hc : int
        智能体 1 坐庄胜场
    draw_hc : int
        智能体 1 坐庄平局数
    win_vc : int
        智能体 1 闲家胜场
    draw_vc : int
        智能体 1 闲家平局数
    """
    win_hc, draw_hc, win_vc, draw_vc = 0, 0, 0, 0
    half_num_round = num_round // 2
    
    logger.write('='*80 + '\n')
    logger.write(f'智能体 {agent1_name} vs. 智能体 {agent2_name}\n')
    logger.write(f'智能体 {agent1_name} 坐庄\n')

    # 智能体 1 坐庄
    env = BlackJack(agent1, logger=logger)
    for c in range(half_num_round):  # 重复 num_round//2 轮
        logger.write('-'*40 + f'第 {c+1:3} 轮' + '-'*40 + '\n')

        # 初始化
        obs = env.reset(seed+c)
        done = False

        # 决策
        action = agent2.act_player(obs)
        while action == 1:  # 玩家不停抽牌
            obs, reward, done, _ = env.step(action)

            if done:  # 智能体 2 爆牌
                win_hc += 1
                break
            else:
                action = agent2.act_player(obs)
        
        if done:  # 智能体 2 爆牌
            continue

        # 玩家停牌，庄家开始决策，并评估结果，一轮到此结束
        obs, reward, done, _ = env.step(action)
        if reward == -1:  # 庄家胜利
            win_hc += 1
        elif reward == 0:
            draw_hc += 1

    # 智能体 2 坐庄
    logger.write(f'\n智能体 {agent2_name} 坐庄\n')
    env = BlackJack(agent2, logger=logger)
    for c in range(half_num_round, num_round):  # 重复 num_round//2 轮
        logger.write('-'*40 + f'第 {c+1:3} 轮' + '-'*40 + '\n')

        # 初始化
        obs = env.reset(seed+c)
        done = False

        # 决策
        action = agent1.act_player(obs)
        while action == 1:  # 玩家不停抽牌
            obs, reward, done, _ = env.step(action)

            if done:  # 智能体 1 爆牌
                assert reward == -1, '理应玩家爆牌，得到 -1 回报'
                break
            else:
                action = agent1.act_player(obs)
        
        if done:  # 智能体 1 爆牌
            continue

        # 玩家停牌，庄家开始决策，并评估结果，一轮到此结束
        obs, reward, done, _ = env.step(action)
        if reward == 1:
            win_vc += 1
        elif reward == 0:
            draw_vc += 1

    logger.write('\n\n')

    # 计算总胜率
    win_rate = (win_hc + win_vc) / num_round
    return win_rate, win_hc, draw_hc, win_vc, draw_vc

def test():
    agent1 = SimplePolicy(17)
    agent1_name = 'SimplePolicy17'
    agent2 = SimplePolicy(15)
    agent2_name = 'SimplePolicy15'
    with open('test.txt', 'w') as of:
        win_rate, win_hc, draw_hc, win_vc, draw_vc = battle(
            agent1, agent1_name, agent2, agent2_name, num_round=20, logger=of)

    score = (2 * win_hc + draw_hc) + (2 * win_vc + draw_vc) - 20
    print(f'智能体 {agent1_name:20} v.s. {agent2_name:20} | 胜率 {win_rate:.1%}')
    print(win_rate, win_hc, draw_hc, win_vc, draw_vc)
    print(score)

if __name__ == '__main__':
    # 注册智能体
    agent_num = len(agents)
    num_round = 100000
    seed=int(time.time())
    # 对局得分表
    scores = np.zeros((agent_num, agent_num))

    # 日志
    logger = open('test.log', 'w')
    score_logger = open('score.log', 'w')

    for i in range(agent_num):
        for j in range(i+1, agent_num):
            win_rate, win_hc, draw_hc, win_vc, draw_vc = battle(agents[i],
                agent_names[i], 
                agents[j], 
                agent_names[j],
                num_round=num_round,
                logger=logger,
                seed=seed+i*agent_num+j)

            # 得分：胜负差
            score = (2 * win_hc + draw_hc) + (2 * win_vc + draw_vc) - num_round
            scores[i, j] = score
            scores[j, i] = -score

            print(f'智能体 {agent_names[i]:20} v.s. {agent_names[j]:20} | 胜率 {win_rate:.1%}')

            score_logger.write('='*60 + '\n')
            score_logger.write(f'智能体 {agent_names[i]:20} v.s. {agent_names[j]:20}\n')
            score_logger.write('-'*60+ '\n')
            score_logger.write(f'智能体 {agent_names[i]:20} 胜率 | {win_rate:.1%}\n')
            score_logger.write(f'{"总对局":29} | {num_round}\n')
            score_logger.write(f'智能体 {agent_names[i]:20} 主场 | 胜 {win_hc:3} | 平 {draw_hc:3} | 负 {num_round//2-win_hc-draw_hc:3}\n')
            score_logger.write(f'智能体 {agent_names[i]:20} 客场 | 胜 {win_vc:3} | 平 {draw_vc:3} | 负 {num_round//2-win_vc-draw_vc:3}\n')
            score_logger.write('='*60+ '\n')
            score_logger.write('\n')

    # 计算平均得分
    mean_score = np.mean(scores, axis=1)

    # 记录
    ob = xlsxwriter.Workbook('得分表.xlsx')
    # 记录对局记录
    worksheet = ob.add_worksheet('对局记录')
    worksheet.write_row(0, 1, agent_names)
    worksheet.write_column(1, 0, agent_names)
    for i, row in enumerate(scores):
        worksheet.write_row(i+1, 1, scores[i])

    # 记录平均得分
    worksheet = ob.add_worksheet('平均得分')
    worksheet.write_column(0, 0, ['组名'] + agent_names)
    worksheet.write_column(0, 1, ['平均得分'] + mean_score.tolist())
    ob.close()

    logger.close()
    score_logger.close()