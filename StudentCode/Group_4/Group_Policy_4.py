# 将该文件改名为Group_Policy_X.py（X是你的组号）
# 文件夹名字"Group_4"中的X也需要修改
import torch

from StudentCode.BlackJackBattleEnv import BasePolicy
from StudentCode.Group_4.mc  import greedy_policy, load_policy


class Policy_4(BasePolicy): # X改为你的组号
    def __init__(self, threshold=17) -> None:
        self.threshold = threshold
        self.V = load_policy('StudentCode/Group_4/policy.pkl')

    def act_player(self, obs):
        return greedy_policy(self.V, obs)

    def act_dealer(self, obs):
        return 1 if obs[0] <= 17 else 0



