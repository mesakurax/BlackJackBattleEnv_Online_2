from StudentCode.BlackJackBattleEnv import SimplePolicy

# Group player
from StudentCode.Group_4.Group_Policy_4 import Policy_4 as P4# 可以参考上面的group_test
Group4 = P4()

# 所有组的策略
agents = [SimplePolicy(), Group4]
agent_names = ['Agent17', 'Group4']

assert len(agents) == len(agent_names), '智能体和组名个数不匹配'
