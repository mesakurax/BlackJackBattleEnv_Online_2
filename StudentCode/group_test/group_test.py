class TestPolicy:
    def __init__(self, threshold=17) -> None:
        self.threshold = threshold

    def act_player(self, obs):
        return 1 if obs[0] < self.threshold else 0

    def act_dealer(self, obs):
        return 1 if obs[0] < self.threshold else 0