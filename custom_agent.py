import numpy as np
import pandas as pd

from netsapi.challenge import *
from sys import exc_info


class CustomAgent:
    def __init__(self, environment, popsize=10):
        self.popsize = popsize
        self.environment = environment
        self.popsize = popsize

        self.episodes = []
        self.scores = []
        self.policies = []

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        try:
            # select a set of random candidate solutions to be evaluated
            policies = np.random.rand(self.popsize, self.environment.policyDimension)
            rewards = self.environment.evaluateReward(policies)
            best_policy = policies[np.argmax(rewards), :]

        except (KeyboardInterrupt, SystemExit):
            print(exc_info())

        return best_policy

    def scoringFunction(self):
        scores = []
        for ii in range(10):
            self.environment.reset()
            finalresult = self.generate()
            self.policies.append(finalresult)
            reward = self.environment.evaluateReward(finalresult)
            self.scores.append(reward)
            self.episodes.append(ii)

        return np.mean(self.scores) / np.std(self.scores)

    def create_submissions(self, filename='my_submission.csv'):
        labels = ['episode_no', 'reward', 'policy']
        rewards = np.array(self.scores)
        data = {'episode_no': self.episodes,
                'rewards': rewards,
                'policy': self.policies,
                }
        submission_file = pd.DataFrame(data)
        submission_file.to_csv(filename, index=False)


if __name__ == '__main__':
    env = ChallengeEnvironment(experimentCount=1000)
    a = CustomAgent(env)
    a.scoringFunction()
    a.create_submissions("test.csv")
