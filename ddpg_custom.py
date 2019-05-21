import numpy as np
from netsapi.challenge import *
from sys import exc_info
import torch
from collections import deque
from ddpg_agent import Agent

class DDPGCustomAgent:
    def __init__(self, environment, episode_number=10,max_time_steps=1000):
        self.environment = environment
        self.episode_number = episode_number
        self.max_time_steps = max_time_steps
        self.run = []
        self.scores = []
        self.policies = []
        self.agent = Agent(state_size=5, action_size=2, random_seed=0)

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        candidates = []

        try:
            # select a set of random candidate solutions to be evaluated
            for i in range(self.episode_number):
                self.environment.reset()

                for t in range(self.max_time_steps):
                    policy = {}
                    for j in range(5):
                        state = j + 1
                        policy[str(state)] = np.array(self.agent.act(state=state))
                        self.environment.evaluatePolicy(candidates)  # send the action to the environment
                    candidates.append(policy)
            print("candidates len=", len(candidates))
            rewards = self.environment.evaluatePolicy(candidates)
            best_policy = candidates[np.argmax(rewards)]
            best_reward = rewards[np.argmax(rewards)]

        except (KeyboardInterrupt, SystemExit):
            print(exc_info())

        return best_policy, best_reward

    def scoringFunction(self):
        scores = []
        for ii in range(10):
            self.environment.reset()
            final_result, reward = self.generate()
            self.policies.append(final_result)
            self.scores.append(reward)
            self.run.append(ii)

        return np.mean(self.scores) / np.std(self.scores)

    def create_submissions(self, filename='my_submission.csv'):
        labels = ['episode_no', 'reward', 'policy']
        rewards = np.array(self.scores)
        data = {'episode_no': self.run,
                'rewards': rewards,
                'policy': self.policies,
                }
        submission_file = pd.DataFrame(data)
        submission_file.to_csv(filename, index=False)


if __name__ == '__main__':
    # env = ChallengeSeqDecEnvironment()
    # # env.evaluatePolicy()
    # a = CustomAgent(env)
    # a.scoringFunction()
    # a.create_submissions("test.csv")
    # print("Hello KDD2019")
    EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, CustomAgent, "tutorial.csv")
