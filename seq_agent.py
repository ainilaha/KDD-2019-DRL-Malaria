import numpy as np
from netsapi.challenge import *
from sys import exc_info


class SeqCustomAgent:
    def __init__(self, environment, episode_number=20):
        self.environment = environment
        self.episode_number = episode_number

        self.run = []
        self.scores = []
        self.policies = []

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        candidates = []
        try:
            # select a set of random candidate solutions to be evaluated
            for i in range(self.episode_number):
                self.environment.reset()
                policy = {}
                for j in range(5):
                    policy[str(j + 1)] = [random.random(), random.random()]
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
