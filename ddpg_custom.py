import numpy as np
from netsapi.challenge import *
from sys import exc_info
import torch
from collections import deque
from ddpg_agent import Agent


class DDPGCustomAgent:
    def __init__(self, environment, episode_number=20, max_time_steps=5):
        self.environment = environment
        self.episode_number = episode_number
        self.max_time_steps = max_time_steps
        self.run = []
        self.scores = []
        self.policies = []
        self.agent = Agent(state_size=1, action_size=2, random_seed=0)

    def train(self):
        # scores_window = deque(maxlen=100)
        # scores = np.zeros(1)
        # scores_episode = []
        for i in range(self.episode_number):
            policy = {}
            self.environment.reset()
            next_state = self.environment.state
            while True:
                state = next_state
                action = self.agent.act(state=state)
                policy[str(state)] = list(action[0])
                # candidates.append(policy)
                #reward = self.environment.evaluateReward(action[0])
                next_state, reward, done, _ = self.environment.evaluateAction(action)
                if done:
                    break

        torch.save(Agent.actor_local.state_dict(), 'models/arm_actor.pth')
        torch.save(Agent.critic_local.state_dict(), 'models/arm_critic.pth')

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        candidates = []
        try:
            # select a set of random candidate solutions to be evaluated
            for i in range(self.episode_number):
                print("epoch=", i)
                self.environment.reset()

                # for t in range(self.max_time_steps):
                policy = {}
                for j in range(self.max_time_steps):
                    state = j + 1
                    state_array = np.array([state]).reshape(1, 1)
                    # print("state=", state_array)
                    # print("action=", list(self.agent.act(state=state_array)[0]))
                    policy[str(state)] = list(self.agent.act(state=state_array)[0])
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
    env = ChallengeSeqDecEnvironment()
    # # env.evaluatePolicy()
    # a = CustomAgent(env)
    # a.scoringFunction()
    # a.create_submissions("test.csv")
    # print("Hello KDD2019")
    # EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, DDPGCustomAgent, "tutorial.csv")

    # env = ChallengeEnvironment(experimentCount=5)
    agent = DDPGCustomAgent(environment=env, episode_number=10)
    agent.train()
