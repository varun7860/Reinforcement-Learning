import numpy as np
import gym
import random

env = gym.make("Taxi-v3")
env.render()

action_size  = env.action_space.n
state_size = env.observation_space.n

Q = np.zeros([state_size,action_size])

ALPHA = 0.1
GAMMA = 0.6
EPSILON = 1.0
EPISODES = 50000
TEST_EPISODES = 500
SHOW = 500

total_reward = 0
rewards = []

for t in range(EPISODES+1):
    state= env.reset()
    done = False
    penalty = 0

    if t%SHOW ==0:
        render = True
        print(t)

    else:
        render = False
        print(t)


    while not done:
        if random.uniform(0,1)> EPSILON:
            action = np.argmax(Q[state])

        else:
            action = env.action_space.sample()


        new_state, reward , done ,info = env.step(action)

        if render:
            env.render()

        Q_old = Q[state,action]
        Q_max = np.max(Q[new_state])

        Q_new = (1-ALPHA)*Q_old + ALPHA*(reward + GAMMA*Q_max)

        Q[state,action]= Q_new
        

        if reward == -10:
            penalty += 1

        state = new_state
    
print("Training Finished")



for i in range(TEST_EPISODES):
    state= env.reset()
    done = False

    if i%SHOW ==0:
        render = True
       

    else:
        render = False

    print("EPISODE:", i)

    while not done:
        action = np.argmax(Q[state])

        new_state,reward,done,info = env.step(action)

        total_reward += reward

        if done:
            rewards.append(reward)
            break

env.close()
print(rewards)
final_reward = sum(rewards)
print("Score over Time:  ",(final_reward/TEST_EPISODES))
       
