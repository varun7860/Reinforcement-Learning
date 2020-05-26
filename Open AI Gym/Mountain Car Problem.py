import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython import display


env = gym.make("CartPole-v0")
env.render()

TIME_LIMIT = 250
SHOW_EVERY = 250
env = gym.wrappers.TimeLimit(
    gym.envs.classic_control.MountainCarEnv(),
    max_episode_steps= TIME_LIMIT+1,
    )


#actions = {'left': 0, 'stop': 1, 'right': 2}

ALPHA = 0.1

GAMMA = 0.95

EPSILON = 0.1

discrete_obs_size = [20]*len(env.observation_space.high)
discrete_obs_win_size = (env.observation_space.high-env.observation_space.low)/discrete_obs_size

Q =  np.random.uniform(low = -2, high = 0, size = (discrete_obs_size +[env.action_space.n]))
print(Q)

def Discrete_Obs(obs):
    discrete_obs = (obs - env.observation_space.low)/discrete_obs_win_size
    return tuple(discrete_obs.astype(np.int))

    
def policy(obs,t):
    action = np.argmax(Q[discrete_obs])
    return action

plt.figure(figsize=(4, 3))
display.clear_output(wait=True)


for t in range(TIME_LIMIT+1):
    discrete_obs = Discrete_Obs(env.reset())
    done = False
    
    if t%SHOW_EVERY == 0:
        render = True
        print(t)

    else:
        render = False
        print(t)
    
    while not done:
        if random.uniform(0,1)>EPSILON:
            action = policy(discrete_obs,t)

        else:
            action = env.action_space.sample()
        
        new_obs,reward,done,info = env.step(action)
        new_discrete_obs =Discrete_Obs(new_obs)

        
        if render:
            plt.imshow(env.render('rgb_array'))
            display.clear_output(wait=True)
            display.display(plt.gcf())

        if not done:
            Q_max = np.max(Q[new_discrete_obs])
            Q_old = Q[discrete_obs + (action,)]


            
            Q_new = (1- ALPHA)*Q_old  + ALPHA*(reward+GAMMA *Q_max)
            Q[discrete_obs + (action,)] = Q_new

        elif new_obs[0]> env.goal_position:
            Q[discrete_obs + (action,)] = 0

        discrete_obs = new_discrete_obs


display.clear_output(wait=True)
