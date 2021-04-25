import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax
from envs.arm_rbdl import Arm_rbdl
# from agents import Deep_Arm_rbdl
from envs.panda_arm import Panda_Arm
from agent import Deep_Agent
import copy
import pickle
from time import gmtime, strftime 
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
import numpy as np
from model_based_RL import MBRL
import os


#configs
lr = 1e-3
episodes_num = 1000
T = 100 #time steps of each episode
horizon = 20 #rollout horizon
render_flag = True
load_params = False
update_params = True
save_interval = 10

# Deep
env = Panda_Arm(render_flag=render_flag)
hybrid_env = None
agent = Deep_Agent(
             state_size = 14,
             action_size = 7,
            )
#init learner
mbrl = MBRL(env, agent, lr = lr)

if load_params == True:
    loaded_params = pickle.load( open( "experiments2021-04-12 22:14:59/cartpole_svg_params_episode_10_2021-04-12 22:22:35.txt", "rb" ) )
    agent.params = loaded_params

exp_dir = "arm_experiments_lr_%f_horizon_%d_" % (lr, horizon)+ strftime("%Y-%m-%d %H:%M:%S", gmtime())
os.mkdir(exp_dir)


#begin training
episode_rewards = []
for j in range(episodes_num):

    rewards = 0
    env.reset()           
    print("episode:{%d}" % j)

    #evaluate rewards and update value function
    rewards, trajectory_state_buffer = mbrl.roll_out_for_render(env, hybrid_env, agent, agent.params, T)

    #update the policy
    if (update_params==True):
        #update policy using 20 horizon 5 partial trajectories
        # for i in range(20):
        env.reset()
        # hybrid_env.reset() 

        random_state_index = np.random.randint(len(trajectory_state_buffer), size=1)[0]
        env.state =  trajectory_state_buffer[random_state_index]

        #train policy use 5-step partial trajectory and learned value function
        # total_return, grads = mbrl.f_grad(env, agent, agent.params, T)
        total_return, grads = mbrl.f_grad(env, agent, (agent.params, agent.value_params), T)
        # total_return, grads = mbrl.f_grad(hybrid_env, agent, (agent.params, agent.value_params),T)

        #get and update policy and value function grads
        policy_grads, value_grads = grads 
        # policy_grads = grads 
        # print("policy_grads",policy_grads)         
        agent.params = mbrl.update(policy_grads, agent.params, mbrl.lr)
        agent.value_params =  mbrl.update(value_grads,agent.value_params, mbrl.lr)

    episode_rewards.append(rewards)
    print("rewards is %f" % rewards)
    # print("hybrid_env.model_losses is %f" % hybrid_env.model_losses[j])
    # if (j%10==0 and j!=0 and update_params==True):
    if (j % save_interval == 0 and j!=0):
        #for agent loss
        with open(exp_dir + "/arm_svg_params"+ "_episode_%d_" % j +".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.params, fp)
        with open(exp_dir + "/arm_rewards"+ "_episode_%d_" % j + ".txt", "wb") as fp:   #Pickling
            pickle.dump(episode_rewards[1:], fp)
        plt.figure()
        plt.plot(episode_rewards[1:])
        plt.savefig((exp_dir + '/arm_svg_loss_episode_%d_' % j) + '.png')
        plt.close()
        #for value function loss
        with open(exp_dir + "/arm_svg_value_params"+ "_episode_%d_" % j +".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.value_params, fp)
        plt.figure()
        plt.plot(agent.value_losses)
        plt.savefig((exp_dir + '/cartpole_svg_agent_value_loss_episode_%d_' % j) + '.png')
        plt.close()        
        # #for model loss
        # with open(exp_dir + "/cartpole_svg_model_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
        #     pickle.dump(hybrid_env.model_params, fp)
        # plt.figure()
        # plt.plot(hybrid_env.model_losses)
        # plt.savefig((exp_dir + '/cartpole_svg_model_loss_episode_%d_' % j) + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        # plt.close()
