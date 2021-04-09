import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax
from envs import Cartpole_rbdl, Cartpole_Hybrid
from agents import Deep_Cartpole_rbdl
import copy
import pickle
from time import gmtime, strftime 
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
import numpy as np
import os
from model_based_ActorCritic import MBAC


# Init env and agent
env = Cartpole_rbdl(render=True) 
hybrid_env = Cartpole_Hybrid(model_lr=5e-1)
agent = Deep_Cartpole_rbdl(
             env_state_size = 4,
             action_space = jnp.array([0]),
             learning_rate = 0.5,
             gamma = 0.99,
             max_episode_length = 500,
             seed = 0
            )

# load_params = True
# update_params = False
# render = True

load_params = False
update_params = True

if load_params == True:
    loaded_params = pickle.load( open( "examples/cartpole_svg_params_episode_100_2021-04-05 06:10:53.txt", "rb" ) )
    agent.params = loaded_params

#init learner
mbac = MBAC(env, agent)

episode_rewards = []
episodes_num = 1000
T = 100
exp_dir = "experiments" + strftime("%Y-%m-%d %H:%M:%S", gmtime())
os.mkdir(exp_dir)



for j in range(episodes_num):

    rewards = 0
    env.reset()           
    print("episode:{%d}" % j)

    #evaluate rewards and update value function
    rewards = mbac.roll_out_for_render(env, hybrid_env, agent, agent.params, T)

    #update the policy
    if (update_params==True):
        #update policy using 20 horizon 5 partial trajectories
        for i in range(20):
            env.reset()
            # hybrid_env.reset() 

            #train policy use 5-step partial trajectory and learned value function
            total_return, grads = mbac.f_grad(env, agent, (agent.params, agent.value_params), T)
            # total_return, grads = f_grad(hybrid_env, agent, (agent.params, agent.value_params),T)

            #get and update policy and value function grads
            policy_grads, value_grads = grads           
            agent.params = agent.update(policy_grads, agent.params, agent.lr)
            agent.value_params =  agent.update(value_grads,agent.value_params, agent.lr)

    episode_rewards.append(rewards)
    print("rewards is %f" % rewards)
    if (j%10==0 and j!=0 and update_params==True):
        #for agent loss
        with open(exp_dir + "/cartpole_svg_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.params, fp)
        plt.figure()
        plt.plot(episode_rewards[1:])
        plt.savefig((exp_dir + '/cartpole_svg_loss_episode_%d_' % j)+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        plt.close()
        #for value function loss
        with open(exp_dir + "/cartpole_svg_value_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.value_params, fp)
        plt.figure()
        plt.plot(agent.value_losses)
        plt.savefig((exp_dir + '/cartpole_svg_agent_value_loss_episode_%d_' % j) + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        plt.close()        
        # #for model loss
        # with open(exp_dir + "/cartpole_svg_model_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
        #     pickle.dump(hybrid_env.model_params, fp)
        # plt.figure()
        # plt.plot(hybrid_env.model_losses)
        # plt.savefig((exp_dir + '/cartpole_svg_model_loss_episode_%d_' % j) + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        # plt.close()
