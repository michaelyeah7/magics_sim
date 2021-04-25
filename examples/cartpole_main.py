import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax
# from envs import Cartpole_rbdl, Cartpole_Hybrid
from envs.cartpole_rbdl import Cartpole_rbdl, Cartpole_Hybrid
# from agents import Deep_Cartpole_rbdl
from agent import Deep_Agent
import copy
import pickle
from time import gmtime, strftime 
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
import numpy as np
import os
from model_based_RL import MBRL


#configs
lr = 1e-3
batch_size = 512
episodes_num = 2000
T = 500 #time steps of each episode
horizon = 20 #rollout horizon
render_flag = True
load_params = False
update_params = True
save_interval = 10

# Init env and agent
env = Cartpole_rbdl(render_flag=render_flag) 
hybrid_env = Cartpole_Hybrid(model_lr=5e-1)
agent = Deep_Agent(
             state_size = 4,
             action_size = 1,
            )
#init learner
mbrl = MBRL(env, agent, lr = lr, batch_size = batch_size)

if load_params == True:
    loaded_policy_params = pickle.load( open( "examples/models/cartpole/cartpole_svg_params_episode_990_.txt", "rb" ) )
    agent.params = loaded_policy_params
    loaded_value_params = pickle.load( open( "examples/models/cartpole/cartpole_svg_value_params_episode_990_.txt", "rb" ) )
    agent.value_params = loaded_policy_params   

exp_dir = "cartpole_experiments_" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "lr_%f_horizon_%d_batch_size_%d" % (lr, horizon, batch_size) 
os.mkdir(exp_dir)
print("exp_dir",exp_dir)

#begin training
episode_rewards = []
for j in range(episodes_num):
    rewards = 0
    env.reset()           
    print("episode:{%d}" % j)

    #evaluate rewards and update value function
    rewards, trajectory_state_buffer = mbrl.roll_out_for_render(env, hybrid_env, agent, agent.params, T)
    # rewards, trajectory_state_buffer = mbrl.roll_out_for_render(env, hybrid_env, agent, (agent.params, agent.rnn_params), T)

    #update the policy
    if (update_params==True):
        env.reset()
        # hybrid_env.reset() 

        random_state_index = np.random.randint(len(trajectory_state_buffer), size=1)[0]
        # print("random_state_index",random_state_index)
        # print("trajectory_state_buffer[random_state_index]",trajectory_state_buffer.shape)
        env.state =  trajectory_state_buffer[random_state_index]

        #train policy use 5-step partial trajectory and learned value function
        total_return, grads = mbrl.f_grad(env, agent, (agent.params, agent.value_params), T)
        # total_return, grads = mbrl.f_grad(env, agent, (agent.params, agent.rnn_params), T)
        # total_return, grads = mbrl.f_grad(env, agent, agent.params, T)
        # total_return, grads = mbrl.f_grad(hybrid_env, agent, (agent.params, agent.value_params),T)

        #get and update policy and value function grads
        policy_grads, value_grads = grads 
        # policy_grads, rnn_grads = grads 
        # policy_grads = grads 
        # print("policy_grads",policy_grads)         
        agent.params = mbrl.update(policy_grads, agent.params, mbrl.lr)
        agent.value_params =  mbrl.update(value_grads,agent.value_params, mbrl.lr)
        # agent.rnn_params = agent.update(rnn_grads, agent.rnn_params, agent.lr)
        # agent.state = jnp.zeros((agent.env_state_size,))
        # agent.h_t = jnp.zeros(4)

    episode_rewards.append(rewards)
    print("rewards is %f" % rewards)
    # print("hybrid_env.model_losses is %f" % hybrid_env.model_losses[j])
    # if (j%10==0 and j!=0 and update_params==True):
    if (j % save_interval == 0 and j!=0):
        #for agent loss
        with open(exp_dir + "/cartpole_svg_params"+ "_episode_%d_" % j + ".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.params, fp)
        # with open(exp_dir + "/cartpole_rnn_params"+ "_episode_%d_" % j  +".txt", "wb") as fp:   #Pickling
        #     pickle.dump(agent.rnn_params, fp)
        with open(exp_dir + "/cartpole_rewards"+ "_episode_%d_" % j + ".txt", "wb") as fp:   #Pickling
            pickle.dump(episode_rewards[1:], fp)
        plt.figure()
        plt.plot(episode_rewards[1:])
        plt.savefig((exp_dir + '/cartpole_svg_loss_episode_%d_' % j)+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        plt.close()

        #for value function loss
        with open(exp_dir + "/cartpole_svg_value_params"+ "_episode_%d_" % j + ".txt", "wb") as fp:   #Pickling
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
