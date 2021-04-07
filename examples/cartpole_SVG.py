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

def forward(state, w, env, agent):
    action = agent.__call__(state, w)
    next_state = env.dynamics(state,action)
    # next_state, reward, done, _ = env.step(state,action)
    reward = env.reward_func(next_state)
    return reward




def loop(context, x):
    env, agent, params = context
    control = agent(env.state, params)
    prev_state = copy.deepcopy(env.state)
    _, reward, done, _ = env.step(env.state,control)

    return (env, agent), reward, done

def roll_out(env, agent, params, T):
    gamma = 0.9
    losses = 0.0
    for i in range(5):
        (env, agent), r, done= loop((env, agent,params), i)
        losses = losses * gamma + r 
        if done:
            print("end this episode because out of threshhold in policy update")
            env.past_reward = 0
            break
    losses += agent.value(env.state,agent.value_params) * gamma        
    return losses

f_grad = jax.value_and_grad(roll_out,argnums=2)

def loss_value(state, next_state, reward, value_params):
    td = reward + agent.value(next_state, value_params) - agent.value(state, value_params)
    value_loss = 0.5 * (td ** 2)
    return value_loss

value_loss_grad = jax.value_and_grad(loss_value,argnums=3)

def loss_hybrid_model(prev_state, control, true_next_state, model_params):
    next_state = hybrid_env.forward(prev_state, control, model_params)
    model_loss = jnp.sum((next_state - true_next_state)**2)
    # model_loss = jnp.linalg.norm(next_state - true_next_state)
    # print("model loss",model_loss)
    # print("model_loss.value",model_loss[0])
    # model_losses.append(model_loss)
    return model_loss
# model_loss_grad = jax.grad(loss_hybrid_model,argnums=3)
model_loss_grad = jax.value_and_grad(loss_hybrid_model,argnums=3)

def loop_for_render(context, x):
    env, hybrid_env, agent, params = context
    if(render==True):
        env.osim_render()
    control = agent(env.state, params)
    prev_state = copy.deepcopy(env.state)
    next_state, reward, done, _ = env.step(env.state,control)

    #update value function
    value_loss, value_grads =  value_loss_grad(prev_state,next_state,reward,agent.value_params)
    agent.value_losses.append(value_loss)
    agent.value_params = agent.update(value_grads,agent.value_params,agent.lr)    
    
    #update hybrid model
    model_loss, model_grads = model_loss_grad(prev_state,control,next_state,hybrid_env.model_params)
    # print("model_loss",model_loss)
    hybrid_env.model_losses.append(model_loss)
    hybrid_env.model_params = agent.update(model_grads,hybrid_env.model_params,hybrid_env.model_lr)


    return (env, hybrid_env, agent), reward, done

def roll_out_for_render(env, hybrid_env, agent, params, T):
    gamma = 0.9
    losses = 0.0
    for i in range(T):
        (env, hybrid_env, agent), r, done= loop_for_render((env, hybrid_env, agent,params), i)
        losses = losses * gamma + r 
        if done:
            print("end this episode because out of threshhold in model update")
            env.past_reward = 0
            break
        
    return losses



# Deep
env = Cartpole_rbdl() 
hybrid_env = Cartpole_Hybrid(model_lr=1e-1)
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
render = True

if load_params == True:
    loaded_params = pickle.load( open( "examples/cartpole_svg_params_episode_100_2021-04-05 06:10:53.txt", "rb" ) )
    agent.params = loaded_params

 # for loop version
# xs = jnp.array(jnp.arange(T))
print(env.reset())
reward = 0
loss = 0
episode_loss = []
# episodes_num = 1000
episodes_num = 1000
T = 100
# T = 1000
for j in range(episodes_num):

    loss = 0
    env.reset()           
    print("episode:{%d}" % j)

    #update hybrid model using real trajectories
    loss = roll_out_for_render(env, hybrid_env, agent, agent.params, T)
    # print("hybrid_env.model_losses",hybrid_env.model_losses)

    #update the parameter
    if (update_params==True):
        #update policy using 20 horizon 5 partial trajectories
        for i in range(20):
            # env.reset()
            hybrid_env.reset() 
            # grads = f_grad(prev_state, agent.params, env, agent)

            #train agent using learned hybrid env
            total_return, grads = f_grad(hybrid_env, agent, agent.params,T)
            # grads = f_grad(env, agent, agent.params, T)
            agent.params = agent.update(grads, agent.params, agent.lr)

    episode_loss.append(loss)
    print("loss is %f" % loss)
    if (j%100==0 and j!=0 and update_params==True):
        #for agent loss
        with open("examples/cartpole_svg_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.params, fp)
        plt.figure()
        plt.plot(episode_loss[1:])
        plt.savefig('cartpole_svg_loss'+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        plt.close()
        #for value function loss
        with open("examples/cartpole_svg_value_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.value_params, fp)
        plt.figure()
        plt.plot(agent.value_losses)
        plt.savefig(('cartpole_svg_agent_value_loss_episode_%d_' % j) + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        plt.close()        
        #for model loss
        with open("examples/cartpole_svg_model_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
            pickle.dump(hybrid_env.model_params, fp)
        plt.figure()
        plt.plot(hybrid_env.model_losses)
        plt.savefig(('cartpole_svg_model_loss_episode_%d_' % j) + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        plt.close()
# reward_forloop = reward
# print('reward_forloop = ' + str(reward_forloop))
# plt.plot(episode_loss[1:])
# plt.plot(hybrid_env.model_losses)

#save plot and params
# plt.savefig('cartpole_svg_loss'+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
# plt.savefig('cartpole_svg_model_loss'+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')

# fp.close()