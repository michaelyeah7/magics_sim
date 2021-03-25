import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax
from envs import Qaudrupedal
from agents import Deep_Qaudrupedal
import copy
import pickle
from time import gmtime, strftime 
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
import numpy as np
from jax.api import jit
from functools import partial





def loop(context, x):
    env, agent, params = context
    control = agent(env.state, params)
    prev_state = copy.deepcopy(env.state)
    _, reward, done, _ = env.step(env.state,control)

    return (env, agent), reward, done

# @partial(jit, static_argnums=(0, 1))
def roll_out(env, agent, params):
    losses = 0.0
    for i in range(100):
        (env, agent), r, done= loop((env, agent,params), i)
        losses += r 
        if done:
            print("end this episode because out of threshhold, total %d steps " % i)
            env.past_reward = 0
            break
        
    return losses

# f_grad = jax.grad(forward,argnums=1)


f_grad = jax.grad(roll_out,argnums=2)

def loop_for_render(context, x):
    env, agent, params = context
    if(render==True):
        env.osim_render()
    control = agent(env.state, params)
    prev_state = copy.deepcopy(env.state)
    _, reward, done, _ = env.step(env.state,control)

    return (env, agent), reward, done

def roll_out_for_render(env, agent, params):
    gamma = 0.9
    losses = 0.0
    for i in range(100):
        (env, agent), r, done= loop_for_render((env, agent,params), i)
        losses = losses * gamma + r 
        if done:
            print("end this episode because out of threshhold")
            env.past_reward = 0
            break
        
    return losses



# Deep
env = Qaudrupedal()
#Quadrupedal has 14 joints
agent = Deep_Qaudrupedal(
             env_state_size = 28,
             action_space = jnp.zeros(14),
             learning_rate = 0.1,
             gamma = 0.99,
             max_episode_length = 500,
             seed = 0
            )

# load_params = False
# update_params = True
# render = False

load_params = False
update_params = True
render = True

if load_params == True:
    loaded_params = pickle.load( open( "examples/qudrupedal_params_episode_270_2021-03-21 16:41:06.txt", "rb" ) )
    agent.params = loaded_params

reward = 0
loss = 0
episode_loss = []
episodes_num = 500
T = 100
for j in range(episodes_num):

    loss = 0
    env.reset()           
    print("episode:{%d}" % j)
    loss = roll_out_for_render(env, agent, agent.params)

    #update the parameter
    if (update_params==True):
        # grads = f_grad(prev_state, agent.params, env, agent)
        grads = f_grad(env, agent, agent.params)
        print("grads",grads)
        #get norm square
        total_norm_sqr = 0                
        for (dw,db) in grads:
            # print("previous dw",dw)
            # dw = normalize(dw)
            # db = normalize(db[:,np.newaxis],axis =0).ravel()
            total_norm_sqr += np.linalg.norm(dw) ** 2
            total_norm_sqr += np.linalg.norm(db) ** 2
        # print("grads",grads)
        #scale the gradient
        gradient_clip = 0.2
        scale = min(
            1.0, gradient_clip / (total_norm_sqr**0.5 + 1e-4))

        agent.params = [(w - agent.lr * scale * dw, b - agent.lr * scale * db)
                for (w, b), (dw, db) in zip(agent.params, grads)]


    episode_loss.append(loss)
    print("loss is %f " % loss)
    if (j%10==0 and j!=0 and update_params==True):
        with open("examples/qudrupedal_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.params, fp)
# reward_forloop = reward
# print('reward_forloop = ' + str(reward_forloop))
plt.plot(episode_loss[1:])

#save plot and params
plt.savefig('quadrupedal_loss'+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')

# fp.close()