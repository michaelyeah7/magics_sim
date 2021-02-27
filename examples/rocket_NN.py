import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax
from envs import Rocket
from agents import Deep_Rocket
import copy
import pickle
from time import gmtime, strftime 

def f(state, params, env, agent):
    r, v, q, w = state
    input_state = jnp.concatenate([r, v, q, w])
    action = agent.__call__(input_state, params)
    next_state = env.dynamics(state,action)
    reward = env.reward_func(next_state)
    return reward

f_grad = jax.grad(f,argnums=1)


# def step(context, x):
def step(env, agent):
    # params, env, agent = context
    # env.render()
    r, v, q, w = env.state
    input_state = jnp.concatenate([r, v, q, w])
    control = agent(input_state, agent.params)
    print("control",control)
    # control = jnp.array([1.0,0.2,0.])
    prev_state = copy.deepcopy(env.state)
    print("prev_state[0]",prev_state[0])
    # print("control",control)
    reward, next_state, done = env.step(env.state, control)
    # if done:
    #     env.reset()

    # agent.feed(reward)
    # agent.update()
    grads = f_grad(prev_state, agent.params, env, agent)
    # print("grads",grads)
    agent.params = [(w - agent.lr * dw, b - agent.lr * db)
            for (w, b), (dw, db) in zip(agent.params, grads)]
    return (env, agent), reward, done

# def roll_out(params,env,agent,xs):
def roll_out(agent,env,init_state,T):
    # _, reward_scan = lax.scan(step, (params,env, agent), xs)
    # return jnp.sum(reward_scan)
    # control = jnp.array([0.,0.2,0.])
    # state = init_state
    loss = 0
    for i in range(T):
        # next_state = env.step(state,control)
        # print("next_state",next_state)
        # state = next_state
        (env, agent), reward, done = step(env, agent)

        # print("current reward",reward)
        loss += reward
        if done:
            print("episode done because out of threshold")
            break
            

    
    return loss


# Deep
env = Rocket()
init_r = jnp.array([10, -8, 5.])
ini_v_I = jnp.array([-.1, 0.0, -0.0])
ini_q = jnp.array(env.toQuaternion(1.5, [0, 0, 1]))
ini_w = jnp.array([0, -0.0, 0.0])

init_state = [init_r,ini_v_I,ini_q,ini_w]
action = jnp.array([0.,0.2,0.])

# next_r,next_v,next_q,next_w = env.dynamics(state,action)
# print("next_r",next_r)
# print("next_v",next_v)
# print("next_q",next_q)
# print("next_w",next_w)


agent = Deep_Rocket(
             env_state_size = 13,
             action_size = 3,
             action_space = jnp.array([0,1]),
             learning_rate = 0.001,
             gamma = 0.99,
             max_episode_length = 500,
             seed = 0
            )
params = pickle.load( open( "examples/rocket_params2021-02-26 17:24:57.txt", "rb" ) )
agent.reset()
agent.params = params

# print(env.reset())
# reward = 0
# loss = 0
episode_loss = []
episodes_num = 1
T = 50
# xs = jnp.array(jnp.arange(T))
# learning_rate = 0.1

# params = [1.0,2.0,3.0]
# grads = f_grad(init_state, params, env, agent)
# print("grads",grads)
for j in range(episodes_num):
    env.reset(init_state)
    # agent.params = params
    print("episode:{%d}" % j)
    # loss = roll_out(params,env,agent,xs)
    loss = roll_out(agent,env,init_state,T)
    #update weights
    # grad = jax.grad(rollout, argnums=0)(params,env,agent,xs)
    # params = [(w - agent.lr * dw, b - agent.lr * db)
    #         for (w, b), (dw, db) in zip(agent.params, grads)]
    
    episode_loss.append(loss)
    # print("loss:",loss)
    # loss = 0
    # env.reset()           

  

plt.plot(episode_loss[1:])
# plt.show()
# plt.savefig('rocket_loss'+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
# with open("rocket_params"+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
#     pickle.dump(agent.params, fp)

env.play_animation()  
