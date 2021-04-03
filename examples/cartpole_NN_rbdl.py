import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax
from envs import CartPole, Cartpole_rbdl
from agents import Deep_Cartpole, Deep_Cartpole_rbdl
import copy
import pickle
from time import gmtime, strftime 
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore

def forward(state, w, env, agent):
    action = agent.__call__(state, w)
    next_state = env.dynamics(state,action)
    # next_state, reward, done, _ = env.step(state,action)
    reward = env.reward_func(next_state)
    return reward

f_grad = jax.grad(forward,argnums=1)
# f_grad = jax.jit(jax.jacfwd(forward,argnums=1))


def loop(context, x):
    env, agent = context
    if(render==True):
        # env.render()
        env.osim_render()
        # env.py_bullet_render()
    control = agent(env.state, agent.params)
    prev_state = copy.deepcopy(env.state)
    # print("control",control)
    # _, reward, done, _ = env.step(control)
    # reward = forward(env.state, agent.params, env, agent)
    _, reward, done, _ = env.step(env.state,control)
    # agent.feed(reward)
    # agent.update()
    # print("prev_state",prev_state)
    if (update_params==True):
        grads = f_grad(prev_state, agent.params, env, agent)
        # print("grads",grads)
        agent.params = [(w - agent.lr * dw, b - agent.lr * db)
                for (w, b), (dw, db) in zip(agent.params, grads)]
    print("agent.params",agent.params)

    # agent.W -= agent.lr * d_reward_d_w
    # print("agent.W",agent.W)
    return (env, agent), reward, done

# Deep
env = Cartpole_rbdl()
agent = Deep_Cartpole_rbdl(
             env_state_size = 4,
             action_space = jnp.array([0]),
             learning_rate = 0.1,
             gamma = 0.99,
             max_episode_length = 500,
             seed = 0
            )
# loaded_params = pickle.load( open( "examples/cartpole_rbdl_params_episode_20_2021-03-07 00:16:03.txt", "rb" ) )

# agent.params = loaded_params

 # for loop version
# xs = jnp.array(jnp.arange(T))
print(env.reset())
update_params = True
render = True
reward = 0
loss = 0
episode_loss = []
episodes_num = 1000
T = 200
for j in range(episodes_num):

    loss = 0
    env.reset()           
    print("episode:{%d}" % j)
    for i in range(T):
        (env, agent), r, done= loop((env, agent), 0)
        loss += r
        if done:
            print("end this episode because out of threshhold")
            env.past_reward = 0
            break
        # print("loss:",r)
        # loss.append(r)
        # reward += r
    episode_loss.append(loss)
    print("loss is %f and lasts for %d steps" % (loss,i))
    if (j%10==0 and j!=0 and update_params==True):
        with open("examples/cartpole_rbdl_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.params, fp)
# reward_forloop = reward
# print('reward_forloop = ' + str(reward_forloop))
plt.plot(episode_loss[1:])

#save plot and params
plt.savefig('cartpole_rbdl_loss'+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')

# fp.close()