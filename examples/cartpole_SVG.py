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

def roll_out(env, agent, params):
    gamma = 0.9
    losses = 0.0
    for i in range(100):
        (env, agent), r, done= loop((env, agent,params), i)
        losses = losses * gamma + r 
        if done:
            print("end this episode because out of threshhold in policy update")
            env.past_reward = 0
            break
        
    return losses

f_grad = jax.grad(roll_out,argnums=2)


def loss_hybrid_model(prev_state, control, true_next_state, model_params):
    next_state = hybrid_env.forward(prev_state, control, model_params)
    model_loss = jnp.sum((next_state - true_next_state)**2)
    return model_loss
model_loss_grad = jax.grad(loss_hybrid_model,argnums=3)

def loop_for_render(context, x):
    env, agent, params = context
    if(render==True):
        env.osim_render()
    control = agent(env.state, params)
    prev_state = copy.deepcopy(env.state)
    next_state, reward, done, _ = env.step(env.state,control)

    #update hybrid model
    model_grads = model_loss_grad(prev_state,control,next_state,hybrid_env.model_params)
    w, b = hybrid_env.model_params
    dw, db = model_grads
    hybrid_env.model_params = [w - hybrid_env.model_lr * dw, b - hybrid_env.model_lr * db]


    return (env, agent), reward, done

def roll_out_for_render(env, agent, params):
    gamma = 0.9
    losses = 0.0
    for i in range(100):
        (env, agent), r, done= loop_for_render((env, agent,params), i)
        losses = losses * gamma + r 
        if done:
            print("end this episode because out of threshhold in model update")
            env.past_reward = 0
            break
        
    return losses



# Deep
env = Cartpole_rbdl() 
hybrid_env = Cartpole_Hybrid(model_lr=1e-2)
agent = Deep_Cartpole_rbdl(
             env_state_size = 4,
             action_space = jnp.array([0]),
             learning_rate = 0.1,
             gamma = 0.99,
             max_episode_length = 500,
             seed = 0
            )

# load_params = True
# update_params = False
# render = True

load_params = False
update_params = True
render = False

if load_params == True:
    loaded_params = pickle.load( open( "examples/arm_rbdl_params_episode_20_2021-03-20 18:19:28.txt", "rb" ) )
    agent.params = loaded_params

 # for loop version
# xs = jnp.array(jnp.arange(T))
print(env.reset())
reward = 0
loss = 0
episode_loss = []
episodes_num = 1000
T = 200
for j in range(episodes_num):

    loss = 0
    env.reset()           
    print("episode:{%d}" % j)

    #update hybrid model using real trajectories
    loss = roll_out_for_render(env, agent, agent.params)

    #update the parameter
    if (update_params==True):
        hybrid_env.reset() 
        # grads = f_grad(prev_state, agent.params, env, agent)

        #train agent using learned hybrid env
        grads = f_grad(hybrid_env, agent, agent.params)
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
        
        # agent.params = [(w - agent.lr * dw, b - agent.lr * db)
        #         for (w, b), (dw, db) in zip(agent.params, grads)]

    episode_loss.append(loss)
    print("loss is %f" % loss)
    if (j%100==0 and j!=0 and update_params==True):
        with open("examples/cartpole_svg_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.params, fp)
# reward_forloop = reward
# print('reward_forloop = ' + str(reward_forloop))
plt.plot(episode_loss[1:])

#save plot and params
plt.savefig('cartpole_svg_loss'+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')

# fp.close()