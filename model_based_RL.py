import jax
import jax.numpy as jnp
import copy
import gym
import numpy as np

class MBRL():
    def __init__(self, env, agent, lr = 1e-3, batch_size = 32):
        self.env = env
        self.agent = agent
        self.lr = lr
        self.batch_size = batch_size
        self.f_grad = jax.value_and_grad(self.roll_out,argnums=2)
        self.value_loss_grad = jax.value_and_grad(self.loss_value,argnums=3)
        self.model_loss_grad = jax.value_and_grad(self.loss_hybrid_model,argnums=3)

        self.trajectory_prev_state_buffer = jnp.zeros(env.state_size)
        self.trajectory_action_buffer = jnp.zeros(env.action_size)
        self.trajectory_reward_buffer = jnp.zeros(1)
        self.trajectory_next_state_buffer = jnp.zeros(env.state_size)

    def step(self, context, x):
        env, agent, params = context
        # env, agent, policy_params, rnn_params = context
        # control = agent(env.state, (policy_params, rnn_params))
        control = agent.sample_action(env.state, params)
        # control = control + 10 * np.random.uniform()
        prev_state = copy.deepcopy(env.state)
        _, reward, done, _ = env.step(env.state,control)
        # next_state, reward, done, _ = self.cartpole_gym_env(control)
        # env.state =  next_state

        return (env, agent), reward, done

    def roll_out(self, env, agent, params, horizon):
        policy_params, value_params =  params
        # policy_params, rnn_params = params
        # policy_params =  params
        gamma = 0.99
        total_return = 0.0
        for i in range(horizon):
        # for i in range(T):
            (env, agent), r, done= self.step((env, agent, policy_params), i)
            # (env, agent), r, done= self.step((env, agent, policy_params, rnn_params), i)
            # total_return = total_return * gamma + r 
            total_return = total_return + (gamma ** i)* r
            if done:
                # total_return -= 1000
                print("end this episode because out of threshhold in policy update")
                env.past_reward = 0
                break
        # total_return += agent.value(env.state,value_params) * gamma 
        total_return += agent.value(env.state,value_params) * gamma ** horizon
        # total_return += agent.value(env.state,value_params)
        losses = -total_return       
        return losses

    def loss_value(self, state, next_state, reward, value_params, agent):
        gamma = 0.99
        # td = reward + gamma * agent.value(next_state, value_params) - agent.value(state, value_params)
        td = reward + agent.value(next_state, value_params) - agent.value(state, value_params)
        value_loss = 0.5 * (td ** 2).mean()
        return value_loss

    def step_for_render(self, context, x):
        env, hybrid_env, agent, params = context
        # policy_params, rnn_params = params

        if(env.render_flag==True):
            # env.osim_render()
            env.render()
            # env.ddpg_render()
        control = agent.sample_action(env.state, params)
        # control = agent(env.state, (policy_params, rnn_params))
        # control = jnp.array([0.0])
        prev_state = copy.deepcopy(env.state)
        next_state, reward, done, _ = env.step(env.state,control)  
        
        # #update hybrid model
        # model_loss, model_grads = self.model_loss_grad(prev_state,control,next_state,hybrid_env.model_params, hybrid_env)
        # # print("model_loss",model_loss)
        # hybrid_env.model_losses.append(model_loss)
        # hybrid_env.model_params = agent.update(model_grads,hybrid_env.model_params,hybrid_env.model_lr)

        return (env, hybrid_env, agent), prev_state, control, reward, next_state, done

    def roll_out_for_render(self, env, hybrid_env, agent, params, T):
        # trajectory_state_buffer.append(env.state)
        gamma = 0.99
        rewards = 0.0
        for i in range(T):
            (env, hybrid_env, agent), prev_state, control, reward, next_state, done= self.step_for_render((env, hybrid_env, agent,params), i)
            # if (i == 0):
            #     trajectory_prev_state_buffer = prev_state
            #     trajectory_action_buffer = control
            #     trajectory_reward_buffer = reward
            #     trajectory_next_state_buffer = next_state
            # else:
            #     trajectory_prev_state_buffer = jnp.vstack((trajectory_prev_state_buffer,prev_state))
            #     trajectory_action_buffer = jnp.vstack((trajectory_action_buffer,control))
            #     trajectory_reward_buffer = jnp.vstack((trajectory_reward_buffer,reward))
            #     trajectory_next_state_buffer = jnp.vstack((trajectory_next_state_buffer,next_state))
            self.trajectory_prev_state_buffer = jnp.vstack((self.trajectory_prev_state_buffer,prev_state))
            self.trajectory_action_buffer = jnp.vstack((self.trajectory_action_buffer,control))
            self.trajectory_reward_buffer = jnp.vstack((self.trajectory_reward_buffer,reward))
            self.trajectory_next_state_buffer = jnp.vstack((self.trajectory_next_state_buffer,next_state))
                
            # rewards = rewards * gamma + r 
            rewards = rewards + reward
            # trajectory_state_buffer.append(env.state)
            if done:
                print("end this episode because out of threshhold in model update")
                env.past_reward = 0
                break
        
        # buffer =  [trajectory_prev_state_buffer, trajectory_action_buffer, trajectory_reward_buffer, trajectory_next_state_buffer]
        #update value function
        prev_state_batch, action_batch, reward_batch, next_state_batch = self.sample_batch()
        # value_loss, value_grads =  self.value_loss_grad(trajectory_prev_state_buffer,trajectory_next_state_buffer,trajectory_reward_buffer,agent.value_params, agent)
        value_loss, value_grads =  self.value_loss_grad(prev_state_batch, next_state_batch, reward_batch, agent.value_params, agent)
        agent.value_losses.append(value_loss)
        agent.value_params = self.update(value_grads,agent.value_params, self.lr)            
        return rewards, prev_state_batch

    def loss_hybrid_model(self, prev_state, control, true_next_state, model_params, hybrid_env):
        next_state = hybrid_env.forward(prev_state, control, model_params)
        model_loss = jnp.sum((next_state - true_next_state)**2)
        # model_loss = jnp.linalg.norm(next_state - true_next_state)
        # print("model loss",model_loss)
        # print("model_loss.value",model_loss[0])
        # model_losses.append(model_loss)
        return model_loss

    def update(self, grads, params, lr):
        """
        Description: update weights
        """
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
        # print("gradient total_norm_sqr",total_norm_sqr)
        gradient_clip = 0.2
        scale = min(
            1.0, gradient_clip / (total_norm_sqr**0.5 + 1e-4))

        params = [(w - lr * dw, b - lr * db)
                for (w, b), (dw, db) in zip(params, grads)]

        return params        

    def sample_batch(self):
        if self.trajectory_prev_state_buffer.shape[0] < self.batch_size:
            batch_size = self.trajectory_prev_state_buffer.shape[0] - 1
        else:
            batch_size = self.batch_size
        idxs = np.random.choice(np.arange(1,self.trajectory_prev_state_buffer.shape[0]), batch_size, replace=False)
        prev_state_batch = self.trajectory_prev_state_buffer[idxs,:]
        action_batch = self.trajectory_action_buffer[idxs,:]
        reward_batch = self.trajectory_reward_buffer[idxs,:]
        next_state_batch = self.trajectory_next_state_buffer[idxs,:]
        return prev_state_batch, action_batch, reward_batch, next_state_batch