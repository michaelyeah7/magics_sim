import jax
import copy

class MBAC():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.f_grad = jax.value_and_grad(self.roll_out,argnums=2)
        self.value_loss_grad = jax.value_and_grad(self.loss_value,argnums=3)

    def step(self, context, x):
        env, agent, params = context
        control = agent(env.state, params)
        prev_state = copy.deepcopy(env.state)
        _, reward, done, _ = env.step(env.state,control)

        return (env, agent), reward, done

    def roll_out(self, env, agent, params, T):
        policy_params, value_params =  params
        gamma = 0.9
        total_return = 0.0
        for i in range(5):
            (env, agent), r, done= self.step((env, agent,policy_params), i)
            total_return = total_return * gamma + r 
            if done:
                print("end this episode because out of threshhold in policy update")
                env.past_reward = 0
                break
        total_return += agent.value(env.state,value_params) * gamma 
        losses = -total_return       
        return losses

    def loss_value(self, state, next_state, reward, value_params, agent):
        td = reward + agent.value(next_state, value_params) - agent.value(state, value_params)
        value_loss = 0.5 * (td ** 2)
        return value_loss

    def step_for_render(self, context, x):
        env, hybrid_env, agent, params = context
        if(env.render==True):
            env.osim_render()
        control = agent(env.state, params)
        prev_state = copy.deepcopy(env.state)
        next_state, reward, done, _ = env.step(env.state,control)

        #update value function
        value_loss, value_grads =  self.value_loss_grad(prev_state,next_state,reward,agent.value_params, agent)
        agent.value_losses.append(value_loss)
        agent.value_params = agent.update(value_grads,agent.value_params,agent.lr)    
        
        # #update hybrid model
        # model_loss, model_grads = model_loss_grad(prev_state,control,next_state,hybrid_env.model_params)
        # # print("model_loss",model_loss)
        # hybrid_env.model_losses.append(model_loss)
        # hybrid_env.model_params = agent.update(model_grads,hybrid_env.model_params,hybrid_env.model_lr)

        return (env, hybrid_env, agent), reward, done

    def roll_out_for_render(self, env, hybrid_env, agent, params, T):
        gamma = 0.9
        rewards = 0.0
        for i in range(T):
            (env, hybrid_env, agent), r, done= self.step_for_render((env, hybrid_env, agent,params), i)
            rewards = rewards * gamma + r 
            if done:
                print("end this episode because out of threshhold in model update")
                env.past_reward = 0
                break
            
        return rewards