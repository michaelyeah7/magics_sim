from numbers import Real
import jax
import jax.numpy as jnp
import numpy as np
import functools
import numpy.random as npr

class Deep_Agent():

    def __init__(
        self,
        state_size,
        action_size,
    ):
        self.state_size = state_size
        self.action_size = action_size

        #init actor and critic
        param_scale = 0.1
        actor_layer_sizes = [self.state_size, 512, 128, 2 * self.action_size]
        self.params = self.init_random_params(param_scale, actor_layer_sizes)
        critic_layer_sizes = [self.state_size, 512, 128, 1]
        self.value_params = self.init_random_params(param_scale, critic_layer_sizes)
        # rnn_layer_sizes = [2 * self.state_size, 32, 32, self.state_size]
        # self.rnn_params = self.init_random_params(param_scale, rnn_layer_sizes)

        self.value_losses = []
        self.h_t = jnp.zeros(4)

    def init_random_params(self, scale, layer_sizes):
        rng=npr.RandomState(0)
        return [(scale * rng.randn(m, n), scale * rng.randn(n))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]        

    def sample_action(self, state, params):
        activations = state
        for w, b in params[:-1]:
            outputs = jnp.dot(activations, w) + b
            activations = jnp.tanh(outputs)
        final_w, final_b = params[-1]
        logits = jnp.dot(activations, final_w) + final_b
        mu, sigma = jnp.split(logits, 2)

        eps = np.random.randn(1)
        self.action =  mu + sigma * eps  
        self.state = state        
        return self.action

    def value(self, state, params):
        """
        estimate the value of state
        """
        activations = state
        for w, b in params[:-1]:
            outputs = jnp.dot(activations, w) + b
            activations = jnp.tanh(outputs)
        final_w, final_b = params[-1]
        logits = jnp.dot(activations, final_w) + final_b
        return logits[0]

    def rnn(self, state, params):
        """
        encode previous states
        """
        activations = state
        for w, b in params[:-1]:
            outputs = jnp.dot(activations, w) + b
            activations = jnp.tanh(outputs)
        final_w, final_b = params[-1]
        logits = jnp.dot(activations, final_w) + final_b
        return logits        
