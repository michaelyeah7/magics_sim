# Magics Differentiable Simulator

This simulator contains several simple robotic envs written in [JAX](https://github.com/google/jax) and with a Neural Network controller implemented.

## Envs
### Cart Pole
![](assets/cart_pole.gif)
Control the cartpole in less than 1000 episodes.

### Rocket Landing
![](assets/rocket_landing.gif)
Control a rocket to landing.

### Rigid Body
![](assets/rigid_body.png)

## QuickStart
```
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(PWD)
python examples/cartpole_NN.py
```