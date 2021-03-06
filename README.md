# Magics Differentiable Simulator

This simulator contains several simple robotic envs written in [JAX](https://github.com/google/jax) and with a Neural Network controller implemented.

## Envs
### Cart Pole
```
python examples/cartpole_NN.py
```
![](assets/cart_pole.gif)
Control the cartpole using manually calculated forward dynamics. Training converges at very fast speed(first two episodes). The cartpole can keep upright for 200 timesteps.

### Cart Pole RBDL
```
python examples/cartpole_NN_rbdl.py
```
For this demo, dynamics is implemented by jaxRBDL. Training time takes 2 hours for 100 episodes on MAC. Can't tell any convergence from loss graph. The cartpole can keep upright for 57 timesteps.

<!-- ### Rocket Landing
![](assets/rocket_landing.gif)
Control a rocket to landing. -->

<!-- ### Rigid Body
![](assets/rigid_body.png) -->

## QuickStart
```
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(PWD)
python examples/cartpole_NN.py
```