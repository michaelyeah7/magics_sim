# Magics Differentiable Simulator

<!-- ## Issue
#### Issue description
This issue might just be a simple one. But while implementing cartpole forward dynamics, I found that rbdl gives very different acceleration output than the two other manually calculated acceleration. On the other hand, I have confirmed jaxRBDL's output matches C++ RBDL's output. The robot is basically a cartpole containing a cart (mass=1.0) and a pole (mass=0.1, length=0.5). There is also a [urdf](urdf/cartpole.urdf) file for jaxRBDL.

#### Steps to reproduce the issue
1. install dependencies
```
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(PWD)
```
2. run following issue.py to reproduce the problem, it will show three different outputs. 
```
python issue.py
```

3. (Optional) If you like, you can run demo example, this example integrate a NN controller and a cartpole dynamics env. You can change the dynamics option in [dynamics env](envs/_cartpole_rbdl.py) init function. There are three options: "RBDL" "Original" "PDP". By running following example, the render will be launched to visualize.

```
python examples/cartpole_NN_rbdl.py
``` -->


## Description
This simulator contains several simple robotic envs written in [JAX](https://github.com/google/jax) and with a Neural Network controller implemented. A render and urdf parser also included to extend to more realistic robotic application.

## Envs
### Cart Pole

![](assets/cart_pole.gif)
Control the cartpole using manually calculated forward dynamics. Training converges at very fast speed(first two episodes). The cartpole can keep upright for 200 timesteps.

For this demo, dynamics is implemented by jaxRBDL. Training time takes 2 hours for 100 episodes on MAC. Can't tell any convergence from loss graph. The cartpole can keep upright for 57 timesteps.
```
python examples/cartpole_NN_rbdl.py
```

### 7-link Arm Robot

A 7 link arm robot contains 6 joints. The first base_link to arm_link_0 fixed joint will be interpreted as prismatic joint (rbdl index 1) by rbdl. The remaining 5 joints are revolute joints (rbdl index 0).
![](assets/arm_robot.gif)
```
python examples/arm_NN_rbdl.py
```
Issue: RBDL returns a very large acceleration given a small force and immediately reaches the threshold, so the episode ends at the first step.

### Quadropedal Robot
This is a quadrupedal robot(UNITREE) rendered using a pre-generated trajectory.
![](assets/quadrupedal.gif)
```
python examples/quadrupedal.py
```



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