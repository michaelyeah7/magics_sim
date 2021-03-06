from jaxRBDL.Simulator.ObdlRender import ObdlRender
from jaxRBDL.Utils.UrdfWrapper_guo import UrdfWrapper
from jaxRBDL.Utils.UrdfReader import URDF
import numpy as np

if __name__ == "__main__":

    model  = UrdfWrapper("urdf/cartpole.urdf").model
    model["jtype"] = np.asarray(model["jtype"])
    model["parent"] = np.asarray(model["parent"])

    rder = ObdlRender(model)
    # q = [0.1] * 7
    # q = np.array([ 0.0, 0.0,np.random.uniform(-math.pi/2,math.pi/2),  np.random.uniform(-math.pi/2,math.pi/2), np.random.uniform(-math.pi/2,math.pi/2), \
        # np.random.uniform(-math.pi/2,math.pi/2),0.0])
    # q = np.array([ 0.0, 0.0,0.0,  np.random.uniform(-math.pi/2,math.pi/2), np.random.uniform(-math.pi/2,math.pi/2), \
        # np.random.uniform(-math.pi/2,math.pi/2),0.0])
    # q = [ 0.0,0.0,0.5,1.10944034 ,-1.41440399, 1.55847655,0.]
    q = [0.0, 3,1.5]

    print("target q:",q)
    rder.step_render(q)
    poslist = rder.get_poslist()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax =  Axes3D(fig)
    ax.plot(np.asarray(poslist[:,0]), np.asarray(poslist[:,1]), np.asarray(poslist[:,2]), 'green',linewidth=7.5)
    plt.show()

    while(True):
        time.sleep(0.5)