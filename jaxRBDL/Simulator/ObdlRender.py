import pybullet as p
import pybullet_data
from pyRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
import numpy as np
import jax.numpy as jnp
from jaxRBDL.Utils.urdf_utils import matrix_to_rpy
from pyRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
from pyRBDL.Kinematics.TransformToPosition import TransformToPosition
import time
import math
from jaxRBDL.Utils.UrdfReader import URDF
from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper


class RenderObject():
    def __init__(self):
        self.type = ""
        self.origin = [0,0,0]
        self.shape = [0,0,0]
        self.parent_joint = -1
        self.link_id = -1 
        self.body_id = -1
        self.rgba = [0,0,0,0]
        self.position = [0.0,0.0,0.0]
        self.quat = [0,0,0,1]
        self.link_name = ""
        return

    def assign_prop(self,shape_type,origin,size,parent_joint,link_id,rgba):
        self.type = shape_type
        self.origin = np.asarray(origin) 
        self.shape = np.asarray(size) /2.0
        self.parent_joint = parent_joint
        self.link_id = link_id
        self.rgba = rgba
        return
    
    def assign_id(self,b_id):
        self.body_id = b_id
        return
    
    def assign_pose(self,pos,q):
        self.position = pos
        self.quat = q
    
    def assign_name(self,name):
        self.link_name = name

class ObdlRender():
    def __init__(self,model):
        self.robot= URDF.load(model["urdf_path"]) 
        self.model=model

        #paunch pybullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        p.loadURDF("plane.urdf")

        #render
        self.get_objects(self.robot)
        self.create_objects()

    def create_objects(self):

        return

    def get_objects(self,robot):
        """
        get render object shape and local position, parents jointsID
        render on the first time
        """
        self.render_objects = []
        NL = len(robot.links)
        NJ = len(robot.joints)

        #get parantes ID according to model
        joint_orders = dict()
        for i in range(len(self.model['jname'])):
            _jname = self.model['jname'][i]
            joint_orders[_jname] = i  #0 is the root joint, not existing here

        #get parents
        parents = [0] * NL
        name_dict = {}
        for i in range(NL):
            _n = robot.links[i].name
            name_dict[_n] = i
        for i in range(NJ):
            _p, _c = robot.joints[i].parent,robot.joints[i].child
            _pi, _ci = name_dict[_p],name_dict[_c]
            _jname = robot.joints[i].name
            parents[_ci] =  joint_orders[_jname]#TODO error may happen here 
            # parents[_ci] =  joint_orders[_jname]

        #get shape and local position
        _lid = 0
        current_q = [0.0]*self.model["NB"]
        self.rpy = np.zeros((3,))
        for l in robot.links:
            visuals = l.visuals
            for v in visuals:
                _obj = RenderObject()
                _pid = parents[_lid] 
                if(v.geometry.box):
                    box_size = v.geometry.box.size
                    box_origin = v.origin[:3,3] #matrix to xyz
                    box_color = v.material.color
                    _obj.assign_prop("box",box_origin,box_size,_pid,_lid,box_color) 
                elif(v.geometry.cylinder):
                    cylinder_radius,cylinder_length = v.geometry.cylinder.radius,v.geometry.cylinder.length
                    cylinder_origin = v.origin[:3,3] #matrix to xyz
                    cylinder_color = v.material.color
                    _obj.assign_prop("cylinder",cylinder_origin,[cylinder_radius,cylinder_length],_pid,_lid,cylinder_color)
                elif(v.geometry.sphere):
                    sphere_radius = v.geometry.sphere.radius
                    sphere_origin = v.origin[:3,3] #matrix to xyz
                    sphere_color = v.material.color
                    _obj.assign_prop("sphere",sphere_origin,[sphere_radius],_pid,_lid,cylinder_color)
                _p,_q = self.transform_pos(self.model,_obj,q=current_q)
                _obj.assign_pose(_p,_q)
                bId = self.create_visualshape(target_obj=_obj)
                _obj.assign_id(bId)
                _obj.assign_name(l.name)
                self.render_objects.append(_obj)
            _lid+=1

    
    def create_visualshape(self,target_obj):
        body_id = -1
        vis_id = -1
        if(target_obj.type == "box"):
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=target_obj.shape,rgbaColor=target_obj.rgba)
        elif(target_obj.type == "cylinder"):
            vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=target_obj.shape[0],length=target_obj.shape[1], rgbaColor=target_obj.rgba)
        elif(target_obj.type == "sphere"):
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=target_obj.shape[0],rgbaColor=target_obj.rgba)
        body_id = p.createMultiBody(baseMass=0,  baseVisualShapeIndex=vis_id, basePosition =target_obj.position,baseOrientation = target_obj.quat)
        return body_id
    
    def step_render(self,targetQ):
        """
        render robot to the target joint angle
        """
        self.rpy = np.zeros((3,))
        n_obj = len(self.render_objects)
        for i in range(n_obj):
            if(self.render_objects[i].parent_joint == 0):
                continue #TODO need discuss
            pos,qua = self.transform_pos(self.model,self.render_objects[i],targetQ)
            self.render_objects[i].assign_pose(pos,qua)
        
        for _obj in self.render_objects:
            self.move_obj(_obj)
        return

    def move_obj(self,_obj):
        p.resetBasePositionAndOrientation(_obj.body_id,_obj.position,_obj.quat)
        return

    def transform_pos(self,model,obj,q):
        """
        obj: render object,calc rpy from q
        q: current angle of all joints
        """
        pos,qua = None,None
        q = np.asarray(q)

        local_pos = np.asarray(obj.origin).flatten() 
        _jid = obj.parent_joint +1 #TODO need discuss
        input = (model, q, _jid, local_pos)
        pos = CalcBodyToBaseCoordinates(*input)
        
        _rid = obj.parent_joint
        if(model['jtype'][_rid] == 0):
            if(model['jaxis'][_rid]=='x'):
                self.rpy[0] += q[_rid]
            elif(model['jaxis'][_rid]=='y'):
                self.rpy[1] += q[_rid]
            elif(model['jaxis'][_rid]=='z'):
                self.rpy[2] += q[_rid]
        rpy = np.array(self.rpy) #TODO nedd discuss
        qua = p.getQuaternionFromEuler(rpy)

        pos = np.asarray(pos).flatten() 
        qua = np.asarray(qua).flatten()

        # print("link",obj.link_name,"parent joint",_jid, "pos",pos,"rpy",rpy)

        return pos,qua
        # return [1.0,1.0,1.0],[0,0,0,1]
    
    def transform_pos_2(self,model,obj,q):
        """
        obj: render object, calc rpy from rotation matrx
        q: current angle of all joints
        """
        pos,qua = None,None
        q = np.asarray(q)

        local_pos = np.asarray(obj.origin).flatten() 
        _jid = obj.parent_joint +1 #TODO need discuss
        input = (model, q, _jid, local_pos)
        spatial_pos = CalcSpatialBodyToBaseCoordinates(*input) # this func use CalcBodyToBaseCoordinates but return X0_point(6x6)
        pos = TransformToPosition(spatial_pos)
        rot_mat = spatial_pos[0:3,0:3]
        rpy = -1 * matrix_to_rpy(rot_mat,solution=1)# don't konw why -1
        qua = p.getQuaternionFromEuler(rpy)

        pos = np.asarray(pos).flatten() 
        qua = np.asarray(qua).flatten()

        # print("link",obj.link_name,"parent joint",_jid, "pos",pos,"rpy",rpy)

        return pos,qua
    
    def get_poslist(self):
        _res = []
        for _obj in self.render_objects:
            _res.append(_obj.position)
        n = len(_res)
        _res = np.reshape(np.asarray(_res),(n,3))
        return _res

if __name__ == "__main__":

    model  = UrdfWrapper("/root/RBDL/urdf/arm.urdf").model
    model["jtype"] = np.asarray(model["jtype"])
    model["parent"] = np.asarray(model["parent"])

    from jaxRBDL.Utils.UrdfReader import URDF
    rder = ObdlRender(model)
    # q = [0.1] * 7
    # q = np.array([ 0.0, 0.0,np.random.uniform(-math.pi/2,math.pi/2),  np.random.uniform(-math.pi/2,math.pi/2), np.random.uniform(-math.pi/2,math.pi/2), \
        # np.random.uniform(-math.pi/2,math.pi/2),0.0])
    # q = np.array([ 0.0, 0.0,0.0,  np.random.uniform(-math.pi/2,math.pi/2), np.random.uniform(-math.pi/2,math.pi/2), \
        # np.random.uniform(-math.pi/2,math.pi/2),0.0])
    q = [ 0.0,0.0,0.5,1.10944034 ,-1.41440399, 1.55847655,0.]

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