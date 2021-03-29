import pybullet as p
import pybullet_data
import numpy as np
from jaxRBDL.Utils.UrdfUtils import matrix_to_rpy
from pyRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
from pyRBDL.Kinematics.TransformToPosition import TransformToPosition
import time
import math
import os
import jax.numpy as jnp
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
        self.init_rpy = [0.0,0.0,0.0]
        return

    def assign_prop(self,shape_type,origin,size,parent_joint,link_id,rgba,scale=[1,1,1]):
        self.type = shape_type
        self.origin = np.asarray(origin) 
        self.shape = size # filename if mesh
        if(self.type != 'mesh'):
            self.shape = np.asarray(size) /2.0
        self.parent_joint = parent_joint
        self.link_id = link_id
        self.rgba = rgba
        self.scale = scale
        return
    
    def assign_id(self,b_id):
        self.body_id = b_id
        return
    
    def assign_pose(self,pos,q):
        self.position = pos
        self.quat = q
    
    def assign_name(self,name):
        self.link_name = name
    
    def assign_initQua(self,qua,rpy):
        """
        this qua is from urdf link rpy
        """
        self.init_qua = qua
        self.init_rpy = rpy

class ObdlRender():
    def __init__(self,model):
        self.urdf_path = model["urdf_path"]
        self.robot= URDF.load(model["urdf_path"]) 
        self.model=model
        self.p = p 

        #launch pybullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        self.plane_id = p.loadURDF("plane.urdf")
        self.p.setTimeStep(1e-9) # for collision detect

        #get mapping from model
        self.id_mapping = dict()

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
        self.NL = NL

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
        _model_lid = 1
        _renderID = 0
        current_q = [0.0]*self.model["NB"]
        # self.rpy = np.zeros((3,))
        self.rpys = np.zeros((NL,3))
        self.quas = np.zeros((NL,4))
        self.quas[:,-1] = 1.0
        for l in robot.links:
            visuals = l.visuals
            for v in visuals:
                _obj = RenderObject()
                _pid = parents[_lid] 
                if(v.geometry.box):
                    box_size = v.geometry.box.size
                    box_origin = v.origin[:3,3] #matrix to xyz
                    box_color = [0.0,1.0,1.0,1.0]
                    if(v.material):
                        box_color = v.material.color
                    _obj.assign_prop("box",box_origin,box_size,_pid,_lid,box_color) 
                elif(v.geometry.cylinder):
                    cylinder_radius,cylinder_length = v.geometry.cylinder.radius,v.geometry.cylinder.length
                    cylinder_origin = v.origin[:3,3] #matrix to xyz
                    cylinder_color = [0.0,1.0,1.0,1.0]
                    if(v.material):
                        cylinder_color = v.material.color
                    _obj.assign_prop("cylinder",cylinder_origin,[cylinder_radius,cylinder_length],_pid,_lid,cylinder_color)
                elif(v.geometry.sphere):
                    sphere_radius = v.geometry.sphere.radius
                    sphere_origin = v.origin[:3,3] #matrix to xyz
                    sphere_color = [0.0,1.0,1.0,1.0]
                    if(v.material):
                        sphere_color = v.material.color
                    _obj.assign_prop("sphere",sphere_origin,[sphere_radius],_pid,_lid,sphere_color)
                elif(v.geometry.mesh):
                    mesh_name = os.path.dirname(self.urdf_path) + '/' + v.geometry.mesh.filename
                    mesh_origin = v.origin[:3,3] #matrix to xyz
                    mesh_scale = v.geometry.mesh.scale
                    mesh_color = [0.0,1.0,1.0,1.0]  #doesn't matter   
                    _obj.assign_prop("mesh",mesh_origin,[mesh_name],_pid,_lid,mesh_color,mesh_scale)
                _p,_q = self.transform_pos(self.model,_obj,q=current_q)
                _obj.assign_pose(_p,_q)
                init_rpy = matrix_to_rpy(v.origin[:3,:3])
                init_qua = self.p.getQuaternionFromEuler(init_rpy)
                _obj.assign_initQua(init_qua,init_rpy)
                bId = self.create_visualshape(target_obj=_obj)
                _obj.assign_id(bId)
                _obj.assign_name(l.name)
                self.render_objects.append(_obj)
                self.id_mapping[_model_lid] = _renderID
                _renderID+=1
            _lid+=1
            _model_lid +=1

    
    def create_visualshape(self,target_obj):
        body_id = -1
        vis_id = -1
        if(target_obj.type == "box"):
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=target_obj.shape,rgbaColor=target_obj.rgba,visualFrameOrientation=target_obj.init_qua)
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=target_obj.shape)
        elif(target_obj.type == "cylinder"):
            vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=target_obj.shape[0],length=target_obj.shape[1], rgbaColor=target_obj.rgba,visualFrameOrientation=target_obj.init_qua)
            col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=target_obj.shape[0],length=target_obj.shape[1])
        elif(target_obj.type == "sphere"):
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=target_obj.shape[0],rgbaColor=target_obj.rgba,visualFrameOrientation=target_obj.init_qua)
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=target_obj.shape[0])
        elif(target_obj.type == "mesh"):
            vis_id = p.createVisualShape(p.GEOM_MESH, fileName=target_obj.shape[0],meshScale=target_obj.scale,visualFrameOrientation=target_obj.init_qua)
            col_id = p.createCollisionShape(p.GEOM_MESH, fileName=target_obj.shape[0],meshScale=target_obj.scale)
        body_id = p.createMultiBody(baseMass=0.01,  baseCollisionShapeIndex = col_id, baseVisualShapeIndex=vis_id, basePosition =target_obj.position,\
                                    baseOrientation = target_obj.quat)
        # body_id = p.createMultiBody(baseMass=0.01,  baseVisualShapeIndex=vis_id, basePosition =target_obj.position,baseOrientation = target_obj.quat)
        return body_id
    
    def step_render(self,targetQ):
        """
        render robot to the target joint angle
        """
        self.rpys = np.zeros((3,))
        # self.transform_rpy(self.model,targetQ)
        self.transform_qua(self.model,targetQ)
        n_obj = len(self.render_objects)
        for i in range(n_obj):
            if(self.render_objects[i].parent_joint == 0):
                continue #TODO need discuss
            pos,qua = self.transform_pos(self.model,self.render_objects[i],targetQ)
            self.render_objects[i].assign_pose(pos,qua)
        
        for _obj in self.render_objects:
            self.move_obj(_obj)
        return

    def transform_rpy(self,model,q):
        """
        transform the q to all rpy
        """
        self.rpys = np.zeros((self.NL,3))
        self.j_rpys = np.zeros((self.NL,3))
        self.counted = np.zeros((self.NL,))
        parent = np.array(self.model['parent'])
        # print(parent)
        #calc joint rpy
        for i in range(self.NL):
            _rpy = np.zeros((3,))
            _pid = parent[i] -1 
            _rpy = np.array(self.j_rpys[_pid])
            if(i == 0):
                if(model['jtype'][0] == 0):
                    if(model['jaxis'][0]=='x'):
                        _rpy[0] = q[0]
                    elif(model['jaxis'][0]=='y'):
                        _rpy[1] =  q[0]
                    elif(model['jaxis'][0]=='z'):
                        _rpy[2] =  q[0]
            else:
                if(model['jtype'][i] == 0):
                    if(model['jaxis'][i]=='x'):
                        _rpy[0] = self.j_rpys[_pid][0] + q[i]
                    elif(model['jaxis'][i]=='y'):
                        _rpy[1] = self.j_rpys[_pid][1] + q[i]
                    elif(model['jaxis'][i]=='z'):
                        _rpy[2] = self.j_rpys[_pid][2] + q[i]
                    elif(model['jaxis'][i]=='a'):
                        _rpy[0] = self.j_rpys[_pid][0] - q[i]
                    elif(model['jaxis'][i]=='b'):
                        _rpy[1] = self.j_rpys[_pid][1] - q[i]
                    elif(model['jaxis'][i]=='c'):
                        _rpy[2] = self.j_rpys[_pid][2] - q[i]
                # print("link",i,"parent",_pid,"angle",_rpy)
            self.j_rpys[i] = _rpy
        
        #calc link's rpy, which is qeqaul to parent joint rpy 
        self.rpys = self.j_rpys
        # print("type",model['jtype'],"current_q",q,"rpys",self.rpys)
        return

    def transform_qua(self,model,q):
        """
        transform the q to all rpy
        """
        zeros_pos = np.zeros((3,))
        self.quas = np.zeros((self.NL,4))
        self.quas[:,-1] = 1.0
        self.j_qua = np.zeros((self.NL,4))
        self.j_qua[:,-1] = 1.0
        parent = np.array(self.model['parent'])
        # print(parent)
        #calc joint rpy
        for i in range(self.NL):
            _pid = parent[i] -1 
            _qua = np.array([0.0,0.0,0.0,1.0])
            _rpy = np.array([0.0,0.0,0.0])
            if(i == 0):
                if(model['jtype'][0] == 0):
                    if(model['jaxis'][0]=='x'):
                        _rpy = [q[0],0.0,0.0]
                        _qua = self.p.getQuaternionFromEuler (_rpy)
                    elif(model['jaxis'][0]=='y'):
                        _rpy = [0.0,q[0],0.0]
                        _qua = self.p.getQuaternionFromEuler (_rpy)
                    elif(model['jaxis'][0]=='z'):
                        _rpy = [0.0,0.0,q[0]]
                        _qua = self.p.getQuaternionFromEuler (_rpy)
            else:
                if(model['jtype'][i] == 0):
                    if(model['jaxis'][i]=='x'):
                        _rpy = [q[i],0.0,0.0]
                    elif(model['jaxis'][i]=='y'):
                        _rpy = [0.0,q[i],0.0]
                    elif(model['jaxis'][i]=='z'):
                        _rpy = [0.0,0.0,q[i]]
                    elif(model['jaxis'][i]=='a'):
                        _rpy = [-q[i],0.0,0.0]
                    elif(model['jaxis'][i]=='b'):
                        _rpy = [0.0,-q[i],0.0]
                    elif(model['jaxis'][i]=='c'):
                        _rpy = [0.0,0.0,-q[i]]
                _qua = self.p.getQuaternionFromEuler (_rpy)
                _pQua = np.array(self.j_qua[_pid])
                _qua = self.p.multiplyTransforms(zeros_pos,_pQua,zeros_pos,_qua)[1]
            # print("link",i,"parent",_pid,"qua:",_qua)
            self.j_qua[i] = _qua
        #calc link's rpy, which is qeqaul to parent joint rpy
        # print(self.j_qua) 
        self.quas = self.j_qua
        return


    def move_obj(self,_obj):
        # p.resetBasePositionAndOrientation(_obj.body_id,_obj.position,(0.0,0.0,0.0,1.0))
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
        # rpy = np.array(self.rpys[_rid]) 
        # qua = p.getQuaternionFromEuler(rpy)
        qua = self.quas[_rid]

        pos = np.asarray(pos).flatten() 
        qua = np.asarray(qua).flatten()

        return pos,qua
    
    def transform_pos_2(self,model,obj,q):
        """
        obj: render object, calc rpy from rotation matrx
        q: current angle of all joints
        problem: matrix_to_rpy has several return 
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
    
    def check_collision(self,contact_ids):
        self.p.stepSimulation() # for collision  detect
        cflags = [0.0] * len(contact_ids)
        cpts = [np.zeros((3,))] * len(contact_ids)
        n_id = len(contact_ids)
        for i in range(n_id):
            _contact_id = contact_ids[i]
            _render_id = 0
            if(_contact_id in self.id_mapping.keys()):
                _render_id = self.id_mapping[_contact_id]
            else:
                continue
            _lid = self.render_objects[_render_id].link_id
            _info = self.p.getContactPoints(self.plane_id,_lid)
            if(len(_info)>0):
                cflags[i] = 2.0 #1.0
                cpts[i] = np.array(_info[0][6])#np.array([0.0,-0.30,0.0])#np.array(_info[0][6])TODO:important: local pos of leg endpoint in the last joint
        return cflags,cpts

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
