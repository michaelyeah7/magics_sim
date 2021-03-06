import numpy as np
import json
from jaxRBDL.Utils.UrdfReader import URDF
from jaxRBDL.Utils.urdf_utils import transform_origin,rpy_to_matrix
from jaxRBDL.Math.SpatialTransform import SpatialTransform
from jaxRBDL.Model.RigidBodyInertia import RigidBodyInertia
import jax.numpy as jnp
import numpy as np

class UrdfWrapper(object):

    def __init__(self, filepath):
        self.load(filepath)

    @property
    def Xtree(self):
        Xtree = self._model["Xtree"]
        while type(Xtree[0]) is not np.ndarray:
            Xtree = Xtree[0]
        Xtree = [Xtree[i].copy() for i in range(len(Xtree))]
        return Xtree


    @property
    def I(self):
        I = self._model["I"]
        while type(I[0]) is not np.ndarray:
            I = I[0]
        I = [I[i].copy() for i in range(len(I))]
        return I

    @property
    def a_grav(self):
        a_grav = self._model['a_grav']
        if not isinstance(a_grav, np.ndarray):
            a_grav = np.asfarray(a_grav) 
        a_grav = a_grav.flatten().reshape(6, 1)
        return a_grav

    @property
    def parent(self):
        parent = self._model['parent']
        if isinstance(parent, np.ndarray):
            parent = self._model['parent'].flatten().astype(int).tolist()
        return parent

    @property
    def NB(self):
        NB = int(self._model["NB"])
        return NB

    @property
    def jtype(self):
        jtype = self._model["jtype"]
        if isinstance(jtype, np.ndarray):
            jtype = jtype.flatten().astype(int).tolist()
        return jtype

    @property
    def jaxis(self):
        jaxis = self._model['jaxis']
        return jaxis

    @property
    def model(self):
        model = dict()
        model["NB"] = self.NB
        model["a_grav"] = self.a_grav
        model["jtype"] = self.jtype
        model["jaxis"] = self.jaxis
        model["Xtree"] = self.Xtree
        model["I"] = self.I
        model["parent"] = self.parent
        model["jname"] = self.jname
        model["urdf_path"] = self.urdf_path
        return model

    @property
    def json(self):
        json_model = dict()
        for key, value in self.model.items():
            if isinstance(value, np.ndarray):
                json_model[key] = value.tolist()
            elif isinstance(value, list):
                json_list = []
                for elem in value:
                    if isinstance(elem, np.ndarray):
                        json_list.append(elem.tolist())
                    else:
                        json_list.append(elem)
                json_model[key] = json_list
            else:
                json_model[key] = value
        return json_model   

    @property
    def jname(self):
        jname = self._model['jname']
        return jname

    @property
    def urdf_path(self):
        urdf_path = self._model['urdf_path']
        return urdf_path

    def save(self, file_path: str):
        with open(file_path, 'w') as outfile:
            json.dump(self.json, outfile, indent=2)

    def load(self, file_path: str):
        urdf_model = load_urdf(file_path)
        self._model = urdf_model
        
    


def load_urdf(file_path):
    robot = URDF.load(file_path)#"/root/Downloads/urdf/arm.urdf"
    model  = dict()

    #NB
    NB = len(robot.joints)
    model["NB"] = NB

    #joint_name
    joint_name = []
    for i in range(NB):
        name = robot.joints[i].name
        joint_name.append(name)
    model['jname'] = joint_name

    #urdf
    model['urdf_path'] = file_path

    #grav
    a_grav = np.zeros((6,1))
    a_grav[5] = -9.81
    model["a_grav"] = a_grav

    #jtype
    joint_type = [0] * NB
    for i in range(NB):
        if(robot.joints[i].joint_type =="revolute"):
            joint_type[i] = 0
        elif(robot.joints[i].joint_type =="prismatic"):
            joint_type[i] = 1
        else:
            joint_type[i] = 1 #TODO need discuss
            # print("not known joint type",i)
    model['jtype'] = joint_type

    #joint_axis 
    joint_axis = ""
    for i in range(NB):
        axis_type = robot.joints[i].axis
        if(axis_type[0] == 1):
            joint_axis += 'x'
        elif(axis_type[1] ==1):
            joint_axis += 'y'
        elif(axis_type[2] == 1):
            joint_axis += 'z'
        else:
            joint_axis += 'x'
            # print("no known joint axis",i)
    model['jaxis'] = joint_axis# joint_axis#

    #parents
    parents = [0]*NB
    name_dict = {}
    for i in range(NB+1):
        _n = robot.links[i].name
        name_dict[_n] = i
    for i in range(NB):
        _p, _c = robot.joints[i].parent,robot.joints[i].child
        _pi, _ci = name_dict[_p],name_dict[_c]
        parents[_ci-1] = _pi  #TODO this is not suggested, but to sync with pyRBDL
    model['parent'] = parents

    #xtree and I
    X_tree, I = [],[]
    for i in range(NB):
        #joint info    
        xyz,rpy = transform_origin(robot.joints[i-1].origin)
        origin_matrix  = robot.joints[i-1].origin
        trans_matrix = origin_matrix[:3,3]#xyz
        rot_matrix = origin_matrix[:3,:3]#rpy_to_matrix(rpy)
        #link info
        link_mass = robot.links[i+1].inertial.mass
        link_intertia =  robot.links[i+1].inertial.inertia
        link_com = transform_origin(robot.links[i+1].inertial.origin)[0] # defaul rpy in link is equal to 0,0,0
        #tranform
        tree_element = SpatialTransform(jnp.asarray(rot_matrix),jnp.asarray(trans_matrix))
        I_element = RigidBodyInertia(link_mass,jnp.asarray(link_com),jnp.asarray(link_intertia))
        #build tree
        X_tree.append(np.asarray(tree_element))
        I.append(np.asarray(I_element))
    model['Xtree'] = X_tree
    model['I'] = I

    return model

# x = UrdfWrapper()
# x.load("/root/Downloads/urdf/arm.urdf")