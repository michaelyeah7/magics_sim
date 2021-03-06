from urdf_parser_py import urdf
from jaxRBDL.Math.SpatialTransform import SpatialTransform
import jax.numpy as jnp
import pybullet as p
import sys
import os


class Robot():
    def __init__(self):
        self.xml = '''<?xml version="1.0"?>
<robot name="test">
  <link name="base">
    <visual>
        <geometry>
           <cylinder length="0.6" radius="0.2"/>
        </geometry>
    </visual>
  </link>
  <link name="r_base_hhip_fore">
    <visual>
        <geometry>
           <cylinder length="0.6" radius="0.2"/>
        </geometry>
    </visual>
  </link>
  <joint name="base_to_hip" type="revolute">
     <parent link="base"/>
     <child link="r_base_hhip_fore"/>
     <origin rpy="0 0 0" xyz="0.2 0.0 0.0"/>
   </joint>
</robot>'''
        self.robot = self.parse(self.xml)
        print("robot",self.robot)
        # origin = robot.links[0].inertial.origin

    def parse(self, xml):
        return urdf.Robot.from_xml_string(xml)

    def getXtree(self):
        # Type of each joints, 0-Revolute, 1-Prismatic
        jtype = [1,1,0,0,0,0,0]
        jaxis = ['x', 'z', 'y', 'y', 'y', 'y', 'y']

        Xtree = []
        EIdentity = jnp.eye(3)
        r_zeros = jnp.zeros((3,1))
        length_body = 0.392
        length_hhip = 0.2115

        r_base_hhip_fore = jnp.array([length_body*0.5, 0.0, 0.0])
        r_base_hhip_hind = jnp.array([-length_body*0.5, 0.0, 0.0])
        r_hhip_knee_fore = jnp.array([0.0, 0.0, -length_hhip])
        r_hhip_knee_hind = jnp.array([0.0, 0.0, -length_hhip])

        Xtree_1 = SpatialTransform(EIdentity, r_zeros) # j0 -> j1
        Xtree_2 = SpatialTransform(EIdentity, r_zeros) # j1 -> j2
        Xtree_3 = SpatialTransform(EIdentity, r_zeros) # j2 -> j3
        Xtree_4 = SpatialTransform(EIdentity, r_base_hhip_fore) # j3 -> j4
        Xtree_5 = SpatialTransform(EIdentity, r_base_hhip_hind) # j3 -> j5
        Xtree_6 = SpatialTransform(EIdentity, r_hhip_knee_fore) # j4 -> j6
        Xtree_7 = SpatialTransform(EIdentity, r_hhip_knee_hind) # j5 -> j7
        Xtree.append(Xtree_1)
        Xtree.append(Xtree_2)
        Xtree.append(Xtree_3)
        Xtree.append(Xtree_4)
        Xtree.append(Xtree_5)
        Xtree.append(Xtree_6)
        Xtree.append(Xtree_7)

        return Xtree, jtype, jaxis
    
    def visualize(self):
        CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
        # sys.path.insert(0,CURRENT_PATH)
        p.connect(p.GUI)
        p.setAdditionalSearchPath(CURRENT_PATH) 
        # test_robot = p.loadURDF("/Users/xieguo/Projects/differentiable_engines/jaxRBDL/test/jaxRBDL/Dynamics/test.urdf",[0, 0, 0.9])
        test_robot = p.loadURDF("test.urdf",[0,0,1])
        # humanoid = p.loadURDF("/Users/xieguo/Projects/differentiable_engines/jaxRBDL/test/jaxRBDL/Dynamics/flame3.urdf")
        # humanoid = p.loadURDF("flame3.urdf")
        p.setRealTimeSimulation(1)
        while(1):
            p.getCameraImage(320,200)
        

if __name__ == '__main__':
    robot = Robot()
    # Xtree = robot.getXtree()
    # print("Xtree",Xtree)
    robot.visualize()
    