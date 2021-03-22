import pybullet as p
import time
import pybullet_data
import pybullet_envs
import numpy as np
import math

class PybulletRender():
    def __init__(self,urdf_file,joint_names,grav=0.0,dt=0.1):
        p.connect(p.GUI)
        self.rb = p.loadURDF(urdf_file,[0.0,0.0,0.8])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # to load plane.urdf
        p.loadURDF("plane.urdf")
        
        self.jointIds = []
        self.grav = grav
        self.joint_dict = {}

        #get all joint info 
        self.paramIds = []
        self.paramsId_joints = []
        jointAngles = [0.0] * 8
        activeJoint=0
        for j in range (p.getNumJoints(self.rb)):
            p.changeDynamics(self.rb,j,linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.rb,j)
            jointId = info[0]
            jointName = info[1]
            jointType = info[2]
            self.joint_dict[jointName.decode("utf-8") ] = jointId
            if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE or jointType==p.JOINT_FIXED):
                activeJoint+=1
                self.paramsId_joints.append(jointId)
                self.paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"),-4,4,jointAngles[activeJoint]))
        
        #sync the joint order to be same as joint_names
        for jname in joint_names:
            if jname not in self.joint_dict:
                self.jointIds.append(-1)
            else:
                self.jointIds.append(self.joint_dict[jname])
        
        #set simulation parameter
        p.setTimeStep(dt)

    def step_vel(self,target_vel):
        # p.setRealTimeSimulation(1)
        p.setGravity(0,0,self.grav)
        for i in range(len(target_vel)):
            if(self.jointIds[i] == -1):
                continue
            p.setJointMotorControl2(self.rb,self.jointIds[i],p.VELOCITY_CONTROL,target_vel[i])
        p.stepSimulation()
        return
    
    def step_torque(self,target_torque):
        # p.setRealTimeSimulation(1)
        p.setGravity(0,0,self.grav)
        for i in range(len(target_torque)):
            if(self.jointIds[i] == -1):
                continue
            p.setJointMotorControl2(self.rb,self.jointIds[i],p.TORQUE_CONTROL, force=target_torque[i])
        p.stepSimulation()
        return

    def step(self,step_pos):
        # target_pos = step_pos[1:]
        p.setRealTimeSimulation(1)
        p.setGravity(0,0,self.grav)
        for i in range(len(step_pos)):
            if(self.jointIds[i] == -1):
                continue
            p.setJointMotorControl2(self.rb,self.jointIds[i],p.POSITION_CONTROL,step_pos[i], force=0.5)

    def step_debuger(self):
        while(1):
            p.getCameraImage(320,200)
            p.setGravity(0,0,0.0)

            for i in range(len(self.paramIds)):
                c = self.paramIds[i]
                targetPos = p.readUserDebugParameter(c)
                p.setJointMotorControl2(self.rb,self.paramsId_joints[i],p.POSITION_CONTROL,targetPos, force=140.)
        return

    def get_joints(self):
        info_dict = {
            'torque':[],
            'poses': [],
            'vels':[]
        }
        for j in self.jointIds:
            if(j==-1):
                continue
            info = p.getJointState(self.rb,j)
            info_dict['torque'].append(info[3])
            info_dict['poses'].append(info[0])
            info_dict['vels'].append(info[1])
        return info_dict


if __name__ == "__main__":

    jname = ['', 'arm_joint_0', 'arm_joint_1', 'arm_joint_2', 'arm_joint_3', 'arm_joint_4', 'arm_joint_5']

    a = PybulletRender("/root/RBDL/urdf/arm.urdf",jname,0.0)

    # q = np.array([0.0,  0.0, 0, math.pi/6, math.pi/6, -math.pi/3,0.0])
    q = np.array([0.0,  0.0, 0.0 , math.pi/6, math.pi/6, -math.pi/3,0.0])
    # q = np.array([0.0]*6)
    a.step(q)

    # a.step_debuger()

    while(True):
        # a.step_debuger()
        a.step(q)
        continue
