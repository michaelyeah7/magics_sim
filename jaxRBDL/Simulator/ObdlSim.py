from jaxRBDL.Simulator.ObdlRender import ObdlRender
from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper
from pyRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm
from pyRBDL.Dynamics.ForwardDynamics import ForwardDynamics
from pyRBDL.Dynamics.InverseDynamics import InverseDynamics
from pyRBDL.Contact.CalcContactJacobian import CalcContactJacobian
from pyRBDL.Contact.CalcContactJdotQdot import CalcContactJdotQdot
import numpy as np
import math

from jaxRBDL.Simulator.SolverContact import solver_ode,dynamics_fun

class ObdlSim():
    def __init__(self,model,dt,vis=False):
        self.model = model
        #for pyjbdl
        self.model["jtype"] = np.asarray(self.model["jtype"])
        self.model["parent"] = np.asarray(self.model["parent"])

        #render
        self.visual = vis
        if self.visual:
            self.render = ObdlRender(model)
        self.dt = dt
        self.jnum = self.model['NB']

        #current state
        self.q = np.zeros((self.jnum,))
        self.qdot = np.zeros((self.jnum,))
        self.qddot = np.zeros((self.jnum,))
        self.tau = np.zeros((self.jnum,))

        self.debug_pts = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    def step_toruqe(self,tau,debug_line=False):
        """
        use pyrbdl to calcaulate next state and render
        """
        q = self.q.copy()
        qdot = self.qdot.copy()

        input = (self.model, q, qdot, tau)
        qddot_hat = ForwardDynamics(*input).flatten()
        q_hat,qdot_hat = self.calculate_q(dt=self.dt,q=q,qdot=qdot,qddot=qddot_hat)

        if(self.visual):
            self.render.step_render(q_hat)
        
        if(debug_line):
            self.draw_line()

        self.q = q_hat
        self.qdot =  np.zeros((self.jnum,))#qdot_hat #np.zeros((self.jnum,))#TODO qdot_hat or zeros
        self.qddot = np.zeros((self.jnum,))#qddot_hat #np.zeros((self.jnum,)) #TODO qddot_hat or zeros
        return


    def draw_line(self):
        q = self.q.copy()
        _model = self.model
        _cflag,_cpts = self.render.check_collision(_model['idcontact'])
        # _model['contactpoint'] = _cpts.copy()
        from pyRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
        idcontact = _model['idcontact']
        contactpoint = _model['contactpoint']
        NC = _model['NC']
        NB = _model['NB']
        for i in range(NC):
            # Calcualte pos and vel of foot endpoint, column vector
            if(_cflag[i]!=0):
                calc_id = idcontact[i] 
                local_pt = contactpoint[i]
                endpos_item = CalcBodyToBaseCoordinates(_model, q, calc_id,local_pt)
                print("id",calc_id,"q:",q,"contactpoint:",local_pt,"calc:",endpos_item)
                self.debug_pts[i].append(endpos_item)
        for pt in self.debug_pts:
            if(len(pt)>1):
                self.render.p.addUserDebugLine(pt[-2],pt[-1],lineColorRGB=[1.0,0.0,0.0],lineWidth=2.8)

    def calculate_q(self,dt,q,qdot,qddot):
        qdot = qdot + qddot * dt
        q = q + qdot * dt
        return q,qdot
    
    def step_theta(self,q):
        """
        TODO:we are suppoed to use inverse dynamics to make it move here 
        """
        if(self.visual):
            self.render.step_render(q)
        
        self.q = np.array(q)
    
    def calc_contact(self,qdot):
        """
        call the render to check the shape overlapping, which means collison happening
        cflag: 1 mean collision happends on the ground
        cpts: collision points
        """
        _cflag,_cpts = self.render.check_collision(self.model['idcontact'])
        _cflag = np.array(_cflag)
        self.model['contactpoint'] = _cpts.copy()

        input = (self.model, self.q , qdot , _cflag, 3) #TODO last argument
        _JdotDdot = CalcContactJdotQdot(*input)

        input = (self.model, self.q ,_cflag, 3)
        _Jacob = CalcContactJacobian(*input)

        _effect = _Jacob.T @ _JdotDdot # TODO odn't know why
        return  _effect.flatten()

        return  py_output
    
    def step_contact_v0(self,tau):
        _X = np.hstack((self.q,self.qdot))
        _model = self.model
        _model['tau'] = tau 
        _cflag,_cpts = self.render.check_collision(_model['idcontact'])
        _model['contactpoint'] = _cpts.copy()
        
        #forward dynamics
        T = self.dt
        input = (_X,_model,np.array(_cflag),T)
        xk, contact_force = solver_ode(*input)

        #calc state
        NB = int(_model["NB"])
        q_hat,qdot_hat = xk[0:NB],xk[NB:2*NB]

        #render
        if(self.visual):
            self.render.step_render(q_hat)
        self.q = np.array(q_hat)
        self.qdot =  np.array(qdot_hat) #np.zeros((self.jnum,))#TODO qdot_hat or zeros
        self.qddot = np.zeros((self.jnum,)) #TODO qddot_hat or zeros
        return

    def step_contact(self,tau):
        from jaxRBDL.Dynamics.StateFunODE import StateFunODE

        contact_cond = dict()
        contact_cond["contact_pos_lb"] = np.array([0.0001, 0.0001, 0.0001]).reshape(-1, 1)
        contact_cond["contact_pos_ub"] = np.array([0.0001, 0.0001, 0.0001]).reshape(-1, 1)
        contact_cond["contact_vel_lb"] = np.array([-0.05, -0.05, -0.05]).reshape(-1, 1)
        contact_cond["contact_vel_ub"] = np.array([0.01, 0.01, 0.01]).reshape(-1, 1)

        _X = np.hstack((self.q,self.qdot))
        _model = self.model
        _model['tau'] = tau 
        _model['ST'] = np.zeros((3,)) # useless
        # _cflag,_cpts = self.render.check_collision(_model['idcontact'])
        # _model['contactpoint'] = _cpts.copy()
        
        #forward dynamics
        T = self.dt
        input = (_model,_X, tau,T,contact_cond)       
        xk, contacts_force = StateFunODE(*input)


        #calc state
        NB = int(_model["NB"])
        q_hat,qdot_hat = xk[0:NB],xk[NB:2*NB]

        #render
        if(self.visual):
            self.render.step_render(q_hat)
        self.q = np.array(q_hat)
        self.qdot =  np.array(qdot_hat) #np.zeros((self.jnum,))#TODO qdot_hat or zeros
        self.qddot = np.zeros((self.jnum,)) #TODO qddot_hat or zeros
        return


    


if __name__ == "__main__":
    model = UrdfWrapper("/root/RBDL/urdf/arm.urdf").model
    osim = ObdlSim(model,dt=0.1,vis=True)

    import time
    while(True):    
        q = np.array([ 0.0, 0.0,0.0,  np.random.uniform(-math.pi/2,math.pi/2), np.random.uniform(-math.pi/2,math.pi/2), \
            np.random.uniform(-math.pi/2,math.pi/2),0.0])
        osim.step_theta(q)
        time.sleep(3)




    
