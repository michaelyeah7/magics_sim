import os
from metakernel.replwrap import python
from numpy import random
from numpy.core.records import array
from oct2py import octave
import numpy as np
from numpy.linalg import inv
import math
import unittest
from pyRBDL.Contact.CalcContactJacobian import CalcContactJacobian
from pyRBDL.Contact.CalcContactJdotQdot import CalcContactJdotQdot
from pyRBDL.Contact.CalcContactForcePD import CalcContactForcePD
from pyRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm
from pyRBDL.Dynamics.InverseDynamics import InverseDynamics
from pyRBDL.Contact.CalcContactForceDirect import CalcContactForceDirect
from pyRBDL.Contact.DetectContact import DetectContact

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OCTAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(CURRENT_PATH)), "octave")
MRBDL_PATH = os.path.join(OCTAVE_PATH, "mRBDL")
MATH_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Math")
MODEL_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Model")
TOOLS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Tools")
KINEMATICS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Kinematics")
DYNAMICS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Dynamics")
IPPHYSICIALPARAMS_PATH = os.path.join(OCTAVE_PATH, "ipPhysicalParams") 
IPTOOLS_PATH = os.path.join(OCTAVE_PATH, "ipTools")
CONTACT_PATH = os.path.join(MRBDL_PATH, "Contact")


octave.addpath(MRBDL_PATH)
octave.addpath(MATH_PATH)
octave.addpath(MODEL_PATH)
octave.addpath(TOOLS_PATH)
octave.addpath(KINEMATICS_PATH)
octave.addpath(DYNAMICS_PATH)
octave.addpath(IPPHYSICIALPARAMS_PATH)
octave.addpath(IPTOOLS_PATH)
octave.addpath(OCTAVE_PATH)
octave.addpath(CONTACT_PATH)


class TestContact(unittest.TestCase):
    def setUp(self):
        ip = dict()
        model = dict()
        octave.push("ip", ip)
        octave.push("model", model)
        self.ip = octave.ipParmsInit(0, 0, 0, 0)
        self.model = octave.model_create()
        self.model["NC"] = 2
        self.q = np.array([0.0,  0.4765, 0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3])
        self.qdot = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.qddot = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.tau = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def test_CalcContactJacobian(self):
        input = (self.model, self.q * np.random.randn(*(7, )), np.array([0, 1]), 2)
        oct_output = octave.CalcContactJacobian(*input)
        py_ouput = CalcContactJacobian(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_ouput)), 0.0, 14)
        input = (self.model, self.q * np.random.randn(*(7, )), np.array([1, 0]), 2)
        oct_output = octave.CalcContactJacobian(*input)
        py_ouput = CalcContactJacobian(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_ouput)), 0.0, 14)
        input = (self.model, self.q * np.random.randn(*(7, )), np.array([1, 1]), 2)
        oct_output = octave.CalcContactJacobian(*input)
        py_ouput = CalcContactJacobian(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_ouput)), 0.0, 14)
        input = (self.model, self.q * np.random.randn(*(7, )), np.array([0, 1]), 3)
        oct_output = octave.CalcContactJacobian(*input)
        py_ouput = CalcContactJacobian(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_ouput)), 0.0, 14)
        input = (self.model, self.q * np.random.randn(*(7, )), np.array([1, 0]), 3)
        oct_output = octave.CalcContactJacobian(*input)
        py_ouput = CalcContactJacobian(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_ouput)), 0.0, 14)
        input = (self.model, self.q * np.random.randn(*(7, )), np.array([1, 1]), 3)
        oct_output = octave.CalcContactJacobian(*input)
        py_ouput = CalcContactJacobian(*input)
        print(self.model["contactpoint"])
        print("output:",py_ouput.shape)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_ouput)), 0.0, 14)

    def test_CalcContactJdotQdot(self):
        input = (self.model, self.q * np.random.randn(*(7, )), self.qdot * np.random.randn(*(7,)), np.array([0, 1]), 2)
        oct_output = octave.CalcContactJdotQdot(*input)
        py_output = CalcContactJdotQdot(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 14)

        input = (self.model, self.q * np.random.randn(*(7, )), self.qdot * np.random.randn(*(7,)), np.array([1, 0]), 2)
        oct_output = octave.CalcContactJdotQdot(*input)
        py_output = CalcContactJdotQdot(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 14)

        input = (self.model, self.q * np.random.randn(*(7, )), self.qdot * np.random.randn(*(7,)), np.array([1, 1]), 2)
        oct_output = octave.CalcContactJdotQdot(*input)
        py_output = CalcContactJdotQdot(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 14)

        input = (self.model, self.q * np.random.randn(*(7, )), self.qdot * np.random.randn(*(7,)), np.array([0, 1]), 3)
        oct_output = octave.CalcContactJdotQdot(*input)
        py_output = CalcContactJdotQdot(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 14)


        input = (self.model, self.q * np.random.randn(*(7, )), self.qdot * np.random.randn(*(7,)), np.array([1, 0]), 3)
        oct_output = octave.CalcContactJdotQdot(*input)
        py_output = CalcContactJdotQdot(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 14)

        input = (self.model, self.q * np.random.randn(*(7, )), self.qdot * np.random.randn(*(7,)), np.array([1, 1]), 3)
        oct_output = octave.CalcContactJdotQdot(*input)
        py_output = CalcContactJdotQdot(*input)
        print(py_output.shape)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 14)

    def test_CalcContactForcePD(self):
        input = (self.model, self.q * np.random.randn(*(7, )), 
                 self.qdot * np.random.randn(*(7,)),
                 np.array([0, 0]),
                 np.array([10000, 10000, 10000]),
                 np.array([1000, 1000, 1000]), 2)
        oct_output = octave.CalcContactForcePD(*input)
        py_output = CalcContactForcePD(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 10)

        input = (self.model, self.q * np.random.randn(*(7, )), 
                 self.qdot * np.random.randn(*(7,)),
                 np.array([0, 1]),
                 np.array([10000, 10000, 10000]),
                 np.array([1000, 1000, 1000]), 2)
        oct_output = octave.CalcContactForcePD(*input)
        py_output = CalcContactForcePD(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 10)

        input = (self.model, self.q * np.random.randn(*(7, )), 
                 self.qdot * np.random.randn(*(7,)),
                 np.array([1, 0]),
                 np.array([10000, 10000, 10000]),
                 np.array([1000, 1000, 1000]), 2)
        oct_output = octave.CalcContactForcePD(*input)
        py_output = CalcContactForcePD(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 10)

        input = (self.model, self.q * np.random.randn(*(7, )), 
                 self.qdot * np.random.randn(*(7,)),
                 np.array([1, 1]),
                 np.array([10000, 10000, 10000]),
                 np.array([1000, 1000, 1000]), 2)
        oct_output = octave.CalcContactForcePD(*input)
        py_output = CalcContactForcePD(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 10)

        input = (self.model, self.q * np.random.randn(*(7, )), 
                 self.qdot * np.random.randn(*(7,)),
                 np.array([0, 0]),
                 np.array([10000, 10000, 10000]),
                 np.array([1000, 1000, 1000]), 3)
        oct_output = octave.CalcContactForcePD(*input)
        py_output = CalcContactForcePD(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 10)

        input = (self.model, self.q * np.random.randn(*(7, )), 
                 self.qdot * np.random.randn(*(7,)),
                 np.array([0, 1]),
                 np.array([10000, 10000, 10000]),
                 np.array([1000, 1000, 1000]), 3)
        oct_output = octave.CalcContactForcePD(*input)
        py_output = CalcContactForcePD(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 10)

        input = (self.model, self.q * np.random.randn(*(7, )), 
                 self.qdot * np.random.randn(*(7,)),
                 np.array([1, 0]),
                 np.array([10000, 10000, 10000]),
                 np.array([1000, 1000, 1000]), 3)
        oct_output = octave.CalcContactForcePD(*input)
        py_output = CalcContactForcePD(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 10)

        input = (self.model, self.q * np.random.randn(*(7, )), 
                 self.qdot * np.random.randn(*(7,)),
                 np.array([1, 1]),
                 np.array([10000, 10000, 10000]),
                 np.array([1000, 1000, 1000]), 3)
        oct_output = octave.CalcContactForcePD(*input)
        py_output = CalcContactForcePD(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_output-py_output)), 0.0, 10)
    
    def test_CalcContactForceDirect(self):
        flag_contact_list = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
        #TODO nf=3 
        nf_list = [2,] 
        for flag_contact in flag_contact_list:
            for nf in nf_list:
                q = self.q * random.randn(*(7,))
                qdot = self.qdot * random.rand(*(7,))
                model = self.model.copy()
                model["H"] = CompositeRigidBodyAlgorithm(model, q)
                model["C"] = InverseDynamics(model, q, qdot, np.zeros((int(model["NB"]), 1)))
                model["Hinv"] = inv(model["H"])
                tau = np.zeros((7, 1))
                input = (model, q, qdot, tau, flag_contact, nf)
                oct_output1, oct_output2 = octave.CalcContactForceDirect(*input, nout=2)
                py_output1, py_output2 = CalcContactForceDirect(*input)
                self.assertAlmostEqual(np.sum(np.abs(oct_output1-py_output1)), 0.0, 10)
                self.assertAlmostEqual(np.sum(np.abs(oct_output2-py_output2)), 0.0, 10)

    def test_DetectContact(self):
        for i in range(100):
            q = np.array([0,  0.1, 0, math.pi/3, math.pi/3, -2*math.pi/3, -2*math.pi/3]) 
            qdot = np.array([np.random.randn()* 0.05 ] * 7)
            contact_cond = dict()
            contact_cond["contact_pos_lb"] = np.array([0.0001, 0.0001, 0.0001])
            contact_cond["contact_pos_ub"] = np.array([0.0001, 0.0001, 0.0001])
            contact_cond["contact_vel_lb"] = np.array([-0.05, -0.05, -0.05])
            contact_cond["contact_vel_ub"] = np.array([0.01, 0.01, 0.01])
            input = (self.model, q, qdot, contact_cond)
            oct_output = octave.DetectContact(*input)
            py_output = DetectContact(*input)
            self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 14)















if __name__ == "__main__":
    unittest.main()