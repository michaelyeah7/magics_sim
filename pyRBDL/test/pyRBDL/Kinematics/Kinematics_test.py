import os
from oct2py import octave
import numpy as np
import math
import unittest
from pyRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
from pyRBDL.Kinematics.CalcPointVelocity import CalcPointVelocity
from pyRBDL.Kinematics.CalcPointAcceleraion import CalcPointAcceleration
from pyRBDL.Kinematics.CalcPointJacobian import CalcPointJacobian
from pyRBDL.Kinematics.CalcPointJacobianDerivative import CalcPointJacobianDerivative
from pyRBDL.Kinematics.CalcWholeBodyCoM import CalcWholeBodyCoM
from pyRBDL.Kinematics.CalcPosVelPointToBase import CalcPosVelPointToBase


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OCTAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(CURRENT_PATH)), "octave")
MRBDL_PATH = os.path.join(OCTAVE_PATH, "mRBDL")
MATH_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Math")
MODEL_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Model")
TOOLS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Tools")
KINEMATICS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Kinematics")
IPPHYSICIALPARAMS_PATH = os.path.join(OCTAVE_PATH, "ipPhysicalParams") 
IPTOOLS_PATH = os.path.join(OCTAVE_PATH, "ipTools")


octave.addpath(MRBDL_PATH)
octave.addpath(MATH_PATH)
octave.addpath(MODEL_PATH)
octave.addpath(TOOLS_PATH)
octave.addpath(KINEMATICS_PATH)
octave.addpath(IPPHYSICIALPARAMS_PATH)
octave.addpath(IPTOOLS_PATH)
octave.addpath(OCTAVE_PATH)


class TestKinematics(unittest.TestCase):
    def setUp(self):
        ip = dict()
        model = dict()
        octave.push("ip", ip)
        octave.push("model", model)
        self.ip = octave.ipParmsInit(0, 0, 0, 0)
        self.model = octave.model_create()
        self.q = np.array([0.0,  0.4765, 0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3])
        self.qdot = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.qddot = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def test_CalcBodyToBaseCoordinates(self):
        for i in range(1, int(self.model['NB']) + 1):
            input = (self.model, self.q, i, np.random.rand(*(3,)))
            py_output = CalcBodyToBaseCoordinates(*input)
            oct_output = octave.CalcBodyToBaseCoordinates(*input)
            self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 15)
    
    def test_CalcPointVelocity(self):
        for i in range(1, int(self.model['NB']) + 1):
            input = (self.model, self.q, self.qdot, i, np.random.rand(*(3,)))
            py_output = CalcPointVelocity(*input)
            oct_output =  octave.CalcPointVelocity(*input)
            self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 14)

    def test_CalcPointAcceleration(self):
        for i in range(1, int(self.model['NB']) + 1):
            input = (self.model, self.q, self.qdot, self.qddot, i, np.random.rand(*(3,)))
            py_output = CalcPointAcceleration(*input)
            oct_output = octave.CalcPointAcceleration(*input)
            self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 14)

    def test_CalcPointJacobian(self):
        for i in range(1, int(self.model['NB']) + 1):
            input = (self.model, self.q, i, np.random.rand(*(3,)))
            oct_output = octave.CalcPointJacobian(*input)
            py_output = CalcPointJacobian(*input)
            self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 14)
        
    def test_CalcPointJacobianDerivative(self):
        for i in range(1, int(self.model['NB']) + 1):
            input = (self.model, self.q, self.qdot, i, np.random.rand(*(3,)))
            py_output = CalcPointJacobianDerivative(*input)
            oct_output = octave.CalcPointJacobianDerivative(*input)
            self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 14)

    def test_CalcWholeBodyCoM(self):
        input = (self.model, self.q * np.random.rand(*(7, )))
        oct_output = octave.CalcWholeBodyCoM(*input)
        py_output = CalcWholeBodyCoM(*input)
        self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 14)

    def test_CalcPosVelPointToBase(self):
        input = (self.model, self.q, self.qdot, 6, 3,  np.random.rand(*(3,)))
        oct_ouput = octave.CalcPosVelPointToBase(*input, nout=2)
        py_output = CalcPosVelPointToBase(*input)
        self.assertAlmostEqual(np.sum(np.abs(oct_ouput[0] - py_output[0])), 0.0, 14)
        self.assertAlmostEqual(np.sum(np.abs(oct_ouput[1] - py_output[1])), 0.0, 14)
        





if __name__ == "__main__":
    unittest.main()