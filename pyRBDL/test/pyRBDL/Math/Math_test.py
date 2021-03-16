from typing import Mapping
import unittest
import numpy as np
import math
from pyRBDL.Math.CrossMatrix import CrossMatrix
from pyRBDL.Math.CrossMotionSpace import CrossMotionSpace
from pyRBDL.Math.CrossForceSpace import CrossForceSpace
from pyRBDL.Math.InverseMotionSpace import InverseMotionSpace
from pyRBDL.Math.SpatialTransform import SpatialTransform
from pyRBDL.Math.Xrotx import Xrotx
from pyRBDL.Math.Xroty import Xroty
from pyRBDL.Math.Xrotz import Xrotz
from pyRBDL.Math.Xtrans import Xtrans
from oct2py import octave
import os


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OCTAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(CURRENT_PATH)), "octave")
MATH_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Math")
octave.addpath(MATH_PATH)


class TestMath(unittest.TestCase):
    def test_CrossMatrix(self):
        input = np.random.rand(*(3, 1))
        py_output = CrossMatrix(input)
        oct_output = octave.CrossMatrix(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.rand(*(1, 3))
        py_output = CrossMatrix(input)
        oct_output = octave.CrossMatrix(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.rand(*(3,))
        py_output = CrossMatrix(input)
        oct_output = octave.CrossMatrix(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

    def test_CrossMotionSpace(self):
        input = np.random.rand(*(6, 1))
        py_output = CrossMotionSpace(input)
        oct_output = octave.CrossMotionSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.rand(*(1, 6))
        py_output = CrossMotionSpace(input)
        oct_output = octave.CrossMotionSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.rand(*(6,))
        py_output = CrossMotionSpace(input)
        oct_output = octave.CrossMotionSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

    def test_CrossForceSpace(self):
        input = np.random.rand(*(6, 1))
        py_output = CrossForceSpace(input)
        oct_output = octave.CrossForceSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.rand(*(1, 6))
        py_output = CrossForceSpace(input)
        oct_output = octave.CrossForceSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.rand(*(6,))
        py_output = CrossForceSpace(input)
        oct_output = octave.CrossForceSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)


    def test_Xrot(self):
        input = np.random.randn() * math.pi
        py_output = Xrotx(input)
        oct_output = octave.Xrotx(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        py_output = Xroty(input)
        oct_output = octave.Xroty(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        py_output = Xrotz(input)
        oct_output = octave.Xrotz(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

    def test_Xtrans(self):
        input = np.random.randn(*(3, 1))
        py_output = Xtrans(input)
        oct_output = octave.Xtrans(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.randn(*(1, 3))
        py_output = Xtrans(input)
        oct_output = octave.Xtrans(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.randn(*(3,))
        py_output = Xtrans(input)
        oct_output = octave.Xtrans(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

    def test_SpatialTransform(self):
        input = (np.random.randn(*(3,3)), np.random.randn(*(3, 1)))
        py_output = SpatialTransform(*input)
        oct_output = octave.SpatialTransform(*input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertAlmostEqual(residual, 0.0, 14)

        input = (np.random.randn(*(3,3)), np.random.randn(*(1, 3)))
        py_output = SpatialTransform(*input)
        oct_output = octave.SpatialTransform(*input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertAlmostEqual(residual, 0.0, 14)

        input = (np.random.randn(*(3,3)), np.random.randn(*(3,)))
        py_output = SpatialTransform(*input)
        oct_output = octave.SpatialTransform(*input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertAlmostEqual(residual, 0.0, 14)

    def test_InverseMotionSpace(self):
        input = np.random.randn(*(6, 6))
        py_output = InverseMotionSpace(input)
        oct_output = octave.InverseMotionSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)
        

if __name__ == "__main__":
    unittest.main()
