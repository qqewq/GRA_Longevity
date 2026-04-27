"""
Unit tests for GRA core functionality.
"""
import numpy as np
import pytest
from src.core import BioGRA

def test_static_foam():
    target = np.array([1.0, 1.0, 1.0])
    gra = BioGRA(target, att_type='static')
    state = np.array([1.1, 1.0, 0.9])
    foam = gra.foam(state)
    assert foam > 0

def test_obnulyator_reduces_foam():
    target = np.array([1.0, 1.0])
    gra = BioGRA(target, att_type='static')
    state = np.array([2.0, 2.0])
    foam_before = gra.foam(state)
    new_state = gra.obnulyator_step(state, lr=0.5)
    foam_after = gra.foam(new_state)
    assert foam_after < foam_before

def test_cyclic_foam():
    target = np.array([1.0, 0.0])
    gra = BioGRA(target, att_type='cyclic', cycle_period=24.0)
    state = np.array([0.5, 0.0])
    foam_t0 = gra.foam(state, t=0)
    foam_t6 = gra.foam(state, t=6)
    assert foam_t0 != foam_t6
