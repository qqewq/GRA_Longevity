"""
GRA-геропротектор: RL-агент, применяющий обнуление к виртуальному пациенту.
"""
import numpy as np
from .core import BioGRA
from .multiverse import BioMultiverse

class GRA_Geroprotector:
    def __init__(self, multiverse: BioMultiverse, learning_rate=0.02):
        self.mv = multiverse
        self.lr = learning_rate

    def intervene(self, state_dict: dict, t: float) -> dict:
        """
        Возвращает скорректированные состояния после одного шага обнуления.
        """
        return self.mv.obnulyator_all(state_dict, self.lr, t)

    def foam_value(self, state_dict, t):
        return self.mv.total_foam(state_dict, t)
