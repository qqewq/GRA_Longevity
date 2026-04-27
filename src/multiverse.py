"""
Биологический мультиверс: рекурсивные уровни GRA.
"""
import numpy as np
from .core import BioGRA

class BioMultiverse:
    """
    Иерархия: клетки (L0) → ткани (L1) → органы (L2) → системы (L3) → организм (L4).
    """
    def __init__(self, structure: dict, goal_tree: dict):
        self.structure = structure  # {level: {names:..., dims:...}}
        self.goal_tree = goal_tree  # путь -> целевой вектор и тип
        self.engines = {}
        self._build()

    def _build(self):
        for path, spec in self.goal_tree.items():
            target = np.array(spec['vector'])
            att = spec.get('type', 'static')
            cycle = spec.get('cycle_period', 24)
            strange = spec.get('strange_params', {})
            self.engines[path] = BioGRA(target, att, cycle, strange)

    def total_foam(self, state_dict: dict, t: float = 0.0) -> float:
        """Взвешенная сумма пены с затуханием α^level."""
        alpha = 0.8
        total = 0.0
        for path, s in state_dict.items():
            level = path.count('/')
            w = alpha ** level
            total += w * self.engines[path].foam(s, t)
        return total

    def obnulyator_all(self, state_dict: dict, lr=0.05, t=0.0) -> dict:
        new = {}
        for path, s in state_dict.items():
            new[path] = self.engines[path].obnulyator_step(s, lr, t)
        return new
