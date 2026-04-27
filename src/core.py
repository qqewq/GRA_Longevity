"""
GRA-ядро для биологических систем: пена Φ, проекторы, градиентное обнуление.
"""
import numpy as np
from scipy.linalg import expm

class BioGRA:
    """
    GRA-движок для одного биологического домена (напр., клеточный гомеостаз).
    Пена Φ (foam) = Σ_{i≠j} |⟨ψ_i|P_G|ψ_j⟩|², где P_G — проектор цели.
    """
    def __init__(self, target_state: np.ndarray, att_type: str = 'static',
                 cycle_period: float = 24.0, strange_params: dict = None):
        self.target = target_state
        self.att_type = att_type
        self.cycle_period = cycle_period
        self.strange_params = strange_params or {}
        self._build_projector()

    def _build_projector(self):
        dim = len(self.target)
        if self.att_type == 'static':
            # P = I - |target><target| (если таргет нормирован)
            norm = np.linalg.norm(self.target)
            if norm > 1e-12:
                u = self.target / norm
                self.P = np.eye(dim) - np.outer(u, u)
            else:
                self.P = np.eye(dim)
        elif self.att_type == 'cyclic':
            # Для предельного цикла проектор на касательное пространство (упрощённо)
            self.P = np.eye(dim)  # будет переопределён в foam
        elif self.att_type == 'chaotic':
            # Проектор, сохраняющий странный аттрактор (статистический)
            self.P = np.eye(dim)
        else:
            self.P = np.eye(dim)

    def foam(self, state: np.ndarray, t: float = 0.0) -> float:
        """
        Вычисляет текущую пену Φ.
        - static: ||state - target||^2
        - cyclic: среднее отклонение от цикла за период
        - chaotic: дисперсия флуктуаций относительно аттрактора
        """
        if self.att_type == 'static':
            return np.sum((state - self.target)**2)
        elif self.att_type == 'cyclic':
            # Идеальная фаза цикла = sin(2π t / T) * target_ref
            phase = 2 * np.pi * t / self.cycle_period
            ref = self.target * np.sin(phase)
            return np.sum((state - ref)**2)
        elif self.att_type == 'chaotic':
            # Пена как отклонение от статистического центра странного аттрактора
            center = np.zeros_like(state)  # упрощённо
            return np.sum((state - center)**2)
        else:
            return np.sum((state - self.target)**2)

    def gradient(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        if self.att_type == 'static':
            return 2 * (state - self.target)
        elif self.att_type == 'cyclic':
            phase = 2 * np.pi * t / self.cycle_period
            ref = self.target * np.sin(phase)
            return 2 * (state - ref)
        return 2 * state  # хаос: градиент к центру

    def obnulyator_step(self, state, lr=0.1, t=0.0):
        return state - lr * self.gradient(state, t)
