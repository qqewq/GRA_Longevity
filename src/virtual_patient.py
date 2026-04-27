"""
Виртуальный пациент: модель старения на основе уравнений повреждений.
"""
import numpy as np

class VirtualPatient:
    def __init__(self, initial_bio_age=30, biomarker_dim=10):
        self.bio_age = initial_bio_age
        self.biomarkers = np.random.randn(biomarker_dim) * 0.1 + 1.0  # нормированные
        # Модель повреждений: dD/dt = a * D + b * возраст + шум
        self.damage_rate = 0.01
        self.noise_std = 0.05

    def step(self, dt=1.0, intervention=None):
        """
        Один год жизни. intervention — вектор коррекции из GRA (обнулятор).
        """
        # Естественное старение
        drift = self.damage_rate * self.biomarkers * dt
        noise = np.random.randn(len(self.biomarkers)) * self.noise_std * np.sqrt(dt)
        self.biomarkers += drift + noise
        # Эффект GRA-обнуления (если есть)
        if intervention is not None:
            self.biomarkers += intervention * dt
        self.bio_age += dt
        # Ограничение снизу (не уходим в отрицательные биомаркеры)
        self.biomarkers = np.clip(self.biomarkers, 0.1, 5.0)

    def get_state(self):
        return self.biomarkers.copy()

    def is_alive(self, mortality_threshold=3.0):
        """Смерть, если любой биомаркер превышает порог или падает ниже критического."""
        return not (np.any(self.biomarkers > mortality_threshold) or
                    np.any(self.biomarkers < 0.01))
