"""
Сценарии in silico экспериментов:
1. Контроль (без вмешательства)
2. Статическое GRA-обнуление (гомеостаз)
3. Полное гибридное GRA
"""
import numpy as np
from .virtual_patient import VirtualPatient
from .multiverse import BioMultiverse
from .geroprotector import GRA_Geroprotector

def run_trial(config, max_years=200):
    # Инициализация мультиверса из конфига
    mv = BioMultiverse(config['structure'], config['goal_tree'])
    protector = GRA_Geroprotector(mv, learning_rate=0.02)
    patient = VirtualPatient(initial_bio_age=30, biomarker_dim=10)
    lifespan = 0
    foam_history = []
    state_dict = {'0': patient.get_state()}  # упрощённо один уровень
    for year in range(max_years):
        t = year * 365.0  # дни
        state_dict = protector.intervene(state_dict, t)
        # Обновляем биомаркеры пациента с учётом вмешательства
        intervention = state_dict['0'] - patient.get_state()
        patient.step(dt=1.0, intervention=intervention)
        foam = mv.total_foam(state_dict, t)
        foam_history.append(foam)
        if not patient.is_alive():
            lifespan = patient.bio_age
            break
    else:
        lifespan = max_years
    return lifespan, foam_history
