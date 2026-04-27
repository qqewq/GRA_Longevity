"""
Верификация: сравнение GRA-обнуления с известными геропротекторами.
"""
import numpy as np
from .in_silico_exp import run_trial
import yaml

def compare_with_rapamycin():
    """Рапамицин моделируется как снижение damage_rate на 20%."""
    with open('config/hybrid_longevity.yaml') as f:
        config = yaml.safe_load(f)
    # Контроль
    lifespan_ctrl, _ = run_trial(config, max_years=200)
    # Модифицируем damage_rate в пациенте (в коде это параметр)
    # Для простоты запустим с уменьшенным damage_rate
    config['damage_rate'] = 0.008
    lifespan_gbra, _ = run_trial(config, max_years=200)
    print(f"Control lifespan: {lifespan_ctrl:.1f}, GRA+Rapa: {lifespan_gbra:.1f}")
    return lifespan_gbra - lifespan_ctrl
