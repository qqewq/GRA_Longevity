import yaml
import argparse
from src.in_silico_exp import run_trial

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/hybrid_longevity.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    lifespan, foams = run_trial(cfg)
    print(f"Lifespan: {lifespan:.1f} years, Final foam: {foams[-1]:.4f}")
