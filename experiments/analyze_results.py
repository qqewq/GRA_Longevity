"""
Analyze results from in silico longevity trials.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_trial_results(results_path):
    """Load and analyze trial results."""
    with open(results_path) as f:
        data = json.load(f)

    lifespans = [r['lifespan'] for r in data]
    print(f"Mean lifespan: {np.mean(lifespans):.1f} ± {np.std(lifespans):.1f}")
    print(f"Max lifespan: {np.max(lifespans):.1f}")

    return lifespans

if __name__ == "__main__":
    import sys
    analyze_trial_results(sys.argv[1])
