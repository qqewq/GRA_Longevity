# API Documentation

## BioGRA

Core GRA engine for biological systems.

### Constructor
- `target_state`: numpy array, target homeostatic state
- `att_type`: str, one of 'static', 'cyclic', 'chaotic'
- `cycle_period`: float, period for cyclic attractors
- `strange_params`: dict, parameters for chaotic attractors

### Methods
- `foam(state, t)`: compute current foam Φ
- `gradient(state, t)`: compute gradient of foam
- `obnulyator_step(state, lr, t)`: perform one nullification step

## BioMultiverse

Hierarchical biological multiverse.

### Constructor
- `structure`: dict, level definitions
- `goal_tree`: dict, path -> target specification

### Methods
- `total_foam(state_dict, t)`: weighted sum of foam across levels
- `obnulyator_all(state_dict, lr, t)`: apply nullification to all levels

## VirtualPatient

Aging model based on damage accumulation.

### Methods
- `step(dt, intervention)`: advance one time step
- `get_state()`: return current biomarkers
- `is_alive()`: check mortality conditions

## GRA_Geroprotector

RL agent applying GRA nullification.

### Methods
- `intervene(state_dict, t)`: return corrected states
- `foam_value(state_dict, t)`: evaluate total foam
