# EROM
Entropy Ranked Object Memory

# `DEV_PLAN`

## Basic System
* `[Y]` Revive Bayesian Updates, 2025-01-21: Re-implemented!, NEEDS TESTING
    - `[Y]` Bayes Class, 2025-01-21: This seems intact, NEEDS TESTING
    - `[Y]` Update as part of the scan process, 2025-01-21: Added!, NEEDS TESTING
    - `[Y]` Use rays in the pose update!, 2025-01-21: Seems pretty hacky!, NEEDS TESTING
* `[ ]` Run planner with Bayesian Updates on multiple scans
* `[ ]` Re-implement periodic scanning
    - `[ ]` Try an interrupt pattern instead of a spaced pattern
* `[ ]` Run planner with Bayesian Updates on periodic scans
* `[ ]` Baseline Experiments
* `[ ]` Shortcut Experiments

## Stretch Goals
* `[ ]` Grasp Planning via RANSAC
* `[ ]` Camera planning WRT occlusion