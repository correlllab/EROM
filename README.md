# EROM
Entropy Ranked Object Memory

# `DEV_PLAN`

## Basic System
* `[Y]` Revive Bayesian Updates, 2025-01-21: Re-implemented!, NEEDS TESTING
    - `[Y]` Bayes Class, 2025-01-21: This seems intact, NEEDS TESTING
    - `[Y]` Update as part of the scan process, 2025-01-21: Added!, NEEDS TESTING
    - `[Y]` Use rays in the pose update!, 2025-01-21: Seems pretty hacky!, NEEDS TESTING
* `[Y]` Run planner with Bayesian Updates on multiple scans, 2025-01-22: TESTED! Stacks blocks and recognizes it
* `[>]` Re-implement periodic scanning
    - `[>]` Try an interrupt pattern instead of a spaced pattern
        * `[Y]` New Behaviour: `Move_Arm_w_Pause`, 2025-01-22: Complete, NEEDS TESTING
        * `[>]` New Sequence: An interruptable motion that will pause for perception updates at timed intervals WITHOUT having to plot out waypoints in a complex way
    - `{C}` If (interrupt is infeasible or overly complex) then { Fall back on pre-planned waypoints! }
* `[ ]` Evaluate the Memory Implementation
    - `[ ]` Is there inappropriate merging in the Bayesian Update?
    - `[ ]` Is my implementation of Maximum Likelyhood masking potential problems?
    - `[ ]` Is there a simpler Maximum Likelyhood implementation?
* `[ ]` Run planner with Bayesian Updates on periodic scans
* `[ ]` Baseline Experiments
* `[ ]` Shortcut Experiments

## Stretch Goals
* `[ ]` Grasp Planning via RANSAC
* `[ ]` Camera planning WRT occlusion