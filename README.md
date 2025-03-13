# EROM
Entropy Ranked Object Memory

# `DEV_PLAN`

## Basic System
* `[Y]` Revive Bayesian Updates, 2025-01-21: Re-implemented!, NEEDS TESTING
    - `[Y]` Bayes Class, 2025-01-21: This seems intact, NEEDS TESTING
    - `[Y]` Update as part of the scan process, 2025-01-21: Added!, NEEDS TESTING
    - `[Y]` Use rays in the pose update!, 2025-01-21: Seems pretty hacky!, NEEDS TESTING
* `[Y]` Run planner with Bayesian Updates on multiple scans, 2025-01-22: TESTED! Stacks blocks and recognizes it
* `[Y]` Re-implement periodic scanning, 2025-02-20: **Pauseable** BTs!
    - `[Y]` Planner runs the BT with periodic pause, 2025-02-20: **Pauseable** BTs!
    - `[Y]` Test pause, 2025-02-20: **Pauseable** BTs!
    - `[Y]` Perceive during pause, 2025-02-20: **Pauseable** BTs!
* `[Y]` Evaluate the Memory Implementation
    - `[Y]` Is there inappropriate merging in the Bayesian Update?, 2025-02-20: *No*, **WARNING**: Z-snapping at the scan stage!
    - `[Y]` Is my implementation of Maximum Likelyhood masking potential problems?, 2025-02-20: *No problems so far*, 
    - `[Y]` Is there a simpler Maximum Likelyhood implementation?, 2025-02-20: *Maybe*, but I don't care right now

* `[>]` Baseline Experiments

* `[>]` Run planner with Bayesian Updates on periodic scans
    - `[>]` Do NOT erase memory between steps!
    - `[>]` Consider the occlusion of supporting blocks!
        * **WARNING**: "Geometry.py" is a **YAGNI** threat!
* `[ ]` Shortcut Experiments

## Memory Testing
* `[ ]` Inspect the segmentation logic
* `[ ]` Adjust `env_var("_OWL2_THRESH")`
* `[ ]` Adjust `env_var("_SEG_SCORE_THRESH")`
* `[ ]` Adjust `env_var("_SEG_MAX_HITS")`


## Stretch Goals
* `[ ]` Grasp Planning via RANSAC
* `[ ]` Camera planning WRT occlusion