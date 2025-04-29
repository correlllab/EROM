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
    - `[>]` Get stacking the blocks down to 4 minutes or less
        * `[>]` Remove one or more camera views, Removed 1, Needs TESTING
        * `[>]` Increase robot speed, Increased, Needs TESTING
        * `[P]` Shorten robot motions, 2025-04-29: "_Z_SAFE" already seems pretty near the stacked tower of 3 
    - `[ ]` Collect data on 100 experiments!

## Responsive System

* `[ ]` Run planner with Bayesian Updates on periodic scans
    - `[ ]` Do NOT erase memory between steps!
    - `[ ]` Make sure that moved blocks are reflected in the beliefs!
    

* `[ ]` Shortcut Experiments

## Memory Testing
* `[>]` Inspect the segmentation logic, 2025-04-29: Works very well!, even tho it still emits overlapping readings
    - `[Y]` TEST the segmentation logic!, 2025-04-29: Works very well!, even tho it still emits overlapping readings
    - `{N}` IF bad, THEN revert!, 2025-04-29: Not needed
* `[P]` Adjust `env_var("_OWL2_THRESH")`
* `[P]` Adjust `env_var("_SEG_SCORE_THRESH")`
* `[P]` Adjust `env_var("_SEG_MAX_HITS")`


## Stretch Goals
* `[ ]` Grasp Planning via RANSAC
* `[ ]` Camera planning WRT occlusion
- `[?]` Consider the occlusion of supporting blocks!
    * Why did I ask for this?
    * **WARNING**: "Geometry.py" is a **YAGNI** threat!