# EROM
Entropy Ranked Object Memory

# `DEV_PLAN`

## ISSUE: POSES FROM THE CAMERA ARE <ins>BAD</ins>
* `[>]` Test alternate means to fetch camera intrinsics
* `[ ]` Characterize Problem
    - `[ ]` Segment ONE BLOCK from several different angles
    - `[ ]` Where are the UN-transformed point clouds?
    - `[ ]` Where are the TRANSFORMED point clouds?
* `[ ]` System Identification: Camera Transform
    - `[ ]` What is the data?
    - `[ ]` What is the error function?
    - `[ ]` Optimization inner loop
    - `[ ]` Optimization outer loop
    - `[ ]` Optimize transform



## Basic System
* `[Y]` Revive Bayesian Updates, 2025-01-21: Re-implemented!, NEEDS TESTING
    - `[Y]` Bayes Class, 2025-01-21: This seems intact, NEEDS TESTING
    - `[Y]` Update as part of the scan process, 2025-01-21: Added!, NEEDS TESTING
    - `[Y]` Use rays in the pose update!, 2025-01-21: Seems pretty hacky!, NEEDS TESTING
* `[Y]` Run planner with Bayesian Updates on multiple scans, 2025-01-22: TESTED! Stacks blocks and recognizes it

* `[>]` Re-implement periodic scanning
    - `[>]` Planner runs the BT with periodic pause
    - `[ ]` Test pause
    - `[ ]` Perceive during pause

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