# [L]imited [U]R5 [M]otion [P]lanner

# 2025-07-16, PROBLEMS

## Search is PAINFUL!
- [x] Begin the search with a Bird's Eye View!
- [x] TEST Bird's Eye Init Search!
## COLLISIONS!
### Level 0
- [ ] Check frames 4-6 against the base
### Level 1, [FCL](https://github.com/BerkeleyAutomation/python-fcl)
- [ ] Use a sphere to represent the Hand
- [ ] Use 2 triangles to represent the Table!
- [ ] Get Cylinders from Graphics Project



# Expansion Ideas
- [ ] **Grasp Planner**!
	- [x] Locate edges
		- --> Many spurious edges!
	- [x] Put edges in space
		- --> Many spurious edges!
	- [ ] Actually LOOK at the PCD, it might be usable now!
	- [ ] Plan Grasp
- [ ] [OMPL](https://ompl.kavrakilab.org/python.html) --> [Install](https://ompl.kavrakilab.org/core/installation.html)
- [ ] [Flexible Collision Library](https://github.com/BerkeleyAutomation/python-fcl)

# Supporting Basic Planning
- [x] Restore the Rays
	- [x] https://github.com/correlllab/EROM/blob/e51e4ecd82f19e8f097057df6fca64754ffbea54/Memory.py#L146
	- [x] https://github.com/correlllab/EROM/blob/e51e4ecd82f19e8f097057df6fca64754ffbea54/Memory.py#L357
	- --> This resulted in MANY false readings!
- [ ] HAX
	- [ ] Pose Repair should also straighten out stacked blocks?
	- [ ] Pose Repair should also quantize Z coord?

- [x] Idea: Don't let the determinizer consider NULL!
## Object Search
- [x] Get a list of where objects **should** be
- [x] Plan shots that cover supposed objects
	- [x] Need a shortcut for obscured objects
- [x] Rank shots

## (Limited) Object Search Ideas
- [x] SOMETIMES POSE REPAIR IS FUCKING ME OVER, SHOULD NOT HAPPEN WHEN THERE IS TROUBLE?
- [x] Idea: If we are coming up with nothing, Then lower OWLv2 Thresholds!
- [ ] Idea: Merge Pose Cheater with LUMP? --> Circular Import?
	- --> No, LUMP is already 1k lines
- [x] Re-rank shots and loop until all req'd objects are found
- [x] Need a shortcut for obscured objects
- [ ] Take shots one at a time
	- [ ] Take note of where info was gained
	- [ ] Take note of greatest confusion
- [ ] Plan shots that would reduce existing confusion and add them to the list

## Near Term Needs
- [x] Check if an effector pose collides with base
- [x] Check if an effector pose collides is too close to the base
- [x] Check if a task space path comes too close to the base
- [x] Add waypoints to a task space path that comes too close to the base
- [x] Move all sensory planning here <-- "Memory.py"
- [x] Test
- [x] Visualize + Test
	- [x] Viz perceptions shots w/ Starting blocks
	- [x] Viz stick links
	- [x] Viz corrected path
	- [x] Account for the effector 
		- --> 2025-05-30: Pretty sure the planner already takes care of this
- [x] Integrate into planner
- [x] Avoid AABB Obstacles + Margin
	- [x] START SIMPLE!
- [x] Penalize being close to the table



## Medium Term Needs
- [x] Adapt to easier IK
	- --> 2025-05-30: Don't appear to need this?
- [x] Viz 3D links
- [ ] Check for self-collisions
