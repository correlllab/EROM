# Plan Outline
- [x] Basic Baseline
	- --> 2025-05-13: 2 minutes!
- [ ] Baseline: Class confusion only <--- $W$  
	- [ ] Needs testing! <--- $W$  
- [ ] Baseline: Remove pose cheat
- [ ] Method w/ Shortcutting

# Repair Plan
- [x] Need to restore "Confirmation Squeeze" 
	- --> 2025-05-14: For some reason setting the gripper width does not work, Just open the gripper instead
- [x] Push an empty cheat frame when an action fails
- [x] Better shot planning
	- [x] Don't allow poses too close to the robot
- [ ] If the problem is insoluble --> Run search of environment with <--- $W$  
	- [ ] More views
	- [ ] Wider area
	- [ ] Accounting of Last Best Object Locations
		- [ ] Keep count of how many views were needed last time
		- [ ] Keep track of where objects were found last time --> More views there!
- TODO: 
	- [ ] Let the robot reset the scene --> Possible confusion?
		- [ ] Let the planner do this?
	- [ ] Log the iteration number as a message

# Information Theoretic Sensory Planning?
- [ ] **Can I incorporate Information Gain in any way?**
- [ ] Given each set of hypotheses, what are the probs that new readings will reduce ambiguity?
- [ ] What is the probability that something is obscured from this view?
- [ ] How to compute a better view?


# Memory w/ Increasing Confusion
- [x] Cheater
	- [x] Allow class confusion
	- [x] Retain Pose Cheat
	- [x] Needs testing!
- [x] If an action fails, then erase it from Beliefs
- [ ] Needs testing!

# KL-Divergence
- [x] Is it being tracked? 
	- --> 2025-05-13: YES!
- [x] Reinstate KLD tracking
	- --> This was already being done!
- [ ] Try to plot it
- What would it mean to track KLD by both Label and Pose?


