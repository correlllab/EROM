

# Data Processing
- "Reader.py": Extract data from tests
- "Example.py": **UNKNOWN**
- "zu_Combined_Metrics.py": Actual - Extract data from recordings and make plots
- "zv_Extract_Examples.py": Actual - Data "Compression" (Removes ALL images, Leaves only poses)

## Data Visualization
- [ ] "zr_Inspect-Episode.ipynb": **FIND OUT**: Possibly Per-Episode Display of ALL state data of EVERY STEP ????
 
- "draw_plots.py": Graphing functions
- [ ] "yc_StateOrderDisp.ipynb": **RUN TO FIND OUT**


# Simulation
- "yb_SimpleSim_02.py": Current Simulation
- "analysis_Utils.py": Helper Classes + Helper Functions for the Simulation

- "ya_SimClasses.py": **NOT USED?**

# Architecture
- "env_config.py": Environment Variables
- "LUMP.py": Cheesy Motion Planning, Collision Avoidance
- "State.py":
    - Data Compression
    - Ground Truth Masks
    - Pose Cheater (Known Pose)
    - Pose Deserialization
    - Point Cloud Construction
    - Ground Truth Reconstruction
    - Count Missing / Hallucinated

- "utils.py": Helper Functions
