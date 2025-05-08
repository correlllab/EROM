(define
    (problem RRR)
    (:domain pick-place-and-stack)
    
    (:objects 
        I0
        I1
        I2
        I3
        P1
        P2
        P3
        P4
        P5
        P0
        redBlock
    )
    
    
    (:init
        (Base I0)
    
        (Graspable I1)
        (Graspable I2)
        (Graspable I3)
        
        (Type redBlock)
        
        (ObjLabel I1 redBlock)
        (ObjLabel I2 redBlock)
        (ObjLabel I3 redBlock)
        
        (Waypoint P0)
        (Waypoint P1)
        (Waypoint P2)
        
        (Waypoint P3)
        (Waypoint P4)
        (Waypoint P5)
    
        (GraspObj I1 P0)
        (GraspObj I2 P1)
        (GraspObj I3 P2)
        
        (PoseAbove P0 I0)
        (PoseAbove P1 I0)
        (PoseAbove P2 I0)
        
        (PoseAbove P3 I0)
        (PoseAbove P4 I1)
        (PoseAbove P5 I2)
        
        (Free P3)
        (Free P4)
        (Free P5)
        
        (Supported I1 I0)
        (Supported I2 I0)
        (Supported I3 I0)
        
        (HandEmpty)
        (AtPose P0)
    )
    
    (:goal (and
            (GraspObj  I1 P3)
            (Supported I2 I1)
            (Supported I3 I2)
            (HandEmpty)
            (ObjLabel I1 redBlock)
            (ObjLabel I2 redBlock)
            (ObjLabel I3 redBlock)
        )
    )
)