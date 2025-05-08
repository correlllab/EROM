(define
    (problem RRR)
    (:domain pick-place-and-stack)
    
    (:objects 
        table
        redBlock
        grnBlock
        bluBlock
        P0
        P1
        P2
        P3
        P4
        P5
        P6
        P7
        P8
    )
    
    
    (:init
        (Base table)
    
        (Graspable redBlock)
        (Graspable grnBlock)
        (Graspable bluBlock)
        
        (Waypoint P0)
        (Waypoint P1)
        (Waypoint P2)
        
        (Waypoint P3)
        (Waypoint P4)
        (Waypoint P5)
        
        (Waypoint P6)
        (Waypoint P7)
        (Waypoint P8)
    
        (GraspObj redBlock P0)
        (GraspObj grnBlock P1)
        (GraspObj bluBlock P2)
        
        (PoseAbove P0 table)
        (PoseAbove P1 table)
        (PoseAbove P2 table)
        
        (PoseAbove P3 table)
        (PoseAbove P4 redBlock)
        (PoseAbove P5 grnBlock)
        
        (Free P3)
        (Free P4)
        (Free P5)
        
        (Free P6)
        (Free P7)
        (Free P8)
        
        (Supported redBlock table)
        (Supported grnBlock table)
        (Supported bluBlock table)
        
        (HandEmpty)
        (AtPose P3)
    )
    
    (:goal ( or
            ( and
                (GraspObj redBlock P3  ) 
                (Supported grnBlock redBlock ) 
                (Supported bluBlock grnBlock ) 
                (HandEmpty)
            )   
            ( and
                (GraspObj redBlock P3  ) 
                (Supported bluBlock redBlock ) 
                (Supported grnBlock bluBlock ) 
                (HandEmpty)
            )   
            ( and
                (GraspObj grnBlock P3  )
                (Supported redBlock grnBlock ) 
                (Supported bluBlock redBlock ) 
                (HandEmpty)
            )   
            ( and
                (GraspObj grnBlock P3  )
                (Supported bluBlock grnBlock ) 
                (Supported redBlock bluBlock ) 
                (HandEmpty)
            )   
            ( and
                (GraspObj bluBlock  P3  )
                (Supported redBlock bluBlock ) 
                (Supported grnBlock redBlock ) 
                (HandEmpty)
            )   
            ( and
                (GraspObj bluBlock P3  ) 
                (Supported grnBlock bluBlock ) 
                (Supported redBlock grnBlock ) 
                (HandEmpty)
            )   
        )
    )
)