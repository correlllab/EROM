(define (domain pick-place-and-stack)
  (:requirements :strips :negative-preconditions)

  ;;;;;;;;;; PREDICATES ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:predicates

    ;;; Domains ;;;
    (Graspable ?id); Name of a real object we can grasp
    (Base ?id); Name of a support object we CANNOT grasp
    (Waypoint ?pose) ; Model of any object we can go to in the world, real or not
    (Type ?label)
    
    ;;; Objects ;;;
    (GraspObj ?id ?pose) ; The concept of a named object at a pose
    (PoseAbove ?pose ?id) ; The concept of a pose being supported by an object
    (ObjLabel ?id ?label)
  
    ;;; Object State ;;;
    (Free ?pose) ; This pose is free of objects, therefore we can place something here without collision
    (Supported ?idUp ?idDn) ; Is the "up" object on top of the "down" object?
    (Blocked ?id) ; This object cannot be lifted
    
    ;;; Robot State ;;;
    (HandEmpty)
    (Holding ?id)
    (AtPose ?pose)

  )

  ;;;;;;;;;; ACTIONS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:action move_free
      :parameters (?poseBgn ?poseEnd)
      :precondition (and 
        ;; Robot State ;;
        (HandEmpty)
        (AtPose ?poseBgn)
      )
      :effect (and 
        ;; Robot State ;;
        (AtPose ?poseEnd)
        (not (AtPose ?poseBgn))
      )
  )
  
  (:action pick
    :parameters (?id ?pose ?prevSupportId)
    :precondition (and
      ;; Domain ;;
      (Graspable ?id)
      (Waypoint ?pose)
      (Base ?prevSupportId)
      ;; Object State ;;
      (GraspObj ?id ?pose)
      (Supported ?id ?prevSupportId)
      (PoseAbove ?pose ?prevSupportId)
      (not (Blocked ?id))
      ;; Robot State ;;
      (HandEmpty)
    )
    :effect (and
      ;; Robot State ;;
      (Holding ?id)
      (not (HandEmpty))
      ;; Object State ;;
      (not (Supported ?id ?prevSupportId))
    )
  )
  
  (:action unstack
    :parameters (?id ?pose ?prevSupportId)
    :precondition (and
      ;; Domain ;;
      (Graspable ?id)
      (Waypoint ?pose)
      (Graspable ?prevSupportId)
      ;; Object State ;;
      (GraspObj ?id ?pose)
      (Supported ?id ?prevSupportId)
      (PoseAbove ?pose ?prevSupportId)
      (not (Blocked ?id))
      ;; Robot State ;;
      (HandEmpty)
    )
    :effect (and
      ;; Robot State ;;
      (Holding ?id)
      (not (HandEmpty))
      ;; Object State ;;
      (not (Supported ?id ?prevSupportId))
      (not (Blocked ?prevSupportId))
    )
  )
  
  (:action move_holding
      :parameters (?poseBgn ?poseEnd ?id)
      :precondition (and 
        ;; Robot State ;;
        (Holding ?id)
        (AtPose ?poseBgn)
        ;; Object State ;;
        (Free ?poseEnd)
        (GraspObj ?id ?poseBgn)
      )
      :effect (and 
        ;; Robot State ;;
        (AtPose ?poseEnd)
        (not (AtPose ?poseBgn))
        ;; Object State ;;
        (GraspObj ?id ?poseEnd)
        (not (GraspObj ?id ?poseBgn))
        (Free ?poseBgn)
        (not (Free ?poseEnd))
      )
  )
  
  (:action place
      :parameters (?id ?pose ?support)
      :precondition (and 
                      ;; Domain ;;
                      (Graspable ?id)
                      (Waypoint ?pose)
                      (Base ?support)
                      ;; Object State ;;
                      (GraspObj ?id ?pose) 
                      (PoseAbove ?pose ?support)
                      ;; Robot State ;;
                      (Holding ?id)
                      )
      :effect (and 
                ;; Robot State ;;
                (HandEmpty)
                (not (Holding ?id))
                (Supported ?id ?support)
              )
  )
  
  (:action stack
      :parameters (?idUp ?poseUp ?idDn)
      :precondition (and 
                      ;; Domain ;;
                      (Graspable ?idUp)
                      (Waypoint ?poseUp)
                      (Graspable ?idDn)
                      ;; Object State ;;
                      (GraspObj ?idUp ?poseUp) 
                      (not (Blocked ?idDn))
                      ;; Requirements ;;
                      (PoseAbove ?poseUp ?idDn)
                      ;; Robot State ;;
                      (Holding ?idUp)
                      )
      :effect (and 
                ;; Object State ;;
                (Supported ?idUp ?idDn)
                (Blocked ?idDn)
                ;; Robot State ;;
                (HandEmpty)
                (not (Holding ?idUp))
              )
  )
  
)