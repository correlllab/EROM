(define (domain pick-place-and-stack)
  (:requirements :strips :negative-preconditions :equality)

  ;;;;;;;;;; PREDICATES ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:predicates

    ;;; Domains ;;;
    (Graspable ?label); Name of a real object we can grasp
    (Identifier ?id) ; ID of a real object we can grasp
    (Base ?label); Name of a support object we CANNOT grasp
    (Waypoint ?pose) ; Model of any object we can go to in the world, real or not
    
    ;;; Objects ;;;
    (GraspObj ?label ?pose ?id) ; The concept of a named object at a pose
    (PoseAbove ?pose ?label) ; The concept of a pose being supported by an object
  
    ;;; Object State ;;;
    (Free ?pose) ; This pose is free of objects, therefore we can place something here without collision
    (Supported ?labelUp ?labelDn ?idUp ?idDn) ; Is the "up" object on top of the "down" object?
    (Blocked ?id) ; This object cannot be lifted
    
    ;;; Robot State ;;;
    (HandEmpty)
    (Holding ?label)
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
    ;:parameters (?label ?pose ?prevSupport ?idUp ?idDn)
    :parameters (?label ?pose ?prevSupport ?idUp)
    :precondition (and
      ;; Domain ;;
      (Graspable ?label)
      (Waypoint ?pose)
      (Base ?prevSupport)
      (Identifier ?idUp)
      ;(Identifier ?idDn)
      ;; Identity ;;
      ;(not (= ?idUp ?idDn))
      ;; Object State ;;
      (GraspObj ?label ?pose ?idUp)
      ;(Supported ?label ?prevSupport ?idUp ?idDn)
      (PoseAbove ?pose ?prevSupport)
      (not (Blocked ?idUp))
      ;; Robot State ;;
      (HandEmpty)
    )
    :effect (and
      ;; Robot State ;;
      (Holding ?label)
      (not (HandEmpty))
      ;; Object State ;;
      ;(not (Supported ?label ?prevSupport ?idUp ?idDn))
    )
  )
  
  (:action unstack
    :parameters (?label ?pose ?prevSupport ?idUp ?idDn)
    :precondition (and
      ;; Domain ;;
      (Graspable ?label)
      (Waypoint ?pose)
      (Graspable ?prevSupport)
      (Identifier ?idUp)
      (Identifier ?idDn)
      ;; Identity ;;
      (not (= ?idUp ?idDn))
      ;; Object State ;;
      (GraspObj ?label ?pose ?idUp)
      (Supported ?label ?prevSupport ?idUp ?idDn)
      (PoseAbove ?pose ?prevSupport)
      (not (Blocked ?idUp))
      ;; Robot State ;;
      (HandEmpty)
    )
    :effect (and
      ;; Robot State ;;
      (Holding ?label)
      (not (HandEmpty))
      ;; Object State ;;
      (not (Supported ?label ?prevSupport ?idUp ?idDn))
      (not (Blocked ?idDn))
    )
  )
  
  (:action move_holding
      :parameters (?poseBgn ?poseEnd ?label ?id)
      :precondition (and 
        ;; Domain ;;
        (Graspable ?label)
        (Waypoint ?poseBgn)
        (Waypoint ?poseEnd)
        (Identifier ?id)
        ;; Robot State ;;
        (Holding ?label)
        (AtPose ?poseBgn)
        ;; Object State ;;
        (Free ?poseEnd)
        (GraspObj ?label ?poseBgn ?id)
      )
      :effect (and 
        ;; Robot State ;;
        (AtPose ?poseEnd)
        (not (AtPose ?poseBgn))
        ;; Object State ;;
        (GraspObj ?label ?poseEnd ?id)
        (not (GraspObj ?label ?poseBgn ?id))
        (Free ?poseBgn)
        (not (Free ?poseEnd))
      )
  )
  
  (:action place
      :parameters (?label ?pose ?support ?idUp)
      :precondition (and 
                      ;; Domain ;;
                      (Graspable ?label)
                      (Waypoint ?pose)
                      (Base ?support)
                      (Identifier ?idUp)
                      ;(Identifier ?idDn)
                      ;; Identity ;;
                      (not (= ?idUp ?idDn))
                      ;; Object State ;;
                      (GraspObj ?label ?pose ?idUp) 
                      (PoseAbove ?pose ?support)
                      ;; Robot State ;;
                      (Holding ?label)
                      )
      :effect (and 
                ;; Robot State ;;
                (HandEmpty)
                (not (Holding ?label))
                ; (Supported ?label ?support ?idUp 0)
              )
  )
  
  (:action stack
      :parameters (?labelUp ?poseUp ?labelDn ?idUp ?idDn)
      :precondition (and 
                      ;; Domain ;;
                      (Graspable ?labelUp)
                      (Waypoint ?poseUp)
                      (Graspable ?labelDn)
                      (Identifier ?idUp)
                      (Identifier ?idDn)
                      ;; Identity ;;
                      (not (= ?idUp ?idDn))
                      ;; Object State ;;
                      (GraspObj ?labelUp ?poseUp ?idUp) 
                      (not (Blocked ?idDn))
                      ;; Requirements ;;
                      (PoseAbove ?poseUp ?labelDn)
                      ;; Robot State ;;
                      (Holding ?labelUp)
                      )
      :effect (and 
                ;; Object State ;;
                (Supported ?labelUp ?labelDn ?idUp ?idDn)
                (Blocked ?idDn)
                ;; Robot State ;;
                (HandEmpty)
                (not (Holding ?labelUp))
              )
  )
  
)