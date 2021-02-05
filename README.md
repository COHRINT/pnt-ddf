# pnt-ddf

## Demo 1 TODO
- [X] New dockerfile and associated setup
- [X] Install multi_jackal
- [X] python pip requirements
- [X] Able to run pntddf main in docker
- [X] Able to launch code from roslaunch
- [X] Change clocks to be based on rospy time
- [ ] Fix get true pos/vel methods to subscribe to true ROS states
- [ ] Change transmission scheduling to use ROS rate
- [ ] Create ROS msg to facilitate packets
- [ ] Each agent has instance of env object
- [ ] Update navigation code
- [ ] Accept jackal odom diff as control input
- [ ] bag data

## Later TODO
- [ ] Fix centralized filter


## Notes
- agent.py can be about the same, it will just need to use different instances of objects for ROS/sympy
- centralized.py needs complete overhaul
- clock.py is fine for both ROS and sympy version. Just changes the time backend
- dynamics.py needs to be completely different. Need to separate rover dynamics from rover navigation. Control is no longer known and redoing that will take a major update
- env.py can be mostly the same, except in the ROS version it won’t create instances of 
- estimator.py can stay the same
- filters.py will be broken until the dynamics overhaul, but should be fine as-is
- radio.py is going to be almost completely different, make a new ROS version
- sensors.py can be almost exactly the same. radio.py can convert ROS messages to native Message type. Will also rely on fixing up dynamics

## Roadmap
- Goal is to incrementally build up ROS version
- [ ] agent_wrapper create env
- [ ] Create agent
- [ ] Run clock from agent
- [ ] Create new radio.py and transmission function. Publish time to ensure things are working
- [ ] Ensure messages can be received by other agents
