# Description of Crosscombe 2017 benchmark

All robots start with a random belief state. That is, flawed and non-flawed robots have completely random belief state vectors. For example, robot 1 can have <0, 1, 1, 0, 0> while robot 2 can have <0, 2, 2, 0, 1>. Note that robot 2's belief will be normalized before being populated and communicated, i.e., robot 2's normalized belief is <0, 1, 1, 0, 0>.

During the updating phase:
- If the non-flawed robots cannot decide option -- they have multiple indeterminate beliefs, e.g., <1, 1, 0, 1, 0> -- they will randomly pick an option from one of the indeterminate beliefs, while broadcasting the belief. This makes sense because just because a robot randomly picks from options it is uncertain about, it doesn't mean that the robot now is certain about the choice.