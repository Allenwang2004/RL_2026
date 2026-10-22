# RL_2026

Homework for the ETH Zurich robot learning course (2026). All robotics exercises use a MuJoCo simulation of the SO-100 robot arm. Each homework folder has its own README with setup instructions and the full task description.

| Folder | Topic |
|---|---|
| `hw2_robot_control_mdps` | Classical robot control and MDPs |
| `hw3_imitation_learning` | Imitation learning from teleoperated demonstrations |
| `hw4_reinforcement_learning` | Deep reinforcement learning for continuous control |

## HW2: Robot Control and MDPs

- **Exercise 1: Keypoints and inverse kinematics.** Generate keypoints along a Lemniscate of Bernoulli (infinity sign) in the workspace, and implement a Damped Least Squares IK solver that computes the joint angles that place the end-effector on each keypoint.
- **Exercise 2: Trajectory generation and PID control.** Interpolate between keypoints with quintic time scaling to get a smooth trajectory, convert it to joint targets with IK, and track it with a PID controller that outputs joint torques. Includes an analysis of how the PID gains affect tracking.
- **Exercise 3: RL policy for waypoint tracking.** Define a gymnasium environment (reset, observation in the robot base frame, action processing, and reward) and train a PPO policy with Stable-Baselines3 to move the end-effector to randomly sampled targets.

## HW3: Imitation Learning

- **Exercise 1: MSE policy.** Record demonstrations by teleoperating the arm with the keyboard to pick up a cube, move it around an obstacle, and place it in a bin. Then train an action-chunking policy that maps the current state to the next chunk of actions with an MSE loss.
- **Exercise 2: DAgger.** Evaluate the policy with adversarial obstacle placements that push it out of the training distribution. Take over control when it starts to fail, so that expert actions are recorded in exactly those states, then merge this data with the original demonstrations and retrain.

## HW4: Reinforcement Learning

- **Exercise 3: PPO on SO100.** Implement Proximal Policy Optimization from scratch (rollout collection, GAE, clipped surrogate loss, clipped value loss, entropy bonus, and a KL-adaptive learning rate) and train it on the SO100 end-effector tracking task from HW2.
- **Exercise 4: SAC on SO100.** Implement Soft Actor-Critic for the same task (squashed Gaussian actor, twin Q-networks with target networks and Polyak averaging, replay buffer, and automatic entropy temperature tuning) and compare it with PPO.
