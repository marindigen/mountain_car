# Mountain Car Environment - Reinforcement Learning Project

This project addresses the challenge of sparse rewards in the Mountain Car environment using three distinct reinforcement learning (RL) approaches:

1. **Deep Q-Networks (DQN) with Heuristic Rewards**: We implemented state-dependent heuristic rewards that guide the agent towards the goal, significantly improving training efficiency. This method provided consistent performance and faster convergence compared to standard DQN.

2. **DQN with Random Network Distillation (RND) Rewards**: This approach introduces a non-domain-specific intrinsic reward to encourage exploration. Although it achieved fast initial learning, the RND method demonstrated instability over time.

3. **Probabilistic Dyna-Agent**: A model-based approach that simulates the environment internally, enabling efficient and stable learning. The Dyna-agent required fewer episodes to converge, showcasing superior performance in sample efficiency and consistency.

### Results and Comparison

All three methods effectively addressed the sparse reward problem in the Mountain Car environment, outperforming standard DQN without auxiliary rewards. The DQN with heuristic rewards and the Dyna-agent both delivered reliable and consistent learning, with Dyna-agent showing the most stable performance over time. While the RND approach showed potential for rapid learning, it was less stable compared to the other two methods.

### Conclusion

This project demonstrates that incorporating auxiliary rewards or utilizing a model-based approach can significantly enhance the learning process in environments with sparse rewards. The Dyna-agent and DQN with heuristic rewards are recommended for tasks requiring stable and consistent policy learning.


### Additional Notes
Please adjust the file paths for saving plots and models to suit your setup.
