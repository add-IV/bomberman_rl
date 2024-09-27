# Monte Carlo Tree Search (MCTS)

Monte Carlo Tree Search (MCTS) is a decision-making algorithm that explores the game tree based on expected rewards. It prioritizes promising moves and avoids exploring less favorable ones.

The MCTS algorithms consists of four steps: selection, expansion, simulation and backpropagation.

1. **Selection:** Starting from the root node, the algorithm selects the child node with the highest Upper Confidence Bound (UCB) value. This is done until a leaf node is reached.

2. **Expansion:** The leaf node is expanded by adding its child nodes to it. These child nodes represent possible moves from the current game state.

3. **Simulation:** A simulation is run from the expanded node until a terminal state is reached. The simulation is done by selecting moves randomly. This classical approach is called "rollout", but for our project we used a different approach to evaluate the game state.

4. **Backpropagation:** The result of the simulation is backpropagated up the tree. The number of visits and the total reward of the node are updated based on the result of the simulation.

Since Bomberman has four agents with 6 possible actions each, the branching factor of one step is in the worst case 6^4 = 1296, which is even bigger than in the game of Go, which is already known for its high branching factor. This makes MCTS a good choice for Bomberman, as it can explore the game tree efficiently and make informed decisions.

AlphaGo, the AI that beat the world champion in Go, used MCTS as its main decision-making algorithm. It was able to achieve this result by using Neural Networks instead of the classical rollout approach to evaluate the game state. This approach is called MCTS with Deep Network Evaluation (TODO: source) and is the approach we used in our project.

Using a MCTS with Deep Network Evaluation allows us to efficiently use the available time to explore the game tree as much as possible. The MCTS can explore any number of states until its time limit is reached, and then use the best move found so far.
