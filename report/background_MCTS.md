# Monte Carlo Tree Search (MCTS)

Monte Carlo Tree Search (MCTS) is a decision-making algorithm that explores the game tree based on expected rewards. It prioritizes promising moves and avoids exploring less favorable ones.

The MCTS algorithms consists of four steps: selection, expansion, simulation and backpropagation.

1. **Selection:** Starting from the root node, the algorithm selects the child node with the highest Upper Confidence Bound (UCB) value. This is done until a leaf node is reached.

2. **Expansion:** The leaf node is expanded by adding its child nodes to it. These child nodes represent possible moves from the current game state.

3. **Simulation:** A simulation is run from the expanded node until a terminal state is reached. The simulation is done by selecting moves randomly. This classical approach is called "rollout", but for our project we used a different approach to evaluate the game state.

4. **Backpropagation:** The result of the simulation is backpropagated up the tree. The number of visits and the total reward of the node are updated based on the result of the simulation.

Since Bomberman has four agents with 6 possible actions each, the branching factor of one step is in the worst case 6^4 = 1296, which is even bigger than in the game of Go, which is already known for its high branching factor. This makes MCTS a good choice for Bomberman, as it can explore the game tree efficiently and make informed decisions.

AlphaGo, the AI that beat the world champion in Go, used MCTS as its main decision-making algorithm. It was able to achieve this result by using Neural Networks instead of the classical rollout approach to evaluate the game state. This approach is called MCTS with Deep Network Evaluation (TODO: source) and is the approach we used in our project.

## Tree nodes and the UCT

The nodes store the prior of each action, the number of visits, and the total reward. The prior of each action is used to guide the selection of the next node, the number of visits is used to calculate the Upper Confidence Bound (OCB) value, and the total reward is used to backpropagate the result of the simulation.

To navigate the tree, the Upper Confidence Bound for Trees (UCT) algorithm is used. The UCT algorithm balances exploration and exploitation by selecting the child node with the highest UCB value.

$$
UCT = \frac{Q}{N} + c \cdot \sqrt{\frac{\log{N_p}}{N}}
$$

- $Q$ is the total reward of the node
- $c$ is a constant that determines the balance between exploration and exploitation, the "exploration constant"
- $N_p$ is the number of visits of the parent node
- $N$ is the number of visits of the node

This algorithm balances exploration and exploitation by selecting the child node with the highest UCB value. The exploration constant $c$ determines the balance between exploration and exploitation. A higher value of $c$ will lead to more exploration, while a lower value will lead to more exploitation.

## MCTS with Deep Network Evaluation

Using a MCTS with Deep Network Evaluation allows us to efficiently use the available time to explore the game tree as much as possible. The MCTS can explore any number of states until its time limit is reached, and then use the best move found so far.

In our implementation, we used a DQN to evaluate the game state. The DQN outputs the value of the state and the prior of each action. The value of the state is used to backpropagate the result of the simulation, and the prior of each action is used to guide the selection of the next node. This was done because updating the game state is too expensive.