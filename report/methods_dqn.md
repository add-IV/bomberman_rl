# Methods

## DQN

The Deep Q-Network (DQN) approach was chosen due to the large state space and complex dynamics of the game. By using a DQN, feature selection and engineering is not necessary, as the neural network can learn the features and their importance by itself. This theoretically allows the DQN to outperform other approaches. However, training is computationally expensive and local minima can be a problem. This section will describe the challenges and the implementation of the DQN approach.

One of the main challenges to the DQN approach is the large state space. With 176 non-wall spaces which can be occupied by at least crates, coins, players, bombs, and explosions (some with multiple states), the state space is cumbersomely large. While we first tried to use a one-hot encoding of the state space, this did not provide any results. We decided to use a state space centered around the player, initially with a radius of 3. This worked for the coin heaven scenario, but for the crate scenario, training again took a long time. To further reduce the state space, we decided on a shape that focuses on from where the player can be hit from.

![Danget Map](danger_map.png)

(TODO: red, bombs, explosions)

As can be seen in the figure above, with this method we only consider 36 spaces around the player. This is incidentally also a square number, which was useful so that we could easily use convolutional layers in the neural network. We mapped the state space to a 6x6 grid, while keeping neighboring spaces as neighbors in the grid.

The DQN used two convolutional layers followed by a dense layer and the output layer. The output layer had 6 nodes, one for each action. The DQN was trained using the Adam optimizer and the mean squared error loss function.

Since this approach was succeeded by the MCTS approach, we did further changes to the DQN so that it can be used in the MCTS. We changed the output to have 2 separate Heads, a Value Head and a Policy Head. The Value Head outputs a single value, which is the expected reward of the state. The Policy Head outputs the prior of each action.

We also changed the other structure of the neural network. First we used a convolutional block, followed by n residual blocks and finally the two heads. The convolutionaly block consisted of one convolutional layer, a batch normalization layer and a ReLU activation function. The residual blocks consisted of two convolutional layers, each followed by a batch normalization layer and a ReLU activation function. Before the second ReLU activation function, we added a skip connection. The output of the last residual block was fed into the two heads. The Policy Head used a convolutional layer with a kernel size of 1x1 with a batch normalization and ReLU layers to reduce the number of channels, and then a dense layer with a softmax activation function. The Value Head used a convolutional layer with a kernel size of 1x1 with a batch normalization and ReLU layers, and then a dense layer with the tanh activation function.

This final structure was first trained using the training data from prior DQN training. Then, the DQN was trained using the MCTS. The DQN was used to predict the value of the state and the prior of each action. The MCTS then used these predictions to guide the search. The DQN was then trained using the results of the MCTS.

## MCTS

The Monte Carlo Tree Search (MCTS) approach was chosen due to its success in other games with high branching factors, specifically AlphaGo's success in Go.

In our implementation, each MCTS node represents a game state. The root node is the current game state, and the children of each node are the possible game states that can be reached by taking an action. The MCTS algorithm consists of four steps: selection, expansion, simulation, and backpropagation as described in the background section.

While the classical approach to MCTS is to use a rollout to evaluate the game state, we used a different approach. We used the DQN to evaluate the game state. The DQN outputs the value of the state and the prior of each action. The value of the state is used to backpropagate the result of the simulation, and the prior of each action is used to guide the selection of the next node. This was done because updating the game state is too expensive.

After running some experiments, we found that the MCTS state evaluation was still too slow on the CPU, so we reduced the size of the state space to 3x4 with 7 layers for the different objects and the number of residual blocks. ![Smaller State](smaller_state.png)

