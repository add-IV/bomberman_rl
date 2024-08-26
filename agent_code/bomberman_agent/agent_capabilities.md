# Overview of the agents capabilities as defined in the Task

predefined functions in callback.py:

- setup(self)
- act(self, game_state: dict)

predefined functions in train.py:

- setup_training(self)
- game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str])
- end_of_round(self, last_game_state: dict, last_action: str, events: List[str])

## setup

function to initialize the agent, executed at the beginning of the game

## act

function to return the next action of the agent, executed each time the agent is asked for an action.
Possible actions are: "UP", "RIGHT", "DOWN", "LEFT", "BOMB", "WAIT"

## setup_training

function to initialize the agent for training, executed at the beginning of the training

## game_events_occurred

function to update the agent after each step, executed after each step of the game

## end_of_round

function to update the agent after each round, executed after each round of the game

## Parameters

### self

the agent instance, can be used to store any information that should be kept between actions

### self.logger

a logger for the agent to print information to the console

### self.train

boolean indicating whether the agent is training or not

### game_state: dict

a dictionary containing the current game state

### game_state["round"]: int

the number of the current round, starting at 1

### game_state["step"]: int

the number of the current step in the round, starting at 1

### game_state["field"]: np.array(width, height)

a 2D numpy array representing the current tiles

- 0: free space
- -1: indestructible wall (stone)
- 1: destructible wall (crate)

### game_state["bombs"]: [((x, y), t)]

TODO: type hint in the exercise sheet is wrong, changed here but double check

a list of tuples ((x,y),t) representing the current bombs

- (x, y): position of the bomb
- t: time until the bomb explodes

### game_state["explosion_map"]

a 2D numpy array representing for each tile how much longer the explosion will be present

- 0: no explosion
- 1..4: explosion for 1 more step

TODO: not sure about the max amount of steps

### game_state["coins"]: [(x, y)]

a list of tuples (x, y) representing the current coins

### game_state["self"]: (str, int, bool, (int, int))

a tuple representing the agent itself, (n, s, b, (x, y))

- n: name of the agent
- s: score of the agent
- b: whether the agent has a bomb
- (x, y): position of the agent

### game_state["others"]: [(str, int, bool, (int, int))]

a list of tuples representing the other agents

### game_state["user_input"]: str

irrelevant for us

## Parameters for training

### old_game_state: dict

a dictionary containing the game state before the action, presumably the same as game_state

TODO: double check

### self_action: str

presumably the action that the agent took

TODO: double check

### new_game_state: dict

a dictionary containing the game state after the action, presumably the same as game_state

TODO: double check

### events: List[str]

defined in events.py

a list of events that occurred after the action

e.MOVED_LEFT, e.MOVED_RIGHT, e.MOVED_UP, e.MOVED_DOWN, e.WAITED, e.INVALID_ACTION, e.BOMB_DROPPED, e.BOMB_EXPLODED, e.CRATE_DESTROYED, e.COIN_FOUND, e.COIN_COLLECTED, e.KILLED_OPPONENT, e.KILLED_SELF, e.GOT_KILLED, e.OPPONENT_ELIMINATED, e.SURVIVED_ROUND

- MOVED_LEFT, MOVED_RIGHT, MOVED_UP, MOVED_DOWN: the agent moved in the respective direction
- WAITED: the agent waited
- INVALID_ACTION: the agent performed an invalid action
- BOMB_DROPPED: the agent dropped a bomb
- BOMB_EXPLODED: own bomb exploded
- CRATE_DESTROYED: the agent destroyed a crate
- COIN_FOUND: the agent revealed a coin by destroying a crate
- COIN_COLLECTED: the agent collected a coin
- KILLED_OPPONENT: the agent killed an opponent
- KILLED_SELF: the agent killed itself
- GOT_KILLED: the agent got killed by an opponent
- OPPONENT_ELIMINATED: an opponent got eliminated
- SURVIVED_ROUND: the agent survived the round

based on last_game_state and new_game_state, we can also define our own events
