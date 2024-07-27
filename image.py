from typing import Tuple, List

import numpy as np

import settings as s
from agents import Agent
from environment import GenericWorld, WorldArgs
from items import Coin, Bomb, Explosion


class ImageWorld(GenericWorld):
    """
    World class for displaying a single image.

    Imput file should be a .txt file with the following format:
    first line: n_rows, n_cols, n_agents
    next n_rows x n_cols space: the arena
    w : wall
      : empty space
    c : crate
    0 : coin
    a0, a1, a2, a3: agent 0, 1, 2, 3
    b0, b1, b2, b3: bomb of agent 0, 1, 2, 3
    e0, e1, e2, e3: explosion in stage 0, 1, 2, 3
    s1, s2: smoke in stage 1, 2
    next n_agents lines: agent name, code name

    Example:
    5 5 2
    w w w w w 
    w 0 w c w 
    w s1e2b1w 
    w a1a2  w 
    w w w w w 
    agent1 user_agent
    agent2 tpl_agent

    the code names are used to load avatar and bomb sprites from the code directory
    """
    def __init__(self, args: WorldArgs):
        super().__init__(args)

        image_file = args.image
        self.logger.info(f'Loading world file "{image_file}"')
        self.image_file = image_file
        with open(image_file, 'r') as f:
            self.image_source = f.read()

        # get map size and number of agents
        n_rows, n_cols, n_agents = self.image_source.split('\n')[0].split()
        self.n_rows, self.n_cols, self.n_agents = int(n_rows), int(n_cols), int(n_agents)

        arena_string = self.image_source.split('\n')[1:self.n_rows+1]
        # Transpose the arena since the gui displays it transposed
        print([[row[i:i+2] for i in range(0, len(row), 2)] for row in arena_string])
        self.arena_char_array = np.array([[row[i:i+2] for i in range(0, len(row), 2)] for row in arena_string]).T

        # get the agents
        self.agents = []
        for line in self.image_source.split('\n')[self.n_rows+1:]:
            split = line.split()
            agent_name = split[0]
            code_name = split[1]
            avatar_sprite_desc = bomb_sprite_desc = self.colors.pop()
            self.agents.append(ImageAgent(agent_name, code_name, avatar_sprite_desc, bomb_sprite_desc))

    def build_arena(self) -> Tuple[np.array, List[Coin], List[Agent]]:
        arena = np.zeros((self.n_rows, self.n_cols))
        agents = [None] * self.n_agents
        coins = []
        bombs = []
        explosions = []
        for index, identifier in np.ndenumerate(self.arena_char_array):
            if identifier[0] == 'w':
                arena[index] = -1
            elif identifier[0] == 'c':
                arena[index] = 1
            elif identifier[0] == '0':
                coins.append(Coin(index, True))
            elif identifier[0] == 'a':
                print(int(identifier[-1]))
                agents[int(identifier[-1])] = self.agents[int(identifier[-1])]
                agents[int(identifier[-1])].x, agents[int(identifier[-1])].y = index
            elif identifier[0] == 'b':
                agent = self.agents[int(identifier[-1])]
                bombs.append(Bomb(index, agent, 4, 4, agent.bomb_sprite))
            elif identifier[0] == 'e':
                timer = int(identifier[1])
                screen_coords = (s.GRID_OFFSET[0] + s.GRID_SIZE * index[0],
                         s.GRID_OFFSET[1] + s.GRID_SIZE * index[1])
                explosions.append(Explosion([index], [screen_coords], None, timer))
            elif identifier[0] == 's':
                timer = int(identifier[1])
                screen_coords = (s.GRID_OFFSET[0] + s.GRID_SIZE * index[0],
                         s.GRID_OFFSET[1] + s.GRID_SIZE * index[1])
                explosion = Explosion([index], [screen_coords], None, timer)
                explosion.stage = 1
                explosions.append(explosion)
        self.bombs = bombs
        self.explosions = explosions
        return arena, coins, agents
    
    
    def do_step(self, user_input='WAIT'):
        assert self.running

        self.step += 1
        self.logger.info(f'STARTING STEP {self.step}')

        self.user_input = user_input
        self.logger.debug(f'User input: {self.user_input}')

        self.poll_and_run_agents()

        if self.time_to_stop():
            self.end_round()

    def poll_and_run_agents(self):
        pass

    def time_to_stop(self):
        time_to_stop = True
        return time_to_stop

class ImageAgent(Agent):
    """
    Agents class firing off a predefined sequence of actions.
    """

    def __init__(self, agent_name, code_name, avatar_sprite_desc, bomb_sprite_desc):
        """Recreate the agent as it was at the beginning of the original game."""
        super().__init__(agent_name, code_name, agent_name, False, None, avatar_sprite_desc, bomb_sprite_desc)

    def setup(self):
        pass

    def act(self, game_state):
        pass

    def wait_for_act(self):
        pass
