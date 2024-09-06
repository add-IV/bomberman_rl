# Meeting 06.09.2024

attendance: 2/3

## Topics discussed

- [x] Current progress
- [x] Next steps
- [x] Discussions
- [x] Next meeting

## Current progress

Deep Network on whole board is not working, might be able to work with better model but a local view should be better.
QTable also has to many states.

## Discussions

### Reducing feature space

For QTable, only look at the four closest squares and some info about the closest coins, crates and safe squares direction. This reduces the feature space to sub-10000.

For the Deep Network, only use a local view. Also maybe use a MCTS. Inspiration from board games.

### MCTS

Used for AlphaGo. It is a tree search algorithm.

need to reduce feature space by a lot

## Next meeting

- sunday 17:00