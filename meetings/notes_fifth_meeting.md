# Meeting 08.09.2024

attendance: 2/3

## Topics discussed

- [x] Current progress
- [x] Next steps
- [x] Discussions
- [x] Next meeting

## Current progress

### MCTS

model exists, mcts still needs to be finished

## Next steps

### QTable

4 features: blocked squares, top, right, bottom, left
4 features: explosion map, top, right, bottom, left
direction to nearest coin, crate, safe square 0 - 4: no coin, top, right, bottom, left (maybe ignore distant coins)

if top, left are blocked, and right is in explosion map

4^2 \* 4^2 \* 5^3

permutation function:
if there is one empty space, and the top is blocked, then rotate until the top is not blocked
if there is two empty spaces

1001 0100 123
right, bottom

1->2->3->4

1100
0010
234
bottom, left

4->3->2->1
right, bottom

1001 0100 -> 0 - 255 binary to decimal
first number \* 25 + second number \* 5 + third number
number between 0-124

first composite number * 125 + second composite number

feature to index -> 1-32000

table = np.random.rand(32000, 6)
divide each row by the sum of the row

table[index] = [0.1, 0.2, 0.3, 0.4, 0, 0]

-> index in table

[] 32000

train -> random sample from table, observe reward, update table

## Discussions

possibly use qtable for mcts to speed up the process. Still need a function to estimate the score/ reward.

## Next meeting

wednesday 11.09.2024

