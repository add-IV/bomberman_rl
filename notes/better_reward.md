# Improving the reward function

## Current reward function

The current reward function gives reward for certain events, as well as a discounted reward for the future. However, the reward for the future might be contributing to a lot of noise in the reward signal. To test this, we can look at the some statistics of the rewards.

## Statistics of the rewards

Using pandas describe function, we can get some statistics of the rewards [Here](testing_rewards.ipynb):

| statistic | value         |
| --------- | ------------- |
| count     | 383781.000000 |
| mean      | 0.165386      |
| std       | 0.477852      |
| min       | -1.000000     |
| 25%       | -0.038170     |
| 50%       | 0.041675      |
| 75%       | 0.471077      |
| max       | 1.000000      |
| = 0       | 23906         |
| > 0       | 221310        |
| < 0       | 138565        |

## Results

The current reward function works well, actually. We could increase the amount of 0 reward position by discarding rewards too much in the future. We could also increase the std by multiplying the reward by a factor.