---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---



In this tutorial we will be explaining the Monte Carlo Tree Search algorithm and each part of the code. Recently we applied MCTS to develop our game. 

![My helpful screenshot](/assets/400by400.png)

Here is the link to the game: [Sudo Tic Tac Toe][jekyll-talk].

Rules for our game(mode 1) are as follows:

1. The game is played on a 9 by 9 grid like Sudoku.
2. This big 9 by 9 grid is divided into 9 smaller 3 by 3 grids (local board).
3. Aim of the game is to win any one local board of the 9 available.
4. Your move determines in which local board A.I has to make a move and viceversa.
5. For example you make a move in position 1 of local board number 5.
 This will force the A.I to make a move in local board number 1.
6. Rules of normal Tic Tac Toe are applied to local board.

As you would have seen this game has a very high branching factor. For the first move the entire board is empty. So there are 81 empty spots. For the first turn it has 81 possible moves. For the second turn by applying rule 4 it has 8 or 9 possible moves. For the first 2 moves this results in 81*9 = 729 possible combinations. Thus the number of possible combinations increases as the game progresses, resulting in a high branching factor. For both the modes of our game the branching factor is very high. For games with such high branching factor it's not possible to apply the minimax algorithm. MCTS algorithm works for these kind of games. Also as you would have seen from playing the game the time it takes for the ai to make a move is just about a second. MCTS has been applied to both the modes of the game. Below we demonstrate the MCTS code in Python.

First we need to import numpy and defaultdict.

```python:
import numpy as np
from collections import defaultdict
```
Define MCTS class as shown below. 

```python:
class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return

```
### Constructor is used to initialize the following variables.
- state: For our game it represents the board state. Generally the board state is represented by an array. For normal Tic Tac Toe, it is a 3 by 3 array.
- parent: It is None for the root node and for other nodes it is equal to the node it is derived from. For the first turn as you have seen from the game  it is None.
- children: It contains all possible actions from the current node. For the first turn in our game this is 9 or 8 depending on where you make your move.
- parent_action: None for the root node and for other nodes it is equal to the action which it's parent carried out.
- _number_of_visits: Number of times current node is visited
- results: It's a dictionary
- _untried_actions: Represents the list of all possible actions
- action: Move which has to be carried out.  
Class consists of the following member functions. All the functions below are member function except the main() function.

```python:
def untried_actions(self):

    self._untried_actions = self.state.get_legal_actions()
    return self._untried_actions
```
Returns the list of untried actions from a given state. For the first turn of our game there are 81 possible actions. For the second turn it is 8 or 9. This varies in our game.

```python:    
def q(self):
    wins = self._results[1]
    loses = self._results[-1]
    return wins - loses
```
Returns the difference of wins - losses

```python:   
def n(self):
    return self._number_of_visits
```
Represents the number of times each node is visited.

```python:
def expand(self):
	
    action = self._untried_actions.pop()
    next_state = self.state.move(action)
    child_node = TwoPlayersGameMonteCarloTreeSearchNode(
		next_state, parent=self, parent_action=action)

 

	
    self.children.append(child_node)
    return child_node 
``` 
In this step all the possible states are appended to the children array and the child_node is returned. The states which are possible from the present state are all appended to the children array and the child_node corresponding to this state is returned.

```python: 
def is_terminal_node(self):
    return self.state.is_game_over()
``` 
This is used to check if the current node is terminal or not. Terminal node is reached when the game is over.

```python: 
def rollout(self):
    current_rollout_state = self.state
    
    while not current_rollout_state.is_game_over():
        
        possible_moves = current_rollout_state.get_legal_actions()
        
        action = self.rollout_policy(possible_moves)
        current_rollout_state = current_rollout_state.move(action)
    return current_rollout_state.game_result()
``` 
From the current state, entire game is simulated till there is an outcome for the game. This outcome of the game is returned. For example if it results in a win, the outcome is 1. Otherwise it is -1 if it results in a loss. And it is 0 if it is a tie. If the entire game is randomly simulated, that is at each turn the move is randomly selected out of set of possible moves, it is called light playout.

```python: 
def backpropagate(self, result):
    self._number_of_visits += 1.
    self._results[result] += 1.
    if self.parent:
        self.parent.backpropagate(result)
``` 
In this step all the statistics for the nodes are updated. Untill the parent node is reached, the number of visits for each node is incremented by 1. If the result is 1, that is it resulted in a win, then the win is incremented by 1. Otherwise if result is a loss, then loss is incremented by 1.

```python:    
def is_fully_expanded(self):
    return len(self._untried_actions) == 0

``` 
All the actions are poped out of _untried_actions one by one. When it becomes empty, that is when the size is zero, it is fully expanded.

```python: 
def best_child(self, c_param=0.1):
    
    choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
    return self.children[np.argmax(choices_weights)]

``` 
Once fully expanded, this function selects the best child out of the children array. The first term in the formula corresponds to exploitation and the second term corresponds to exploration.

```python: 
def rollout_policy(self, possible_moves):
    
    return possible_moves[np.random.randint(len(possible_moves))]

```
Randomly selects a move out of possible moves. This is an example of random playout.

```python:
def _tree_policy(self):

    current_node = self.root
    while not current_node.is_terminal_node():
        
        if not current_node.is_fully_expanded():
            return current_node.expand()
        else:
            current_node = current_node.best_child()
    return current_node
```       
Selects node to run rollout.
    
```python:
def best_action(self):
    simulation_no = 100
	
	
    for i in range(simulation_no):
		
        v = self._tree_policy()
        reward = v.rollout()
        v.backpropagate(reward)
	
    return self.best_child(c_param=0.)
```
This is the best action function which returns the node corresponding to best possible move.
The step of expansion, simulation and backpropagation are carried out by the code above.

```python:
def get_legal_actions(self): 
    '''
    Modify according to your game or needs.
    constructs a list of all possible states from current state.
    Returns an array.
    '''
```

```python:
def is_game_over(self):
    '''
    Modify according to your game or
    needs. It is the game over condition
    and depends on your game. Returns
    true or false
    '''
```

```python:
def game_result(self):
    '''
    Modify according to your game or 
    needs. Returns 1 or 0 or -1 depending
    on your state corresponding to win,
    tie or a loss.
    '''
```
```python:
def move(self,action):
    '''
    Modify according to your game or 
    needs. Changes the state of your 
    board with a new value. For a normal
    Tic Tac Toe game, it can be a 3 by 3
    array with all the elements of array
    being 0 initially. 0 means the board 
    position is empty. If you place x in
    row 2 column 3, then it would be some 
    thing like board[2][3] = 1, where 1
    represents that x is placed. Returns 
    the new state after making a move.
    '''

```

```python:
def main():
    root = MonteCarloTreeSearchNode(state,None,action)
    selected_node = root.best_action()
```
This is the main() function. Initialize the root node and call the best_action function to get the best node. This is not a member function of the class. All the other functions are member function of the class.

MCTS consists of 4 steps:

## SELECTION
The idea is to keep selecting best child nodes until we reach the leaf node of the tree. A good way to select such a child node is to use UCT (Upper Confidence Bound applied to trees) formula:

			wi/ni + c*sqrt(t)/ni
  

wi = number of wins after the i-th move  
ni = number of simulations after the i-th move  
c = exploration parameter (theoretically equal to √2)  
t = total number of simulations for the parent node  

## EXPANSION:

When it can no longer apply UCT to find the successor node, it expands the game tree by appending all possible states from the leaf node.

## SIMULATION:

After Expansion, the algorithm picks a child node arbitrarily, and it simulates entire game from selected node until it reaches the resulting state of the game. If nodes are picked randomly during the play out, it is called light play out. You can also opt for heavy play out by writing quality heuristics or evaluation functions.

## BACKPROPAGATION:

Once the algorithm reaches the end of the game, it evaluates the state to figure out which player has won. It traverses upwards to the root and increments visit score for all visited nodes. It also updates win score for each node if the player for that position has won the playout.


## DESIGNING YOUR GAME
If you plan to make your own game, you will have to think about the following questions.
1. How will you represent the state of your game? Think about the initial state in our game. 
2. What will be the end game condition for your game? Compare it with the end game condition of our game.
3. How will you get the legal actions in your game? Try getting the legal actions for the first move of our game. 

[Sudo Tic Tac Toe][jekyll-talk]


[![homepage](/assets/google-play-badge.png)][2]



[jekyll-talk]: https://play.google.com/store/apps/details?id=com.myComp.sudo
[2]: https://play.google.com/store/apps/details?id=com.myComp.sudo