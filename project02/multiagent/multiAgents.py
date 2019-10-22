# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        # sucessorGameState: next position of current position
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # get all ghost's position
        ghostPos = [(ghost.getPosition()[0], ghost.getPosition()[1]) for ghost in newGhostStates]

        # pacman can eat ghost
        if newScaredTimes > 0 and (newPos in ghostPos):
            return -1

        # pacman eat a new dot
        if newPos in currentGameState.getFood().asList():
            return 1

        # calculate manhattan distance of all food and ghost
        # sort it and get the closest one
        closestFood = sorted(newFood, key=lambda food: util.manhattanDistance(food, newPos))[0]
        closestGhost = sorted(ghostPos, key=lambda ghost: util.manhattanDistance(ghost, newPos))[0]

        return 1.0/util.manhattanDistance(closestFood, newPos) - 1.0/util.manhattanDistance(closestGhost, newPos)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        GhostIndex = [i for i in range(1, gameState.getNumAgents())]
        largeNumber = 1000000
        # Check if the game is End or search to the end of the tree
        def checkTerminal(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        # for ghost
        def min_value(state, depth, ghost):
            v = largeNumber
            if checkTerminal(state, depth):
                return self.evaluationFunction(state)
            # find v min from all ghost
            for action in state.getLegalActions(ghost):
                # If all ghost've been estimate, next will be maximize with pacman
                if ghost == GhostIndex[-1]:
                    v = min(v, max_value(state.generateSuccessor(ghost, action), depth + 1))
                # next is minimizer with next-ghost
                else:
                    v = min(v, min_value(state.generateSuccessor(ghost, action), depth, ghost + 1))
            return v

        # for pacman
        def max_value(state, depth):
            v = -largeNumber
            if checkTerminal(state, depth):
                return self.evaluationFunction(state)
            for action in state.getLegalActions():
                # generateSuccessor(0, action): 0 is agentIndex - pacman
                v = max(v, min_value(state.generateSuccessor(0, action), depth, 1))
            return v

        # action: move East, West, South, North
        result = [(action, min_value(gameState.generateSuccessor(0, action), 0, 1)) for action in gameState.getLegalActions(0)]
        # sort result by v - score of action
        result.sort(key=lambda k: k[1])
        return result[-1][0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        GhostIndex = [i for i in range(1, gameState.getNumAgents())]
        largeNumber = 1000000
        # Check if the game is End or search to the end of the tree
        def checkTerminal(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def min_value(state, depth, ghost, A, B):
            v = largeNumber
            if checkTerminal(state, depth):
                return self.evaluationFunction(state)
            # find v min from all ghost
            for action in state.getLegalActions(ghost):
                # If all ghost've been estimate, next will be maximize with pacman
                if ghost == GhostIndex[-1]:
                    v = min(v, max_value(state.generateSuccessor(ghost, action), depth + 1, A, B))
                # next is minimizer with next-ghost
                else:
                    v = min(v, min_value(state.generateSuccessor(ghost, action), depth, ghost + 1, A, B))
                # Keep v in range: B < v < A
                if v < A:
                    return v
                B = min(B, v)
            return v

        def max_value(state, depth, A, B):
            v = -largeNumber
            if checkTerminal(state, depth):
                return self.evaluationFunction(state)
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), depth, 1, A, B))

                if v > B:
                    return v
                A = max(A, v)
            return v

        def alphabeta(state):
            v = -largeNumber
            A = -largeNumber
            B = largeNumber
            act = None
            
            # maximizing
            for action in state.getLegalActions(0):
                temp = min_value(gameState.generateSuccessor(0, action), 0, 1, A, B)
                
                # v = max(v, tmp)
                if v < temp:
                    v = temp
                    act = action
                # pruning
                if v > B:
                    return v
                A = max(A, temp)
            return act

        return alphabeta(gameState)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        GhostIndex = [i for i in range(1, gameState.getNumAgents())]
        largeNumber = 1000000
        # Check if the game is End or search to the end of the tree
        def checkTerminal(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def exp_value(state, depth, ghost):  # minimizer
            v = 0
            if checkTerminal(state, depth):
                return self.evaluationFunction(state)
            for action in state.getLegalActions(ghost):
                if ghost == GhostIndex[-1]:
                    v +=  max_value(state.generateSuccessor(ghost, action), depth + 1)
                else:
                    v +=  exp_value(state.generateSuccessor(ghost, action), depth, ghost + 1)
            return v/len(state.getLegalActions(ghost))
        
        def max_value(state, depth):
            v = -largeNumber
            if checkTerminal(state, depth):
                return self.evaluationFunction(state)
            for action in state.getLegalActions(0):
                v = max(v, exp_value(state.generateSuccessor(0, action), depth, 1))
            return v

        result = [(action, exp_value(gameState.generateSuccessor(0, action), 0, 1)) for action in
               gameState.getLegalActions(0)]
        result.sort(key=lambda k: k[1])

        return result[-1][0]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()
    # list of all food distance
    foodDistance = [manhattanDistance(newPos, foodPosition) for foodPosition in currentGameState.getFood().asList()]
    # all ghost position
    ghostPos = [ghost for ghost in newGhostStates]
    # new food distance
    newFoodDistance = min(foodDistance) if len(foodDistance) > 0 else 0

    newGhostDistance = min([manhattanDistance(newPos, ghost.getPosition()) if ghost.scaredTimer < manhattanDistance(newPos, ghost.getPosition()) else 10000 for ghost in ghostPos])
    
    if newGhostDistance <= 0:
        return -1000000
    score = newFoodDistance
    return currentGameState.getScore() - score

# Abbreviation
better = betterEvaluationFunction
