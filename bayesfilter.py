import numpy as np
from scipy.stats import binom
from pacman_module.util import PriorityQueue

from pacman_module.game import Agent, Directions, manhattanDistance

def gridSize(grid):
    n_x = 0
    n_y = 0

    for x in grid:
        n_x += 1

    for x in grid[0]:
        n_y += 1

    return (n_x, n_y)


def mazeDistance(walls, position, destination):
    
    if walls[destination[0]][destination[1]]:
        return -1
    
    fringe = PriorityQueue()
    
    fringe.push((position,0),0)
    
    closed = set()
    
    while True:
        
        if fringe.isEmpty():
            return 0
        
        current, dist = fringe.pop()[1]
        
        if current == destination:
            return dist
        
        if current in closed:
            continue
        else:
            closed.add(current)
            
        x,y = current
    
        actions = [(x-1, y),(x+1, y),(x, y-1),(x,y+1)]
        
        for action in actions:
            if not walls[action[0]][action[1]]:
                a = manhattanDistance(action, destination) + dist
                fringe.push((action,dist + 1),a)
            
    return dist
            

class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def transition_matrix(self, walls, position):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (k, l) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (k, l).
        """
            
        W,H = gridSize(walls)
        
        trans = np.zeros((W,H,W,H))
        
        fear = 0
        
        if self.ghost == "afraid":
            fear = 1.0
        
        elif self.ghost == "terrified":
            fear = 3.0
            
        for i in range(W):
            for j in range(H):        
                if walls[i][j]:
                    continue
                
                currentDist = manhattanDistance(position,(i,j))
                
                sum = 0
                
                if not walls[i-1][j]: # proba of going right
                    val = 2*fear if manhattanDistance(position,(i-1,j)) >= currentDist else 1
                    trans[i][j][i-1][j] = val
                    sum += val
                    
                if not walls[i+1][j]: # proba of going left 
                    val = 2*fear if manhattanDistance(position,(i+1,j)) >= currentDist else 1
                    trans[i][j][i+1][j] = val
                    sum += val
                    
                if not walls[i][j-1]: #proba of going down
                    val = 2*fear if manhattanDistance(position,(i,j-1)) >= currentDist else 1
                    trans[i][j][i][j-1] = val
                    sum += val
                    
                if not walls[i][j+1]: #proba of going up
                    val = 2*fear if manhattanDistance(position,(i,j+1)) >= currentDist else 1
                    trans[i][j][i][j+1] = val
                    sum += val
                
             
                
                # normalizing
                if sum == 0:
                    continue
                trans[i][j][i-1][j] /= sum
                trans[i][j][i+1][j] /= sum
                trans[i][j][i][j-1] /= sum
                trans[i][j][i][j+1] /= sum


                #print(f"dim of trans matrix : {trans.shape} and dim of trans.T matrix : {trans.T.shape}") 
               
            
        return trans

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """

        W,H = gridSize(walls)
        
        n = 4
        p = 0.5
        
        Observ = np.ndarray((W,H))
        sum = 0
        
        for x in range(W):
            for y in range(H):
                trueDist = manhattanDistance(position,(x,y))
                z = evidence - trueDist + n*p
                p_z = binom.pmf(z,n,p)
                
                Observ[x][y] = p_z
                sum += p_z

        #normalizing        
        for x in range(W):
            for y in range(H):
                Observ[x][y] /= sum 
                

        """
        plt.imshow(Observ, cmap = 'hot')
        leg = "dist = " + str(evidence)
        plt.title(leg)
        plt.show()
        """
                
        return Observ
    
    def update(self, walls, belief, evidence, position): #Do I have to use a prior or is comprised in evidence ? 
        
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1}) (user note : b_{t-1} is actually f_{t-1} in the course given its definition.)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1} (f_{t-1}).
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """
        W,H = gridSize(walls)
            
        T = self.transition_matrix(walls, position)
        O = self.observation_matrix(walls, evidence, position)
        sum = 0
        alpha = 0
       
        """
        The problem is T is 4 dimensional whereas O is 2 dimensional so we must marginalize T with respect to k and l in order to compute the updated belief state.
        We call A the matrix obtained by this marginalization.
        """
        A = np.zeros((W,H)) 
        for i in range(W):
            for j in range(H):
                for k in range(W):
                    for l in range(H):
                        sum += T.T[j][i][l][k]*belief[k][l] 
                        
                if not walls[i][j]: #The ghost can't be on a wall therefore the belief must be zero.
                    A[i][j] = O[i][j]*sum
                
    
                alpha += A[i][j] #computing the normalization constant. 
        
        
        
        #Then we multiply A by the normalization constant such that sum of all the elements equals to 1.
        for i in range(W):
            for j in range(H):
                if alpha != 0:
                    A[i][j] /= alpha
                """
                The following condition translates the fact that updated believes have to be near the previous believes since the ghost can only move
                from one case to another and can't cross a wall. Without this conditions, new believes appear at the other side of walls while the ghost 
                is not physically capable of going there in one move. Hence, such believes have to be suppressed from the updated belief matrix A.
                """   
                if not walls[i][j] and A[i][j] != 0 and not belief[i-1][j] and not belief[i+1][j] and not belief[i][j+1] and not belief[i][j-1]:
                    A[i][j] = 0
                
        #print(A)
        return A       

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()
        
        

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls,
                    beliefs[i],
                    evidences[i],
                    position,
                )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()
        self.one = True

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        
        """  
        if self.one:
            breakpoint()
            self.one = False
        
        def proba(walls, position, beliefs, gIndex): #heuristic
        
            prob = 0
            
            W,H = gridSize(walls)
            
            for i in range(W):
                for j in range(H):
                    prob += mazeDistance(walls,position,(i,j))*beliefs[gIndex][i][j]
                    
            return prob
        
        
        #determining ghost to be eaten
        gIndex = 0
        
        while eaten[gIndex] and gIndex < len(eaten):
            gIndex += 1
            
        W,H = gridSize(walls)
        
        x,y = position
            
        actions = [((x-1,y),"West"),((x+1,y),"East"),((x,y-1),"South"),((x,y+1),"North")]
        
        
        
        """proba = 0
        maxProbaPos = (0,0)
        
        for x in range(W):
            for y in range(H):
                if beliefs[gIndex][x][y] > proba:
                    proba = beliefs[gIndex][x][y]
                    maxProbaPos = (x,y)"""
                    
        
        #chosing action leading the closest to the highest proba point
        
        dist = 100000
        
        taken = None
        
        for next, action in actions:
            if walls[next[0]][next[1]]:
                continue
            
            nextDist = proba(walls,next,beliefs,gIndex)
            
            if nextDist < dist:
                dist = nextDist
                taken = action
        
        self.lastPosition = position
                
        return taken
            

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
