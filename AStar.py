class Grid:
    """Grid class to represent the maze/grid"""
    def __init__(self, board: list):
        # Initialize grid from 2D list
        self.grid = board
        self.x = len(board)
        self.y = len(board[0])
    
    def __getitem__(self, position: tuple) -> int:
        # Access grid value at given position
        x, y = position
        return self.grid[x][y]

class Node:
    """A node class for A* Pathfinding"""
    def __init__(self, pos: tuple, g: int, h: int, parent=None):
        self.pos = pos
        self.g = g  # cost
        self.h = h  # heuristic
        self.f = g + h
        self.parent = parent
    
    def update(self, g: int, parent: 'Node'):
        self.g = g
        self.f = g + self.h
        self.parent = parent

class PriorityQueue:
    """A simple priority queue based on a list"""
    def __init__(self):
        self.queue = []
        self.nodes = {}
        self.len = 0
        
    def enqueue(self, node: Node):
        # Insert node in sorted order based on f value
        idx = 0
        while idx < self.len and self.nodes[self.queue[idx]].f < node.f:
            idx += 1

        self.queue.insert(idx, node.pos)
        self.nodes[node.pos] = node
        self.len += 1
    
    def dequeue(self) -> tuple:
        # Remove and return the node with the lowest f value
        self.len -= 1
        return self.nodes.pop(self.queue.pop(0))
    
    def is_empty(self) -> bool:
        # Check if the priority queue is empty
        return self.len == 0
    
    def get(self, pos: tuple) -> Node:
        # Retrieve a node by its position
        return self.nodes.get(pos, None)


class TieBreakingPQ(PriorityQueue):
    """A priority queue with tie-breaking on h value"""
    def enqueue(self, node: Node):
        # Insert node in sorted order based on f value with tie-breaking on h
        idx = 0
        while idx < self.len:
            curr = self.nodes[self.queue[idx]]
            # Lower f first
            if curr.f > node.f:
                break
            # Tie-breaking: lower h
            if curr.f == node.f and curr.h > node.h:
                break
            idx += 1

        self.queue.insert(idx, node.pos)
        self.nodes[node.pos] = node
        self.len += 1


def h(a: tuple, b: tuple) -> int:
    """Calculate Manhattan distance between two points"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def c(a: tuple, b: tuple) -> int:
    """Cost function between two adjacent nodes"""
    return 1

def get_children(position: tuple, grid: Grid) -> list:
    """Get valid neighboring positions (up, down, left, right)"""
    neighbors = []
    x, y = position
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.x and 0 <= ny < grid.y and grid[(nx, ny)] == 0:
            neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(node: Node) -> list:
    """Reconstruct path from start to goal"""
    path = []
    while node:
        path.append(node.pos)
        node = node.parent
    return path[::-1]

def A_star(start: tuple, goal: tuple, board: list) -> list:
    """A* Search Algorithm"""
    grid = Grid(board)
    if grid[start] != 0 or grid[goal] != 0:
        return None  # Start or goal is blocked

    open_set = TieBreakingPQ()
    open_set.enqueue(Node(start, 0, h(start, goal)))
    
    closed_set = set()

    while not open_set.is_empty():
        current = open_set.dequeue()

        if current.pos == goal:
            return reconstruct_path(current)

        closed_set.add(current.pos)

        for child in get_children(current.pos, grid):
            if child in closed_set:
                continue
            
            g = current.g + c(current.pos, child)

            child_node = open_set.get(child)
            if child_node is None:
                open_set.enqueue(Node(child, g, h(goal, child), current))
            elif g < child_node.g:
                child_node.update(g, current)

    return None

def Bi_HS(start: tuple, goal: tuple, board: list) -> list:
    """Bidirectional Heuristic Search (Bidirectional A*) Search Algorithm"""
    grid = Grid(board)
    if grid[start] != 0 or grid[goal] != 0:
        return None  # Start or goal is blocked

    open_f, open_b = TieBreakingPQ(), TieBreakingPQ()
    open_f.enqueue(Node(start, 0, h(start, goal)))
    open_b.enqueue(Node(goal, 0, h(goal, start)))
    
    closed_f, closed_b = set(), set()

    while not open_f.is_empty() and not open_b.is_empty():
        # Search forward from start
        curr_f = open_f.dequeue()

        meeting_point = open_b.get(curr_f.pos)
        if meeting_point is not None:
            # Backtrack path from start to meeting point and from meeting point to goal
            return reconstruct_path(curr_f) + reconstruct_path(meeting_point)[::-1][1:]
        
        closed_f.add(curr_f.pos)

        for child in get_children(curr_f.pos, grid):
            if child in closed_f:
                continue
            
            g = curr_f.g + c(curr_f.pos, child)

            child_node = open_f.get(child)
            if child_node is None:
                open_f.enqueue(Node(child, g, h(goal, child), curr_f))
            elif g < child_node.g:
                child_node.update(g, curr_f)

        # Search backward from goal
        curr_b = open_b.dequeue()
        
        meeting_point = open_f.get(curr_b.pos)
        if meeting_point is not None:
            # Backtrack path from start to meeting point and from meeting point to goal
            return reconstruct_path(meeting_point) + reconstruct_path(curr_b)[::-1][1:]
        
        closed_b.add(curr_b.pos)
        
        for child in get_children(curr_b.pos, grid):
            if child in closed_b:
                continue
            
            g = curr_b.g + c(curr_b.pos, child)

            child_node = open_b.get(child)
            if child_node is None:
                open_b.enqueue(Node(child, g, h(start, child), curr_b))
            elif g < child_node.g:
                child_node.update(g, curr_b)          

    return None

def MM(start: tuple, goal: tuple, board: list) -> list:
    """Meet-in-the-Middle A* Search Algorithm"""
    grid = Grid(board)
    if grid[start] != 0 or grid[goal] != 0:
        return None  # Start or goal is blocked

    open_f, open_b = TieBreakingPQ(), TieBreakingPQ()
    open_f.enqueue(Node(start, 0, h(start, goal)))
    open_b.enqueue(Node(goal, 0, h(goal, start)))
    
    closed_f, closed_b = set(), set()

    U = float('inf')
    meet = None

    def prmin(q: PriorityQueue) -> int:
        return q.nodes[q.queue[0]].f if not q.is_empty() else int('inf')
    
    def gmin(q: PriorityQueue) -> int:
        return min(node.g for node in q.nodes.values()) if not q.is_empty() else int('inf')

    while not open_f.is_empty() and not open_b.is_empty():
        fminF = prmin(open_f)
        fminB = prmin(open_b)
        C = min(fminF, fminB)
        gminF = gmin(open_f)
        gminB = gmin(open_b)

        if U <= max(C, fminF, fminB, gminF + gminB + 0):
            break 
        
        if C == fminF:
            # Search forward from start
            curr_f = open_f.dequeue()
            if curr_f.pos in closed_f:
                continue
            closed_f.add(curr_f.pos)

            for child in get_children(curr_f.pos, grid):
                g = curr_f.g + c(curr_f.pos, child)
                
                child_node = open_f.get(child)
                if child_node is None:
                    open_f.enqueue(Node(child, g, h(child, goal), curr_f))
                elif g < child_node.g:
                    child_node.update(g, curr_f)
                    open_f.enqueue(child_node)
                
                node_b = open_b.get(child)
                if node_b is not None:
                    total_g = g + node_b.g
                    if total_g < U:
                        U = total_g
                        meet = child
        else:
            # Search backward from goal
            curr_b = open_b.dequeue()
            if curr_b.pos in closed_b:
                continue
            closed_b.add(curr_b.pos)

            for child in get_children(curr_b.pos, grid):
                g = curr_b.g + c(curr_b.pos, child)
                
                child_node = open_b.get(child)
                if child_node is None:
                    open_b.enqueue(Node(child, g, h(child, start), curr_b))
                elif g < child_node.g:
                    child_node.update(g, curr_b)
                    open_b.enqueue(child_node)
                
                node_f = open_f.get(child)
                if node_f is not None:
                    total_g = g + node_f.g
                    if total_g < U:
                        U = total_g
                        meet = child
    
    if meet is not None:
        return reconstruct_path(open_f.get(meet)) + reconstruct_path(open_b.get(meet))[1:]

    return None

# Example usage
if __name__ == "__main__":
    board = [
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0]
    ]
    start = (0, 0)
    goal = (4, 4)

    for func in [A_star, Bi_HS, MM]:
        path = func(start, goal, board)
        print(f"Path found by {func.__name__}: {path}")
