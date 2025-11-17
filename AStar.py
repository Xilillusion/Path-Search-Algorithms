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
    
    def peek(self) -> tuple:
        # Return the node with the lowest f value
        return self.nodes[self.queue[0]]
    
    def is_empty(self) -> bool:
        # Check if the priority queue is empty
        return self.len == 0
    
    def get(self, pos: tuple) -> Node:
        # Retrieve a node by its position
        return self.nodes.get(pos, None)

    def decrease_key(self, node: Node):
        if node.pos in self.queue:
            self.queue.remove(node.pos)
            self.len -= 1
            self.enqueue(node)


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

def A_star(start: tuple, goal: tuple, board: list) -> tuple:
    """A* Search Algorithm"""
    grid = Grid(board)
    if grid[start] != 0 or grid[goal] != 0:
        return None, 0  # Start or goal is blocked

    open_set = TieBreakingPQ()
    open_set.enqueue(Node(start, 0, h(start, goal)))
    
    closed_set = set()

    nodes_visited = 0

    while not open_set.is_empty():
        current = open_set.dequeue()
        nodes_visited += 1

        if current.pos == goal:
            return reconstruct_path(current), nodes_visited

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
                open_set.decrease_key(child_node)

    return None, nodes_visited

def Bi_HS(start: tuple, goal: tuple, board: list) -> tuple:
    """Bidirectional Heuristic Search (Bidirectional A*) Search Algorithm"""
    grid = Grid(board)
    if grid[start] != 0 or grid[goal] != 0:
        return None, 0  # Start or goal is blocked

    open_f, open_b = TieBreakingPQ(), TieBreakingPQ()
    open_f.enqueue(Node(start, 0, h(start, goal)))
    open_b.enqueue(Node(goal, 0, h(goal, start)))
    
    closed_f, closed_b = {}, {}

    U = float('inf')
    meet_f = None
    meet_b = None

    nodes_visited = 0

    while not open_f.is_empty() and not open_b.is_empty():
        # Early termination check
        if open_f.peek().f + open_b.peek().f >= U:
            break
            
        if open_f.peek().f <= open_b.peek().f:
            # Search forward from start
            curr_node = open_f.dequeue()
            if curr_node.pos in closed_f:
                continue
            nodes_visited += 1
            closed_f[curr_node.pos] = curr_node

            # Check if we've met the backward search
            if curr_node.pos in closed_b:
                g_total = curr_node.g + closed_b[curr_node.pos].g
                if g_total < U:
                    U = g_total
                    meet_f = curr_node
                    meet_b = closed_b[curr_node.pos]

            for child in get_children(curr_node.pos, grid):
                if child in closed_f:
                    continue
                
                g = curr_node.g + c(curr_node.pos, child)

                child_node = open_f.get(child)
                if child_node is None:
                    open_f.enqueue(Node(child, g, h(goal, child), curr_node))
                elif g < child_node.g:
                    child_node.update(g, curr_node)
                    open_f.decrease_key(child_node)
        else:
            # Search backward from goal
            curr_node = open_b.dequeue()
            if curr_node.pos in closed_b:
                continue
            nodes_visited += 1
            closed_b[curr_node.pos] = curr_node
            
            # Check if we've met the forward search
            if curr_node.pos in closed_f:
                g_total = curr_node.g + closed_f[curr_node.pos].g
                if g_total < U:
                    U = g_total
                    meet_f = closed_f[curr_node.pos]
                    meet_b = curr_node
            
            for child in get_children(curr_node.pos, grid):
                if child in closed_b:
                    continue
                
                g = curr_node.g + c(curr_node.pos, child)

                child_node = open_b.get(child)
                if child_node is None:
                    open_b.enqueue(Node(child, g, h(start, child), curr_node))
                elif g < child_node.g:
                    child_node.update(g, curr_node)
                    open_b.decrease_key(child_node)

    if U < float('inf') and meet_f and meet_b:
        return reconstruct_path(meet_f) + reconstruct_path(meet_b)[::-1][1:], nodes_visited
    return None, nodes_visited

def MM(start: tuple, goal: tuple, board: list, epsilon: int = 0) -> tuple:
    """Meet-in-the-Middle A* Search Algorithm"""
    grid = Grid(board)
    if grid[start] != 0 or grid[goal] != 0:
        return None, 0  # Start or goal is blocked

    open_f, open_b = TieBreakingPQ(), TieBreakingPQ()
    open_f.enqueue(Node(start, 0, h(start, goal)))
    open_b.enqueue(Node(goal, 0, h(goal, start)))
    
    closed_f, closed_b = {}, {}

    U = float('inf')
    meet_f = None
    meet_b = None

    nodes_visited = 0
    
    def gmin(q: PriorityQueue) -> int:
        return min(node.g for node in q.nodes.values()) if not q.is_empty() else float('inf')

    while not open_f.is_empty() and not open_b.is_empty():
        gmin_f = gmin(open_f)
        gmin_b = gmin(open_b)
        C = min(gmin_f, gmin_b)
        if U <= max(C, open_f.peek().f, open_b.peek().f, gmin_f + gmin_b + epsilon):
            break

        if C == gmin_f:
            curr_node = open_f.dequeue()
            if curr_node.pos in closed_f or curr_node.f >= U:
                continue
            nodes_visited += 1
            closed_f[curr_node.pos] = curr_node
            if curr_node.pos in closed_b:
                g_total = curr_node.g + closed_b[curr_node.pos].g
                if g_total < U:
                    U = g_total
                    meet_f = curr_node
                    meet_b = closed_b[curr_node.pos]
            for child in get_children(curr_node.pos, grid):
                if child in closed_f:
                    continue
                g = curr_node.g + c(curr_node.pos, child)
                child_node = open_f.get(child)
                if child_node is None:
                    open_f.enqueue(Node(child, g, h(goal, child), curr_node))
                elif g < child_node.g:
                    child_node.update(g, curr_node)
                    open_f.decrease_key(child_node)
        else:
            curr_node = open_b.dequeue()
            if curr_node.pos in closed_b or curr_node.f >= U:
                continue
            nodes_visited += 1
            closed_b[curr_node.pos] = curr_node
            if curr_node.pos in closed_f:
                g_total = curr_node.g + closed_f[curr_node.pos].g
                if g_total < U:
                    U = g_total
                    meet_f = closed_f[curr_node.pos]
                    meet_b = curr_node
            for child in get_children(curr_node.pos, grid):
                if child in closed_b:
                    continue
                g = curr_node.g + c(curr_node.pos, child)
                child_node = open_b.get(child)
                if child_node is None:
                    open_b.enqueue(Node(child, g, h(start, child), curr_node))
                elif g < child_node.g:
                    child_node.update(g, curr_node)
                    open_b.decrease_key(child_node)

    if U < float('inf') and meet_f and meet_b:
        return reconstruct_path(meet_f) + reconstruct_path(meet_b)[::-1][1:], nodes_visited
    return None, nodes_visited

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
        path, nodes = func(start, goal, board)
        print(f"Path found by {func.__name__}: {path}, Nodes visited: {nodes}")
