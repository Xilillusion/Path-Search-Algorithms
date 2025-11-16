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


def get_children(position: tuple, grid: Grid) -> list:
    """Get valid neighboring positions (up, down, left, right)"""
    neighbors = []
    x, y = position
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.x and 0 <= ny < grid.y and grid[(nx, ny)] == 0:
            neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(visited: dict, current: tuple) -> list:
    """Reconstruct path from start to goal"""
    path = []
    while current is not None:
        path.append(current)
        current = visited.get(current, None)
    return path[::-1]

def BFS(start: tuple, goal: tuple, board: list) -> list:
    """Breadth-First Search Algorithm"""
    grid = Grid(board)
    if grid[start] != 0 or grid[goal] != 0:
        return None  # Start or goal is blocked

    queue = []
    queue.append(start)
    visited = {start: None}

    while queue:
        current = queue.pop(0)
        if current == goal:
            # Reconstruct path
            return reconstruct_path(visited, current)

        for child in get_children(current, grid):
            if child not in visited:
                visited[child] = current
                queue.append(child)

    return None

def Bi_BFS(start: tuple, goal: tuple, board: list) -> list:
    """Bi-Directional Breadth-First Search Algorithm"""
    grid = Grid(board)
    if grid[start] != 0 or grid[goal] != 0:
        return None  # Start or goal is blocked

    queue_f, queue_b = [], []
    queue_f.append(start)
    queue_b.append(goal)

    visited_f = {start: None}
    visited_b = {goal: None}

    while queue_f and queue_b:
        # Forward search
        current_f = queue_f.pop(0)
        for child in get_children(current_f, grid):
            if child not in visited_f:
                visited_f[child] = current_f
                queue_f.append(child)
                if child in visited_b:
                    return reconstruct_path(visited_f, child) + reconstruct_path(visited_b, child)[:-1][::-1]

        # Backward search
        current_b = queue_b.pop(0)
        for child in get_children(current_b, grid):
            if child not in visited_b:
                visited_b[child] = current_b
                queue_b.append(child)
                if child in visited_f:
                    return reconstruct_path(visited_f, child) + reconstruct_path(visited_b, child)[:-1][::-1]

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

    for func in [BFS, Bi_BFS]:
        path = func(start, goal, board)
        print(f"Path found by {func.__name__}: {path}")