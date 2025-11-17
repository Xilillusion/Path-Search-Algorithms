import random
import numpy as np

def break_wall(grid, prob=0.1):
    x, y = grid.shape
    for i in range(1, x):
        for j in range(1, y):
            if grid[i, j] == 1 and random.random() < prob:
                grid[i, j] = 0

def dfs_maze(x, y, start, end, complexity=0.5):
    x = (x // 2) * 2 + 3
    y = (y // 2) * 2 + 3
    grid = np.ones((x, y), dtype=np.int8)
    
    # Convert user coordinates to maze coordinates
    start_x = start[0] + 1
    start_y = start[1] + 1
    if start_x % 2 == 0:
        start_x += 1
    if start_y % 2 == 0:
        start_y += 1

    grid[start_x, start_y] = 0

    visited = np.zeros((x, y), dtype=bool)
    directions = np.array([[-2, 0], [2, 0], [0, -2], [0, 2]], dtype=np.int16)
    
    # DFS stack for backtracking
    stack = [(start_x, start_y)]
    visited[start_x, start_y] = True
    
    def get_unvisited(cx, cy):
        # Neighbor calculation
        current_pos = np.array([cx, cy])
        neighbors = current_pos + directions
        
        # Bounds checking
        valid_bounds = (
            (neighbors[:, 0] >= 1) & 
            (neighbors[:, 0] < x - 1) &
            (neighbors[:, 1] >= 1) & 
            (neighbors[:, 1] < y - 1)
        )
        neighbors = neighbors[valid_bounds]
        
        # Check if unvisited
        if len(neighbors) > 0:
            unvisited_mask = ~visited[neighbors[:, 0], neighbors[:, 1]]
            unvisited_neighbors = neighbors[unvisited_mask]
            return [tuple(pos) for pos in unvisited_neighbors]
        return []
    
    # DFS with backtracking and complexity control
    while stack:
        # Complexity controls which cell to pick from stack
        if random.random() < complexity and len(stack) > 1:
            # Pick a random cell from the stack for more branching
            stack_index = random.randint(0, len(stack) - 1)
            current_x, current_y = stack[stack_index]
            # Move picked cell to end of stack
            stack[stack_index], stack[-1] = stack[-1], stack[stack_index]
        else:
            current_x, current_y = stack[-1]
        
        neighbors = get_unvisited(current_x, current_y)
        
        if neighbors:
            next_x, next_y = random.choice(neighbors)
            
            # Remove wall between current and next cell
            wall_coords = np.array([current_x + next_x, current_y + next_y]) // 2
            grid[wall_coords[0], wall_coords[1]] = 0
            grid[next_x, next_y] = 0
            
            # Mark as visited and add to stack
            visited[next_x, next_y] = True
            stack.append((next_x, next_y))
        else:
            stack.pop()
    
    grid = grid[1:-1, 1:-1]
    grid[end] = 0
    
    return grid

def kruskal_maze(x, y, start, end):
    x = (x // 2) * 2 + 3
    y = (y // 2) * 2 + 3
    grid = np.ones((x, y), dtype=np.int8)
    
    # Mark all odd positions as potential cells
    i = np.arange(1, x, 2)
    j = np.arange(1, y, 2)

    cells = np.array(np.meshgrid(i, j, indexing='ij')).T.reshape(-1, 2)
    grid[np.ix_(i, j)] = 0

    # Create all possible walls between adjacent cells
    cx = np.arange(1, x, 2)
    cy = np.arange(1, y, 2)

    CX, CY = np.meshgrid(cx, cy, indexing='ij')
    CX = CX.ravel()
    CY = CY.ravel()

    walls = []

    # Right walls: (i,j)-(i+2,j) with wall at (i+1,j)
    mask_r = CX + 2 < x
    if np.any(mask_r):
        c1 = np.stack((CX[mask_r], CY[mask_r]), axis=1)
        c2 = c1 + np.array([2, 0])
        w  = c1 + np.array([1, 0])
        walls.extend(list(zip(map(tuple, c1), map(tuple, c2), map(tuple, w))))

    # Bottom walls: (i,j)-(i,j+2) with wall at (i,j+1)
    mask_b = CY + 2 < y
    if np.any(mask_b):
        c1 = np.stack((CX[mask_b], CY[mask_b]), axis=1)
        c2 = c1 + np.array([0, 2])
        w  = c1 + np.array([0, 1])
        walls.extend(list(zip(map(tuple, c1), map(tuple, c2), map(tuple, w))))

    random.shuffle(walls)
    
    parent = {tuple(cell): tuple(cell) for cell in cells}

    def find(a):
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return True
        return False

    # Remove walls that connect different sets
    for c1, c2, wall in walls:
        if union(c1, c2):
            grid[wall] = 0

    grid = grid[1:-1, 1:-1]
    grid[start] = 0
    grid[end] = 0
    
    return grid

def prim_maze(x, y, start, end):
    x = (x // 2) * 2 + 3
    y = (y // 2) * 2 + 3
    grid = np.ones((x, y), dtype=np.int8)
    
    # Convert start position to maze coordinates
    start_x = start[0] + 1
    start_y = start[1] + 1
    if start_x % 2 == 0:
        start_x += 1
    if start_y % 2 == 0:
        start_y += 1
    
    grid[start_x, start_y] = 0
    
    # Frontier list: potential walls to remove
    frontiers = []
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    
    def add_frontiers(cell_x, cell_y):
        for dx, dy in directions:
            nx, ny = cell_x + dx, cell_y + dy
            # Check if neighbor is within bounds and on odd coordinates (potential cell)
            if (1 <= nx < x - 1 and 1 <= ny < y - 1 and 
                nx % 2 == 1 and ny % 2 == 1 and grid[nx, ny] == 1):
                
                # Wall coordinates (between current and neighbor)
                wall_x = (cell_x + nx) // 2
                wall_y = (cell_y + ny) // 2
                
                frontier = (cell_x, cell_y, nx, ny, wall_x, wall_y)
                if frontier not in frontiers:
                    frontiers.append(frontier)
    
    add_frontiers(start_x, start_y)
    
    while frontiers:
        # Pick a random frontier
        frontier_idx = random.randint(0, len(frontiers) - 1)
        _, _, next_x, next_y, wall_x, wall_y = frontiers.pop(frontier_idx)
        
        # If the target cell is still a wall (unvisited)
        if grid[next_x, next_y] == 1:
            # Make the target cell and wall part of the maze
            grid[next_x, next_y] = 0
            grid[wall_x, wall_y] = 0
            
            add_frontiers(next_x, next_y)
    
    grid = grid[1:-1, 1:-1]
    grid[end] = 0
    
    return grid

def aldous_broder_maze(x, y, start, end):
    x = (x // 2) * 2 + 3
    y = (y // 2) * 2 + 3
    grid = np.ones((x, y), dtype=np.int8)
    
    # Convert user coordinates to maze coordinates
    start_x = start[0] + 1
    start_y = start[1] + 1
    if start_x % 2 == 0:
        start_x += 1
    if start_y % 2 == 0:
        start_y += 1

    grid[start_x, start_y] = 0

    visited = np.zeros((x, y), dtype=bool)
    visited[start_x, start_y] = True
    
    # Count total maze cells
    total_cells = ((x - 1) // 2) * ((y - 1) // 2)
    visited_count = 1
    
    current_x, current_y = start_x, start_y
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    
    while visited_count < total_cells:
        # Pick random direction
        dx, dy = random.choice(directions)
        next_x, next_y = current_x + dx, current_y + dy
        
        # Check if next position is within bounds and on valid cell coordinates
        if (1 <= next_x < x - 1 and 1 <= next_y < y - 1 and 
            next_x % 2 == 1 and next_y % 2 == 1):
            
            if not visited[next_x, next_y]:
                # Remove wall between current and next cell
                wall_x = (current_x + next_x) // 2
                wall_y = (current_y + next_y) // 2
                grid[wall_x, wall_y] = 0
                
                # Mark destination cell as visited and part of maze
                grid[next_x, next_y] = 0
                visited[next_x, next_y] = True
                visited_count += 1
            
            current_x, current_y = next_x, next_y
    
    grid = grid[1:-1, 1:-1]
    grid[end] = 0
    
    return grid

def print_maze(grid, title="Maze"):
    print(f"\n=== {title} ===")
    cell_symbols = {0: '  ', 1: '🟥', 2: '🟦'}
    for row in grid:
        print("".join(cell_symbols[cell] for cell in row))

# Example usage
if __name__ == "__main__":
    from AStar import A_star
    width, height = 31, 31
    start = (0, 0)
    end = (width-1, height-1)

    for gen in [dfs_maze, kruskal_maze, prim_maze, aldous_broder_maze]:
        maze = gen(width, height, start, end)
        break_wall(maze)
        path, _ = A_star(start, end, maze)
        if path:
            for step in path:
                maze[step[0]][step[1]] = 2  # Mark path in the maze
        print_maze(maze, title=f"{gen.__name__} Maze")