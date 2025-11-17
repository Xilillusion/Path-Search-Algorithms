from AStar import A_star, Bi_HS, MM
from BFS import BFS, Bi_BFS
from maze_generator import *

def show_statistics(stat: dict, x_axis: list = None):
    import matplotlib.pyplot as plt
    algorithms = list(stat.keys())
    points = len(next(iter(stat.values()), []))
    totals = [
        sum(stat[algo][i] for algo in algorithms)
        for i in range(points)
    ]
    xs = list(x_axis) if x_axis is not None else list(range(points))
    for algo in algorithms:
        percents = [
            (stat[algo][i] / totals[i] * 100) if totals[i] else 0
            for i in range(points)
        ]
        plt.plot(xs, percents, label=algo)
    plt.xlabel("Width")
    plt.ylabel("Visited nodes (%)")
    plt.legend()
    plt.show()

def main():
    width = range(11, 102, 10)
    height = range(11, 102, 10)
    n_mazes = 50

    start = (0, 0)
    stat = {
        "BFS": [],
        "Bi_BFS": [],
        "A_star": [],
        "Bi_HS": [],
        "MM": []
    }

    for w, h in zip(width, height):
        end = (w-1, h-1)
        for s in stat.values():
            s.append(0)  # Initialize count for this size

        for gen in [dfs_maze, kruskal_maze, prim_maze, aldous_broder_maze]:
            for _ in range(n_mazes):
                maze = gen(w, h, start, end)
                break_wall(maze)

                optimal = 0

                for algo in [BFS, Bi_BFS, A_star, Bi_HS, MM]:
                    maze_copy = maze.copy()
                    path, visited = algo(start, end, maze_copy)

                    stat[algo.__name__][-1] += visited

                    # Check path optimality
                    if optimal == 0:
                        optimal = len(path)
                    else:
                        if optimal < len(path):
                            print(f"Warning: Longer path lengths found by {algo.__name__}!")
                        elif optimal > len(path):
                            print(f"Warning: Shorter path lengths found by {algo.__name__}!")
        
        print(f"Maze Size: {w}x{h}")
        for algo in stat.keys():
            # Average over all mazes
            avg_nodes = stat[algo][-1] / (n_mazes * len(stat))
            stat[algo][-1] = avg_nodes
            print(f"{algo} Avg Nodes Visited: {round(avg_nodes)}")
        print()
    
    show_statistics(stat, x_axis=width)

if __name__ == "__main__":
    main()