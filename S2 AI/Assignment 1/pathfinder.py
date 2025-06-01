import sys
import math
from queue import PriorityQueue
from collections import deque

STUDENT_ID = 'a1880714'
DEGREE = 'UG'

# Function to read the map from a file
def read_map(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    size = tuple(map(int, lines[0].strip().split()))  # Read the size of the map
    start = tuple(map(int, lines[1].strip().split()))  # Read the start position
    end = tuple(map(int, lines[2].strip().split()))  # Read the end position
    grid = []
    for line in lines[3:]:  # Read the grid
        row = []
        for char in line.strip().split():
            if char == 'X':
                row.append(char)  # Append obstacles as 'X'
            else:
                row.append(int(char))  # Append elevation values as integers
        grid.append(row)
    start = (start[0] - 1, start[1] - 1)  # Adjust start position to 0-based index
    end = (end[0] - 1, end[1] - 1)  # Adjust end position to 0-based index
    return size, start, end, grid

# Function to perform Breadth-First Search (BFS)
def bfs(start, end, grid, mode):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start])])  # Initialize queue with start position and path
    closed = set()  # Set of closed nodes
    visits = [[0] * cols for _ in range(rows)]  # Matrix to count visits to each cell
    first_visit = [[None] * cols for _ in range(rows)]  # Matrix to record first visit order
    last_visit = [[None] * cols for _ in range(rows)]  # Matrix to record last visit order
    visit_count = 1
    first_visit[start[0]][start[1]] = visit_count
    last_visit[start[0]][start[1]] = visit_count

    while queue:
        current, path = queue.popleft()  # Dequeue the front element
        visits[current[0]][current[1]] += 1
        if current == end:
            return path, visits, first_visit, last_visit  # Return path if end is reached

        if current not in closed:
            closed.add(current)
            for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Explore neighbors
                next_pos = (current[0] + direction[0], current[1] + direction[1])
                if (0 <= next_pos[0] < rows and 0 <= next_pos[1] < cols and
                        next_pos not in closed and grid[next_pos[0]][next_pos[1]] != 'X'):
                    queue.append((next_pos, path + [next_pos]))  # Enqueue valid neighbors
                    visit_count += 1
                    if first_visit[next_pos[0]][next_pos[1]] is None:
                        first_visit[next_pos[0]][next_pos[1]] = visit_count
                    last_visit[next_pos[0]][next_pos[1]] = visit_count

    return None, visits, first_visit, last_visit  # Return None if no path is found

# Function to perform Uniform Cost Search (UCS)
def ucs(start, end, grid, mode):
    rows, cols = len(grid), len(grid[0])
    pq = PriorityQueue()
    counter = 0  # Initialize counter to track enqueue order
    pq.put((0, counter, start, [start]))  # (cost, counter, position, path)
    g_costs = {start: 0}
    visits = [[0] * cols for _ in range(rows)]
    first_visit = [[None] * cols for _ in range(rows)]
    last_visit = [[None] * cols for _ in range(rows)]
    visit_count = 1
    first_visit[start[0]][start[1]] = visit_count
    last_visit[start[0]][start[1]] = visit_count

    while not pq.empty():
        cost, _, current, path = pq.get()  # Ignore counter on dequeue
        visits[current[0]][current[1]] += 1
        if current == end:
            return path, visits, first_visit, last_visit

        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, down, left, right
            next_pos = (current[0] + direction[0], current[1] + direction[1])
            if (0 <= next_pos[0] < rows and 0 <= next_pos[1] < cols and
                    grid[next_pos[0]][next_pos[1]] != 'X'):
                elevation_diff = grid[next_pos[0]][next_pos[1]] - grid[current[0]][current[1]]
                move_cost = 1 + max(elevation_diff, 0)
                new_g_cost = cost + move_cost
                if next_pos not in g_costs or new_g_cost < g_costs[next_pos]:
                    g_costs[next_pos] = new_g_cost
                    counter += 1  # Increment counter for each enqueue
                    pq.put((new_g_cost, counter, next_pos, path + [next_pos]))
                    visit_count += 1
                    if first_visit[next_pos[0]][next_pos[1]] is None:
                        first_visit[next_pos[0]][next_pos[1]] = visit_count
                    last_visit[next_pos[0]][next_pos[1]] = visit_count

    return None, visits, first_visit, last_visit


# Function to perform A* Search
def astar(start, end, grid, heuristic, mode):
    rows, cols = len(grid), len(grid[0])
    pq = PriorityQueue()
    counter = 0  # Initialize counter to track enqueue order
    pq.put((0, counter, 0, start, [start]))  # (f_cost, counter, g_cost, position, path)
    g_costs = {start: 0}
    visits = [[0] * cols for _ in range(rows)]
    first_visit = [[None] * cols for _ in range(rows)]
    last_visit = [[None] * cols for _ in range(rows)]
    visit_count = 1
    first_visit[start[0]][start[1]] = visit_count
    last_visit[start[0]][start[1]] = visit_count

    while not pq.empty():
        _, _, g_cost, current, path = pq.get()  # Ignore f_cost and counter on dequeue
        visits[current[0]][current[1]] += 1
        if current == end:
            return path, visits, first_visit, last_visit

        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, down, left, right
            next_pos = (current[0] + direction[0], current[1] + direction[1])
            if (0 <= next_pos[0] < rows and 0 <= next_pos[1] < cols and
                    grid[next_pos[0]][next_pos[1]] != 'X'):
                elevation_diff = grid[next_pos[0]][next_pos[1]] - grid[current[0]][current[1]]
                move_cost = 1 + max(elevation_diff, 0)
                new_g_cost = g_cost + move_cost
                if next_pos not in g_costs or new_g_cost < g_costs[next_pos]:
                    g_costs[next_pos] = new_g_cost
                    f_cost = new_g_cost + heuristic(next_pos, end)
                    counter += 1  # Increment counter for each enqueue
                    pq.put((f_cost, counter, new_g_cost, next_pos, path + [next_pos]))
                    visit_count += 1
                    if first_visit[next_pos[0]][next_pos[1]] is None:
                        first_visit[next_pos[0]][next_pos[1]] = visit_count
                    last_visit[next_pos[0]][next_pos[1]] = visit_count

    return None, visits, first_visit, last_visit


# Heuristic function for Euclidean distance
def euclidean_heuristic(current, goal):
    return math.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)

# Heuristic function for Manhattan distance
def manhattan_heuristic(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])


# Function to print debug output
def print_debug_output(path, grid, visits, first_visit, last_visit):
    if path is None:
        print("path:\nnull")
        print("#visits:\n...")
        print("first visit:\n...")
        print("last visit:\n...")
    else:
        path_grid = [row[:] for row in grid]
        for (i, j) in path:
            path_grid[i][j] = '*'

        print("path:")
        for row in path_grid:
            print(' '.join(str(cell) if cell != 'X' else 'X' for cell in row))

        print("#visits:")
        for row in visits:
            print(' '.join(str(cell) if cell != 0 else '.' for cell in row))

        print("first visit:")
        for row in first_visit:
            print(' '.join(str(cell) if cell is not None else '.' for cell in row))

        print("last visit:")
        for row in last_visit:
            print(' '.join(str(cell) if cell is not None else '.' for cell in row))

# Function to print release output
def print_release_output(path, grid):
    if path is None:
        print("null")
    else:
        path_grid = [row[:] for row in grid]  # Create a copy of the grid
        for (i, j) in path:
            path_grid[i][j] = '*'  # Mark the path with '*'
        
        # Print the grid with the path marked
        for row in path_grid:
            print(' '.join(str(cell) if cell != 'X' else 'X' for cell in row))

# Main function to run the program
def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python pathfinder.py [mode] [map] [algorithm] [heuristic]")
        return

    mode = sys.argv[1]
    map_file = sys.argv[2]
    algorithm = sys.argv[3]
    heuristic = sys.argv[4] if len(sys.argv) == 5 else None

    size, start, end, grid = read_map(map_file)

    if algorithm == 'bfs':
        path, visits, first_visit, last_visit = bfs(start, end, grid, mode)
    elif algorithm == 'ucs':
        path, visits, first_visit, last_visit = ucs(start, end, grid, mode)
    elif algorithm == 'astar':
        if heuristic == 'euclidean':
            path, visits, first_visit, last_visit = astar(start, end, grid, euclidean_heuristic, mode)
        elif heuristic == 'manhattan':
            path, visits, first_visit, last_visit = astar(start, end, grid, manhattan_heuristic, mode)
        else:
            print("Invalid heuristic")
            return
    else:
        print("Invalid algorithm")
        return

    if mode == 'debug':
        # Print debug output
        print_debug_output(path, grid, visits, first_visit, last_visit)
    elif mode == 'release':
        # Print release output
        print_release_output(path, grid)
    else:
        print("Invalid mode")

if __name__ == "__main__":
    main()