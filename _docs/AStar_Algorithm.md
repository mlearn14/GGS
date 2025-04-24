# A* Pathfinding: Theory and Oceanographic Implementation

## 1. General A* Algorithm

A* is a best-first graph search that finds the least-cost path from a start node to a goal node by combining:

- **g(n)**: the exact cost to reach node **n** from the start
- **h(n)**: a heuristic estimate of the cost to go from **n** to the goal

It prioritizes nodes by their **f-score**:

``` math
f(n) = g(n) + h(n)
```

Where:

- `g(n)` is known exactly
- `h(n)` must be admissible (never overestimates the true cost) to guarantee optimality

### Pseudocode (Generic A*)

```text
function A_Star(start, goal):
    open_set ← {start}
    came_from ← empty map
    g_score[start] ← 0
    f_score[start] ← h(start)

    while open_set is not empty:
        current ← node in open_set with lowest f_score
        if current == goal:
            return ReconstructPath(came_from, current)

        remove current from open_set
        add current to closed_set

        for each neighbor of current:
            tentative_g ← g_score[current] + cost(current, neighbor)
            if neighbor in closed_set and tentative_g ≥ g_score[neighbor]:
                continue

            if neighbor not in open_set or tentative_g < g_score[neighbor]:
                came_from[neighbor] ← current
                g_score[neighbor] ← tentative_g
                f_score[neighbor] ← tentative_g + h(neighbor)
                if neighbor not in open_set:
                    add neighbor to open_set

    return failure  // no path found

function ReconstructPath(came_from, current):
    total_path ← [current]
    while current in came_from:
        current ← came_from[current]
        prepend current to total_path
    return total_path
```

### `A_Star` function

The **`A_star`** function takes two imputs:

- `start`: the starting point of the path
- `goal`: the destination point of the path

The function returns the shortest path from `start` to `goal` as a sequence of nodes.

1. **Initialization:**
   - Creates an `open_set` containing only the `start` node. This set will store nodes to be explored.
   - Creates an empty `came_from` map to store the node that each node came from.
   - Sets the `g_score` (cost from start to node) and `f_score` (estimated total cost to goal) for the `start` node.
2. **Main Loop:**
    - While there are nodes in the `open_set`:
      - Select the node with the lowest `f_score` (i.e., the node with the lowest estimated total cost to the goal).
      - If this node is the `goal`, return the reconstructed path using the `ReconstructPath` function.
      - Remove the current node from the `open_set` and add it to the `closed_set` (a set of nodes that have already been explored).
      - For each neighbor of the current node:
        - Calculate the tentative `g_score` (cost from start to neighbor) by adding the cost of moving from the current node to the neighbor.
        - If the neighbor is already in the `closed_set` and the tentative `g_score` is not better than the existing `g_score`, skip it.
        - If the neighbor is not in the `open_set` or the tentative `g_score` is better than the existing `g_score`, update the `came_from` map, `g_score`, and `f_score` for the neighbor.
        - Add the neighbors to the `open_set` if it's not already there.
3. **Failure:**
    - If the `open_set` is empty and no path has been found, return a failure indicator.

### `ReconstructPath` function

The **`ReconstructPath`** function takes two inputs:

- `come_from`: the map of nodes that each node came from
- `current`: the current node

The function returns the reconstructed path from the `start` node to the `current` node.

1. **Initialization:**
    - Creates an empty `total_path` list to store the reconstructed path.
2. **Backtracking:**
    - While the `current` node is in the `came_from` map:
      - Add the `current` node to the `total_path` list.
      - Move to the node that the `current` node came from (i.e. the node in the `came_from` map).
3. **Return:**
   - Return the reconstructed `total_path` list.

## 2. Oceanographic A* (Current-Aware for Glider Missions)

This implementation adapts A* for underwater glider navigation using global ocean current models.

### 2.1. Graph Representation

- **Nodes**: Grid cells `(i, j)` representing lat/lon indices
- **Edges**: Up to 8-way connectivity (including diagonals), filtered by:
  - Not masked as land
  - Haversine step distance ≤ `max_leg_distance_m` (e.g., 50–60 km)

### 2.2. Cost So Far — `g(n)`

Calculated using the `calculate_movement()` function:

1. Convert grid indices to coordinates
2. Compute the great-circle distance:

   ``` math
   d = 2 * R * arcsin(\sqrt{sin²(Δφ/2) + cos(φ₁) * cos(φ₂) * sin²(Δλ/2)})
   ```

3. Compute heading vector and project the ocean current vector `(u, v)` onto it
4. Add to glider speed:

   ``` python
   v_net = clamp(v_glider + dot(current_vector, heading_vector))
   ```

5. Time is distance over net speed; fall back to glider speed if current data is missing

### 2.3. Heuristic — `h(n)`

Provided by `calculate_drift_aware_heuristic_cost()`:

- Base guess: straight-line time (distance over glider speed)
- Improved guess: uses the same vector projection logic, estimating net speed toward the goal

### 2.4. Weighted A*

The final A* uses a weighted heuristic:

``` math
f(n) = g(n) + W * h(n)
```

- `W = 1.0` → classic A* (optimal)
- `W > 1.0` → more greedy, faster with near-optimal paths (e.g. `W = 1.2`)

### 2.5. Performance Optimizations

| Feature                     | Description |
|-----------------------------|-------------|
| **Land Masking**           | Based on NaNs in u/v, blocks land nodes |
| **Leg Distance Filter**    | Avoids large, unrealistic jumps |
| **Visited Set**            | Avoids redundant node evaluations |
| **`open_set_hash`**        | Speeds up membership checks |
| **u/v Field Caching**      | Preloads `ds.u.values`, `ds.v.values` to avoid xarray overhead |
| **Spatial Subsetting**     | Clips dataset to bounding box near waypoints |

### 2.6. Custom A* Pseudocode (Glider-Aware)

```text
# Precompute:
ds ← ensure_land_mask(ds)
ds ← subset_around_waypoints(ds)
u_array, v_array ← ds.u.values, ds.v.values

# A* Init:
g[start] ← 0
f[start] ← h(start, goal)
open_set ← min-heap by f
open_hash ← set for fast lookups
came_from ← {}
visited ← set()

while open_set not empty:
    current ← node in open_set with lowest f
    if current == goal:
        return ReconstructPath(came_from, current)

    visited.add(current)

    for neighbor in generate_neighbors(current):
        if neighbor in visited:
            continue

        cost ← movement_time(current, neighbor, u_array, v_array)
        tentative_g ← g[current] + cost

        if neighbor not in g or tentative_g < g[neighbor]:
            came_from[neighbor] ← current
            g[neighbor] ← tentative_g
            f[neighbor] ← tentative_g + W * h(neighbor)
            add neighbor to open_set and open_hash if not present

return fallback_direct_path()
```
