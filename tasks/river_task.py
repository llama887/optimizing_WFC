import numpy as np
from .utils import calc_num_regions
from collections import deque

def get_river_biome(grid: list[list[set[str]]]) -> str:
    water_tiles = {
        "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
        "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
        "shore_bl", "shore_br", "shore_lr", "shore_rl"
    }

    water_cells = 0
    shore_cells = 0
    for row in grid:
        for cell in row:
            if len(cell) == 1:
                tile = next(iter(cell)).lower()
                if tile in water_tiles:
                    water_cells += 1
                    if "shore" in tile:
                        shore_cells += 1

    total_cells = len(grid) * len(grid[0])
    if total_cells == 0:
        return "unknown"

    water_ratio = water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0

    has_flow = check_river_flow(grid, water_tiles, "horizontal") or \
               check_river_flow(grid, water_tiles, "vertical")

    if has_flow and 0.2 <= water_ratio <= 0.4 and shore_ratio <= 0.3:
        return "river"
    return "unknown"

def check_river_flow(
    grid: list[list[set[str]]], water_tiles: set[str], direction: str
) -> bool:
    if direction == "horizontal":
        for y in range(len(grid)):
            if has_water_path(grid, (0, y), (len(grid[0]) - 1, y), water_tiles):
                return True
    else:
        for x in range(len(grid[0])):
            if has_water_path(grid, (x, 0), (x, len(grid) - 1), water_tiles):
                return True
    return False

def river_reward(grid: list[list[set[str]]]) -> tuple[float, dict]:
    water_tiles = {
        "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
        "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
        "shore_bl", "shore_br", "shore_lr", "shore_rl"
    }

    # Create water map and count cells
    water_map = np.zeros((len(grid), len(grid[0])), dtype=bool)
    water_cells = 0
    shore_cells = 0
    
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if len(grid[y][x]) == 1:
                tile = next(iter(grid[y][x])).lower()
                if tile in water_tiles:
                    water_map[y, x] = True
                    water_cells += 1
                    if "shore" in tile:
                        shore_cells += 1

    total_cells = len(grid) * len(grid[0])
    if total_cells == 0:
        return -float('inf'), {}

    water_ratio = water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0
    regions = calc_num_regions(water_map.astype(np.int8))

    # Calculate river metrics
    river_length = calculate_river_length_simple(water_map)
    norm_length = river_length / max(len(grid), len(grid[0]))
    aspect_ratio = calculate_aspect_ratio(water_map)

    # Ideal parameters
    IDEAL_WATER_RATIO_MIN = 0.2
    IDEAL_WATER_RATIO_MAX = 0.4
    IDEAL_SHORE_RATIO = 0.3
    MIN_LENGTH_RATIO = 0.6  # River should span at least 60% of the longer grid dimension
    MIN_ASPECT_RATIO = 3.0  # Minimum width/height ratio to be considered river-like

    # Penalties and bonuses
    region_penalty = (regions - 1) * -200  # Heavily penalize multiple regions
    
    if water_ratio < IDEAL_WATER_RATIO_MIN:
        water_penalty = (IDEAL_WATER_RATIO_MIN - water_ratio) * -200
    elif water_ratio > IDEAL_WATER_RATIO_MAX:
        water_penalty = (water_ratio - IDEAL_WATER_RATIO_MAX) * -200
    else:
        water_penalty = 0

    shore_penalty = max(0, (shore_ratio - IDEAL_SHORE_RATIO)) * -100
    aspect_penalty = -100 if aspect_ratio < MIN_ASPECT_RATIO else 0
    
    # Length rewards
    length_bonus = 0
    if norm_length >= MIN_LENGTH_RATIO:
        length_bonus = 100 * (norm_length - MIN_LENGTH_RATIO) / (1 - MIN_LENGTH_RATIO)
    
    # Bonus for connecting opposite sides
    connects_sides = check_connects_opposite_sides(water_map)
    connection_bonus = 50 if connects_sides else 0

    total_reward = (
        region_penalty + water_penalty + shore_penalty + 
        aspect_penalty + length_bonus + connection_bonus
    )

    return min(max(total_reward, -200), 200), {
        "regions": regions,
        "water_ratio": water_ratio,
        "shore_ratio": shore_ratio,
        "river_length": river_length,
        "norm_length": norm_length,
        "aspect_ratio": aspect_ratio,
        "connects_sides": connects_sides,
        "region_penalty": region_penalty,
        "water_penalty": water_penalty,
        "shore_penalty": shore_penalty,
        "aspect_penalty": aspect_penalty,
        "length_bonus": length_bonus,
        "connection_bonus": connection_bonus,
        "reward": total_reward
    }

def calculate_river_length_simple(water_map: np.ndarray) -> int:
    """Calculate river length using a simple BFS approach."""
    if not water_map.any():
        return 0
    
    # Find all water cells
    water_cells = np.argwhere(water_map)
    
    # If only one cell, length is 1
    if len(water_cells) == 1:
        return 1
    
    # Find maximum distance between any two water cells
    max_distance = 0
    for i in range(len(water_cells)):
        for j in range(i+1, len(water_cells)):
            dist = np.linalg.norm(water_cells[i] - water_cells[j])
            if dist > max_distance:
                max_distance = dist
    
    return int(max_distance)

def calculate_aspect_ratio(water_map: np.ndarray) -> float:
    """Calculate width/height ratio of water area."""
    if not water_map.any():
        return 0
    
    rows = np.any(water_map, axis=1)
    cols = np.any(water_map, axis=0)
    height = np.sum(rows)
    width = np.sum(cols)
    
    if height == 0:
        return float('inf')
    return width / height

def check_connects_opposite_sides(water_map: np.ndarray) -> bool:
    """Check if water connects two opposite sides of the grid."""
    height, width = water_map.shape
    
    # Check left to right connection
    left_edge = [(0, y) for y in range(height) if water_map[y, 0]]
    right_edge = [(width-1, y) for y in range(height) if water_map[y, width-1]]
    
    if left_edge and right_edge:
        for start in left_edge:
            for end in right_edge:
                if has_path(water_map, start, end):
                    return True
    
    # Check top to bottom connection
    top_edge = [(x, 0) for x in range(width) if water_map[0, x]]
    bottom_edge = [(x, height-1) for x in range(width) if water_map[height-1, x]]
    
    if top_edge and bottom_edge:
        for start in top_edge:
            for end in bottom_edge:
                if has_path(water_map, start, end):
                    return True
    
    return False

def has_path(water_map: np.ndarray, start: tuple, end: tuple) -> bool:
    """Check if there's a path between two points in the water map."""
    if not water_map[start[1], start[0]] or not water_map[end[1], end[0]]:
        return False
    
    visited = np.zeros_like(water_map, dtype=bool)
    queue = deque([start])
    visited[start[1], start[0]] = True
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    height, width = water_map.shape
    
    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            return True
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if water_map[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))
    
    return False