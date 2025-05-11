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

    # Check all possible flow directions (including diagonals)
    has_flow = (
        check_river_flow(grid, water_tiles, "horizontal") or
        check_river_flow(grid, water_tiles, "vertical") or
        check_river_flow(grid, water_tiles, "diagonal_tl_br") or
        check_river_flow(grid, water_tiles, "diagonal_tr_bl")
    )

    # More lenient conditions for rivers
    if (has_flow and 
        0.1 <= water_ratio <= 0.6 and  # Expanded water ratio range
        shore_ratio <= 0.5 and          # More allowed shore tiles
        has_significant_flow(grid, water_tiles)):  # Additional check
        return "river"
    return "unknown"

def check_river_flow(
    grid: list[list[set[str]]], 
    water_tiles: set[str], 
    direction: str
) -> bool:
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    if direction == "horizontal":
        for y in range(height):
            if has_water_path(grid, (0, y), (width - 1, y), water_tiles):
                return True
    elif direction == "vertical":
        for x in range(width):
            if has_water_path(grid, (x, 0), (x, height - 1), water_tiles):
                return True
    elif direction == "diagonal_tl_br":  # Top-left to bottom-right
        for offset in range(-width + 1, height):
            if has_diagonal_path(grid, water_tiles, offset, "tl_br"):
                return True
    elif direction == "diagonal_tr_bl":  # Top-right to bottom-left
        for offset in range(-width + 1, height):
            if has_diagonal_path(grid, water_tiles, offset, "tr_bl"):
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
    river_length = calculate_river_length(water_map)
    norm_length = river_length / max(len(grid), len(grid[0]))
    aspect_ratio = calculate_aspect_ratio(water_map)
    compactness = calculate_compactness(water_map)
    branching_factor = calculate_branching_factor(water_map)

    # Ideal parameters
    IDEAL_WATER_RATIO_MIN = 0.15
    IDEAL_WATER_RATIO_MAX = 0.5
    IDEAL_SHORE_RATIO = 0.4
    MIN_LENGTH_RATIO = 0.5 
    MIN_ASPECT_RATIO = 2.0 
    MAX_COMPACTNESS = 0.3   # Maximum allowed compactness (0=line, 1=circle)
    MAX_BRANCHING = 2       # Maximum allowed branching points

    # Penalties (all negative)
    region_penalty = (regions - 1) * -100  # Penalty for multiple regions
    
    if water_ratio < IDEAL_WATER_RATIO_MIN:
        water_penalty = (IDEAL_WATER_RATIO_MIN - water_ratio) * -100
    elif water_ratio > IDEAL_WATER_RATIO_MAX:
        water_penalty = (water_ratio - IDEAL_WATER_RATIO_MAX) * -100
    else:
        water_penalty = 0

    shore_penalty = max(0, (shore_ratio - IDEAL_SHORE_RATIO)) * -50
    aspect_penalty = -50 if aspect_ratio < MIN_ASPECT_RATIO else 0
    compactness_penalty = max(0, (compactness - MAX_COMPACTNESS)) * -75
    branching_penalty = max(0, (branching_factor - MAX_BRANCHING)) * -25
    
    # Bonuses (positive but capped by penalties)
    length_bonus = 0
    if norm_length >= MIN_LENGTH_RATIO:
        length_bonus = 50 * (norm_length - MIN_LENGTH_RATIO) / (1 - MIN_LENGTH_RATIO)
    
    connects_sides = check_connects_opposite_sides(water_map)
    connection_bonus = 30 if connects_sides else 0

    straightness_bonus = 25 * (1 - compactness) if norm_length >= MIN_LENGTH_RATIO else 0

    # Calculate total reward (capped at 0)
    total_reward = min(
        region_penalty + water_penalty + shore_penalty + 
        aspect_penalty + compactness_penalty + branching_penalty +
        length_bonus + connection_bonus + straightness_bonus,
        0
    )

    return total_reward, {
        "regions": regions,
        "water_ratio": water_ratio,
        "shore_ratio": shore_ratio,
        "river_length": river_length,
        "norm_length": norm_length,
        "aspect_ratio": aspect_ratio,
        "compactness": compactness,
        "branching_factor": branching_factor,
        "connects_sides": connects_sides,
        "region_penalty": region_penalty,
        "water_penalty": water_penalty,
        "shore_penalty": shore_penalty,
        "aspect_penalty": aspect_penalty,
        "compactness_penalty": compactness_penalty,
        "branching_penalty": branching_penalty,
        "length_bonus": length_bonus,
        "connection_bonus": connection_bonus,
        "straightness_bonus": straightness_bonus,
        "reward": total_reward
    }

def calculate_compactness(water_map: np.ndarray) -> float:
    """Calculate how compact the water shape is (0 = line, 1 = circle)."""
    water_cells = np.sum(water_map)
    if water_cells < 2:
        return 0
    
    # Get bounding box
    rows = np.any(water_map, axis=1)
    cols = np.any(water_map, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
    
    return water_cells / bbox_area

def calculate_branching_factor(water_map: np.ndarray) -> int:
    """Count how many branching points exist in the water."""
    if not water_map.any():
        return 0
    
    branching_points = 0
    for y in range(1, len(water_map)-1):
        for x in range(1, len(water_map[0])-1):
            if water_map[y, x]:
                # Count adjacent water cells
                neighbors = (water_map[y-1, x] + water_map[y+1, x] + 
                            water_map[y, x-1] + water_map[y, x+1])
                if neighbors > 2:  # Junction point
                    branching_points += 1
    return branching_points

def calculate_river_length(water_map: np.ndarray) -> int:
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

def has_water_path(
    grid: list[list[set[str]]], start: tuple, end: tuple, water_tiles: set[str]
) -> bool:
    """Check if there's a continuous water path between two points."""
    from collections import deque

    water_map = np.zeros((len(grid), len(grid[0])), dtype=bool)
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if len(grid[y][x]) == 1 and next(iter(grid[y][x])).lower() in water_tiles:
                water_map[y, x] = True

    if not water_map[start[1], start[0]] or not water_map[end[1], end[0]]:
        return False

    visited = set()
    queue = deque([start])
    visited.add(start)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current = queue.popleft()
        if current == end:
            return True

        for dx, dy in directions:
            x, y = current[0] + dx, current[1] + dy
            if (0 <= x < len(grid[0]) and 0 <= y < len(grid) 
                and water_map[y, x] and (x, y) not in visited):
                visited.add((x, y))
                queue.append((x, y))

    return False

def has_diagonal_path(
    grid: list[list[set[str]]],
    water_tiles: set[str],
    offset: int,
    diagonal_type: str
) -> bool:
    height = len(grid)
    width = len(grid[0])
    min_continuous = min(width, height) * 0.5  # Require 50% continuous
    
    continuous = 0
    max_continuous = 0
    
    if diagonal_type == "tl_br":
        for y in range(height):
            x = y - offset
            if 0 <= x < width:
                if (len(grid[y][x]) == 1 and 
                    next(iter(grid[y][x])).lower() in water_tiles):
                    continuous += 1
                    max_continuous = max(max_continuous, continuous)
                else:
                    continuous = 0
    else:  # "tr_bl"
        for y in range(height):
            x = (width - 1) - (y - offset)
            if 0 <= x < width:
                if (len(grid[y][x]) == 1 and 
                    next(iter(grid[y][x])).lower() in water_tiles):
                    continuous += 1
                    max_continuous = max(max_continuous, continuous)
                else:
                    continuous = 0
                    
    return max_continuous >= min_continuous

def has_significant_flow(
    grid: list[list[set[str]]],
    water_tiles: set[str]
) -> bool:
    """Check if water forms a significant flow path with some width"""
    water_map = np.zeros((len(grid), len(grid[0])), dtype=bool)
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if len(grid[y][x]) == 1 and next(iter(grid[y][x])).lower() in water_tiles:
                water_map[y, x] = True
    
    # Check if water connects two opposite sides with some width
    height, width = water_map.shape
    
    # Check left-right connection
    left_side = water_map[:, 0]
    right_side = water_map[:, -1]
    if np.any(left_side) and np.any(right_side):
        for y1 in np.where(left_side)[0]:
            for y2 in np.where(right_side)[0]:
                if has_wide_path(water_map, (0, y1), (width-1, y2), min(width, height)//4):
                    return True
    
    # Check top-bottom connection
    top_side = water_map[0, :]
    bottom_side = water_map[-1, :]
    if np.any(top_side) and np.any(bottom_side):
        for x1 in np.where(top_side)[0]:
            for x2 in np.where(bottom_side)[0]:
                if has_wide_path(water_map, (x1, 0), (x2, height-1), min(width, height)//4):
                    return True
    
    # Check diagonal connections
    if (has_diagonal_flow(water_map, "tl_br") or 
        has_diagonal_flow(water_map, "tr_bl")):
        return True
        
    return False

def has_wide_path(
    water_map: np.ndarray,
    start: tuple,
    end: tuple,
    min_width: int
) -> bool:
    """Check if there's a path with at least min_width water cells"""
    # Simplified version - could be enhanced with proper width calculation
    # For now just check multiple parallel paths
    paths_found = 0
    for offset in range(-min_width//2, min_width//2 + 1):
        x1, y1 = start
        x2, y2 = end
        if 0 <= y1 + offset < water_map.shape[0]:
            if has_path(water_map, (x1, y1 + offset), (x2, y2 + offset)):
                paths_found += 1
        if paths_found >= min(2, min_width):
            return True
    return False

def has_diagonal_flow(water_map: np.ndarray, diagonal_type: str) -> bool:
    """Check for significant diagonal flow with some width"""
    height, width = water_map.shape
    min_continuous = min(width, height) * 0.6  # 60% of map length
    
    if diagonal_type == "tl_br":
        # Check top-left to bottom-right diagonal
        for offset in range(-width//2, width//2):
            continuous = 0
            max_continuous = 0
            for y in range(height):
                x = y - offset
                if 0 <= x < width and water_map[y, x]:
                    continuous += 1
                    max_continuous = max(max_continuous, continuous)
                else:
                    continuous = 0
            if max_continuous >= min_continuous:
                return True
    else:  # "tr_bl"
        # Check top-right to bottom-left diagonal
        for offset in range(-width//2, width//2):
            continuous = 0
            max_continuous = 0
            for y in range(height):
                x = (width - 1) - (y - offset)
                if 0 <= x < width and water_map[y, x]:
                    continuous += 1
                    max_continuous = max(max_continuous, continuous)
                else:
                    continuous = 0
            if max_continuous >= min_continuous:
                return True
    return False