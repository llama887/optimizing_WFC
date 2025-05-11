import numpy as np
from .utils import calc_num_regions

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

    # Calculate basic ratios
    water_ratio = water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0

    # Check for flow patterns
    has_horizontal_flow = check_river_flow(grid, water_tiles, "horizontal")
    has_vertical_flow = check_river_flow(grid, water_tiles, "vertical")
    has_flow = has_horizontal_flow or has_vertical_flow

    # Calculate river-specific metrics
    regions = calc_num_regions(water_map.astype(np.int8))
    linearity = calculate_linearity(water_map) if has_flow else 0
    meander_score = calculate_meander_score(water_map) if has_flow else 0
    width_consistency = calculate_width_consistency(water_map) if has_flow else 0

    # Ideal parameters
    IDEAL_WATER_RATIO = 0.25  # Rivers should cover about 25% of the map
    IDEAL_SHORE_RATIO = 0.2   # About 20% of water tiles should be shores
    IDEAL_REGIONS = 1         # Single connected water region
    IDEAL_LINEARITY = 0.7     # Rivers should be somewhat linear
    IDEAL_MEANDER = 0.4       # Some gentle meandering is good
    IDEAL_WIDTH = 0.8         # Consistent width is good for rivers

    # Calculate rewards and penalties
    reward = 0
    
    # Base flow requirement (must have flow to be a river)
    if not has_flow:
        return -float('inf'), {"message": "No continuous flow detected"}
    
    # Water ratio scoring (bell curve around ideal)
    water_ratio_diff = abs(water_ratio - IDEAL_WATER_RATIO)
    reward += max(0, 100 - (water_ratio_diff * 400))

    # Shore ratio scoring
    shore_diff = abs(shore_ratio - IDEAL_SHORE_RATIO)
    reward += max(0, 50 - (shore_diff * 250))

    # Region scoring (strong penalty for multiple regions)
    reward += (1 - min(regions, 5)) * 50  # 50 points for single region, decreasing

    # River shape characteristics
    if has_flow:
        reward += linearity * 30 * IDEAL_LINEARITY
        reward += meander_score * 20 * (1 - abs(IDEAL_MEANDER - meander_score))
        reward += width_consistency * 30 * IDEAL_WIDTH
        
        # Bonus for having both directions (river delta)
        if has_horizontal_flow and has_vertical_flow:
            reward += 30

    # Ensure reward isn't negative
    reward = max(reward, 0)

    return reward, {
        "flow": has_flow,
        "regions": regions,
        "water_ratio": water_ratio,
        "shore_ratio": shore_ratio,
        "linearity": linearity,
        "meander": meander_score,
        "width_consistency": width_consistency,
        "reward": reward
    }

def calculate_linearity(water_map: np.ndarray) -> float:
    """Calculate how straight the river is (0-1)"""
    # Find all water cells
    y, x = np.where(water_map)
    if len(x) < 2:
        return 0
    
    # Perform linear regression
    slope, intercept = np.polyfit(x, y, 1)
    predicted_y = slope * x + intercept
    errors = np.abs(y - predicted_y)
    
    # Normalize error to 0-1 range
    max_error = np.max(errors) if np.max(errors) > 0 else 1
    linearity = 1 - (np.mean(errors) / max_error)
    return linearity

def calculate_meander_score(water_map: np.ndarray) -> float:
    """Calculate how much the river meanders (0-1)"""
    # Find the main path (simplified)
    y, x = np.where(water_map)
    if len(x) < 3:
        return 0
    
    # Calculate the sinuosity (actual path length vs straight-line distance)
    path_length = len(x)
    straight_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    sinuosity = path_length / straight_distance if straight_distance > 0 else 1
    
    # Normalize to 0-1 range (1 being very meandering)
    return min(sinuosity / 1.5, 1)  # Cap at 1 even for very meandering rivers

def calculate_width_consistency(water_map: np.ndarray) -> float:
    """Calculate how consistent the river width is (0-1)"""
    # For each column (or row), count the water cells
    widths = np.sum(water_map, axis=0)
    non_zero_widths = widths[widths > 0]
    
    if len(non_zero_widths) < 2:
        return 0
    
    # Calculate coefficient of variation (lower is more consistent)
    cv = np.std(non_zero_widths) / np.mean(non_zero_widths)
    return 1 - min(cv, 1)  # Convert to 0-1 scale where 1 is most consistent

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