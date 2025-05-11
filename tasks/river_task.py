import numpy as np
from .utils import calc_num_regions

def get_river_biome(grid: list[list[set[str]]]) -> str:
    water_tiles = {
        "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
        "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
        "shore_bl", "shore_br", "shore_lr", "shore_rl"
    }

    # Additional river-specific tiles
    river_tiles = {
        "river", "river_t", "river_b", "river_l", "river_r",
        "river_tl", "river_tr", "river_bl", "river_br"
    }
    water_tiles.update(river_tiles)

    water_cells = 0
    shore_cells = 0
    river_cells = 0
    
    for row in grid:
        for cell in row:
            if len(cell) == 1:
                tile = next(iter(cell)).lower()
                if tile in water_tiles:
                    water_cells += 1
                    if "shore" in tile:
                        shore_cells += 1
                    if tile in river_tiles:
                        river_cells += 1

    total_cells = len(grid) * len(grid[0])
    if total_cells == 0:
        return "unknown"

    water_ratio = water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0
    river_ratio = river_cells / water_cells if water_cells > 0 else 0

    # Check for river flow patterns
    has_flow = check_river_flow(grid, water_tiles, "horizontal") or \
               check_river_flow(grid, water_tiles, "vertical")
    
    # Check for river source and mouth (narrow ends)
    has_source_mouth = check_river_source_mouth(grid, water_tiles)
    
    # Additional river constraints
    is_river = (
        has_flow and 
        has_source_mouth and
        0.15 <= water_ratio <= 0.4 and 
        shore_ratio <= 0.4 and
        river_ratio >= 0.2  # At least 20% of water tiles should be river-specific
    )

    if is_river:
        return "river"
    return "unknown"

def check_river_source_mouth(grid: list[list[set[str]]], water_tiles: set[str]) -> bool:
    """Check if the river has narrow ends (source and mouth)"""
    # Check horizontal flow pattern
    if check_river_flow(grid, water_tiles, "horizontal"):
        left_col = 0
        right_col = len(grid[0]) - 1
        left_water = sum(1 for y in range(len(grid)) 
                        if len(grid[y][left_col]) == 1 and 
                        next(iter(grid[y][left_col])).lower() in water_tiles)
        right_water = sum(1 for y in range(len(grid)) 
                         if len(grid[y][right_col]) == 1 and 
                         next(iter(grid[y][right_col])).lower() in water_tiles)
        
        # One end should be narrow (1-2 water tiles), the other can be wider
        return (left_water <= 2 or right_water <= 2)
    
    # Check vertical flow pattern
    elif check_river_flow(grid, water_tiles, "vertical"):
        top_row = 0
        bottom_row = len(grid) - 1
        top_water = sum(1 for x in range(len(grid[0])) 
                     if len(grid[top_row][x]) == 1 and 
                     next(iter(grid[top_row][x])).lower() in water_tiles)
        bottom_water = sum(1 for x in range(len(grid[0])) 
                      if len(grid[bottom_row][x]) == 1 and 
                      next(iter(grid[bottom_row][x])).lower() in water_tiles)
        
        return (top_water <= 2 or bottom_water <= 2)
    
    return False

def check_river_flow(
    grid: list[list[set[str]]], water_tiles: set[str], direction: str
) -> bool:
    """Check if there's a continuous water path from one side to another"""
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
        "shore_bl", "shore_br", "shore_lr", "shore_rl", "river", 
        "river_t", "river_b", "river_l", "river_r", "river_tl", 
        "river_tr", "river_bl", "river_br"
    }

    water_map = np.zeros((len(grid), len(grid[0])), dtype=bool)
    water_cells = 0
    shore_cells = 0
    river_cells = 0

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if len(grid[y][x]) == 1:
                tile = next(iter(grid[y][x])).lower()
                if tile in water_tiles:
                    water_map[y, x] = True
                    water_cells += 1
                    if "shore" in tile:
                        shore_cells += 1
                    if "river" in tile:
                        river_cells += 1

    total_cells = len(grid) * len(grid[0])
    if total_cells == 0:
        return -float('inf'), {}

    # Calculate basic ratios
    water_ratio = water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0
    river_ratio = river_cells / water_cells if water_cells > 0 else 0

    # Check for flow patterns
    has_horizontal_flow = check_river_flow(grid, water_tiles, "horizontal")
    has_vertical_flow = check_river_flow(grid, water_tiles, "vertical")
    has_flow = has_horizontal_flow or has_vertical_flow
    has_source_mouth = check_river_source_mouth(grid, water_tiles)

    regions = calc_num_regions(water_map.astype(np.int8))

    # Ideal parameters for a good river
    IDEAL_WATER_RATIO_MIN = 0.15
    IDEAL_WATER_RATIO_MAX = 0.4
    IDEAL_SHORE_RATIO = 0.3
    IDEAL_RIVER_RATIO = 0.3
    IDEAL_REGIONS = 1

    # Base penalties and bonuses
    flow_penalty = 0 if has_flow else -200
    source_mouth_penalty = 0 if has_source_mouth else -100

    # Water ratio penalty
    if water_ratio < IDEAL_WATER_RATIO_MIN:
        water_penalty = (IDEAL_WATER_RATIO_MIN - water_ratio) * -300
    elif water_ratio > IDEAL_WATER_RATIO_MAX:
        water_penalty = (water_ratio - IDEAL_WATER_RATIO_MAX) * -300
    else:
        water_penalty = 0

    # Shore ratio penalty
    shore_penalty = max(0, (shore_ratio - IDEAL_SHORE_RATIO)) * -150
    
    # River ratio bonus/penalty
    river_penalty = max(0, (IDEAL_RIVER_RATIO - river_ratio)) * -200
    
    # Region penalty
    region_penalty = abs(regions - IDEAL_REGIONS) * -100
    
    # Flow pattern bonuses
    flow_bonus = 50 if (has_horizontal_flow and has_vertical_flow) else 0
    source_mouth_bonus = 30 if has_source_mouth else 0
    river_tile_bonus = min(100, river_ratio * 200)  # Up to 100 bonus for more river tiles

    total_reward = (
        flow_penalty + 
        source_mouth_penalty + 
        water_penalty + 
        shore_penalty + 
        river_penalty + 
        region_penalty + 
        flow_bonus + 
        source_mouth_bonus + 
        river_tile_bonus
    )

    return min(total_reward, 100), {  # Cap reward at 100
        "flow": has_flow,
        "source_mouth": has_source_mouth,
        "regions": regions,
        "water_ratio": water_ratio,
        "shore_ratio": shore_ratio,
        "river_ratio": river_ratio,
        "flow_bonus": flow_bonus,
        "source_mouth_bonus": source_mouth_bonus,
        "river_tile_bonus": river_tile_bonus,
        "reward": total_reward
    }

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
            if (0 <= x < len(grid[0]) and 0 <= y < len(grid) and \
               water_map[y, x] and (x, y) not in visited):
                visited.add((x, y))
                queue.append((x, y))

    return False