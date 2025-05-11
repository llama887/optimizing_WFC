import numpy as np
from collections import deque
from .utils import calc_num_regions, percent_target_tiles_excluding_excluded_tiles

def get_river_biome(grid: list[list[set[str]]]) -> str:
    water_tiles = {
        "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
        "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
        "shore_bl", "shore_br", "shore_lr", "shore_rl"
    }

    water_cells = 0
    shore_cells = 0
    pure_water_cells = 0
    
    for row in grid:
        for cell in row:
            if len(cell) == 1:
                tile = next(iter(cell)).lower()
                if tile in water_tiles:
                    water_cells += 1
                    if tile == "water":
                        pure_water_cells += 1
                    if "shore" in tile:
                        shore_cells += 1

    total_cells = len(grid) * len(grid[0])
    if total_cells == 0:
        return "unknown"

    water_ratio = water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0

    flow_length = measure_river_flow(grid, water_tiles)
    max_dimension = max(len(grid), len(grid[0]))
    flow_quality = flow_length / max_dimension if max_dimension > 0 else 0
    
    has_large_cluster = check_large_water_clusters(grid, water_tiles, max_dimension)

    if (flow_quality >= 0.75 and 
        0.15 <= water_ratio <= 0.35 and 
        shore_ratio <= 0.3 and
        not has_large_cluster):
        return "river"
    return "unknown"

def check_large_water_clusters(grid: list[list[set[str]]], water_tiles: set[str], max_dimension: int) -> bool:
    h, w = len(grid), len(grid[0])
    water_map = np.zeros((h, w), dtype=bool)
    
    for y in range(h):
        for x in range(w):
            cell = grid[y][x]
            if len(cell) == 1 and next(iter(cell)).lower() in water_tiles:
                water_map[y, x] = True

    for y in range(h - 2):
        for x in range(w - 2):
            if (water_map[y:y+3, x:x+3].all()):
                return True
    return False

def measure_river_flow(grid: list[list[set[str]]], water_tiles: set[str]) -> float:
    h, w = len(grid), len(grid[0])
    water_map = np.zeros((h, w), dtype=bool)
    
    for y in range(h):
        for x in range(w):
            cell = grid[y][x]
            if len(cell) == 1 and next(iter(cell)).lower() in water_tiles:
                water_map[y, x] = True

    def measure_flow(axis: int) -> float:
        max_flow = 0
        if axis == 0:  # horizontal
            for y in range(h):
                current = 0
                for x in range(w):
                    if water_map[y, x]:
                        current += 1
                        max_flow = max(max_flow, current)
                    else:
                        current = 0
        else:  # vertical
            for x in range(w):
                current = 0
                for y in range(h):
                    if water_map[y, x]:
                        current += 1
                        max_flow = max(max_flow, current)
                    else:
                        current = 0
        return max_flow

    horizontal_flow = measure_flow(0)
    vertical_flow = measure_flow(1)
    
    return max(horizontal_flow, vertical_flow)

def river_reward(grid: list[list[set[str]]]) -> tuple[float, dict]:
    water_tiles = {
        "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
        "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
        "shore_bl", "shore_br", "shore_lr", "shore_rl"
    }

    h, w = len(grid), len(grid[0])
    water_map = np.zeros((h, w), dtype=bool)
    shore_map = np.zeros((h, w), dtype=bool)
    
    for y in range(h):
        for x in range(w):
            cell = grid[y][x]
            if len(cell) == 1:
                tile = next(iter(cell)).lower()
                if tile in water_tiles:
                    water_map[y, x] = True
                    if "shore" in tile:
                        shore_map[y, x] = True

    if h == 0 or w == 0:
        return -float('inf'), {}

    water_ratio = percent_target_tiles_excluding_excluded_tiles(
        grid,
        is_target_tiles=lambda t: t.lower() in water_tiles,
        exclude_prefixes=["path"],
    )
    
    shore_ratio = (shore_map.sum() / water_map.sum()) if water_map.sum() > 0 else 0.0

    flow_length = measure_river_flow(grid, water_tiles)
    max_dimension = max(h, w)
    flow_quality = flow_length / max_dimension if max_dimension > 0 else 0.0

    regions = calc_num_regions(water_map.astype(np.int8))

    def calculate_river_width(wmap: np.ndarray) -> float:
        if flow_quality < 0.5:
            return float('inf')
        
        # Use the existing measure_river_flow logic to determine direction
        horizontal_flow = 0
        for y in range(h):
            current = 0
            for x in range(w):
                if wmap[y, x]:
                    current += 1
                    horizontal_flow = max(horizontal_flow, current)
                else:
                    current = 0

        vertical_flow = 0
        for x in range(w):
            current = 0
            for y in range(h):
                if wmap[y, x]:
                    current += 1
                    vertical_flow = max(vertical_flow, current)
                else:
                    current = 0

        is_horizontal = horizontal_flow > vertical_flow
        
        widths = []
        if is_horizontal:
            for y in range(h):
                current_width = 0
                max_width = 0
                for x in range(w):
                    if wmap[y, x]:
                        current_width += 1
                        max_width = max(max_width, current_width)
                    else:
                        current_width = 0
                if max_width > 0:
                    widths.append(max_width)
        else:
            for x in range(w):
                current_width = 0
                max_width = 0
                for y in range(h):
                    if wmap[y, x]:
                        current_width += 1
                        max_width = max(max_width, current_width)
                    else:
                        current_width = 0
                if max_width > 0:
                    widths.append(max_width)
        
        return np.mean(widths) if widths else float('inf')

    river_width = calculate_river_width(water_map)
    has_large_cluster = check_large_water_clusters(grid, water_tiles, max_dimension)

    IDEAL_WATER_RATIO_MIN = 0.15
    IDEAL_WATER_RATIO_MAX = 0.35
    IDEAL_SHORE_RATIO = 0.3
    IDEAL_REGIONS = 1
    IDEAL_WIDTH_MAX = 2.5
    MIN_FLOW_QUALITY = 0.75
    FLOW_BONUS = 30
    WIDTH_PENALTY_MULTIPLIER = 50

    if water_ratio < IDEAL_WATER_RATIO_MIN:
        water_penalty = (IDEAL_WATER_RATIO_MIN - water_ratio) * -200
    elif water_ratio > IDEAL_WATER_RATIO_MAX:
        water_penalty = (water_ratio - IDEAL_WATER_RATIO_MAX) * -200
    else:
        water_penalty = 0

    shore_penalty = max(0, (shore_ratio - IDEAL_SHORE_RATIO)) * -100
    region_penalty = abs(regions - IDEAL_REGIONS) * -50
    flow_penalty = -100 if flow_quality < MIN_FLOW_QUALITY else 0
    flow_bonus = FLOW_BONUS if flow_quality >= MIN_FLOW_QUALITY else 0
    width_penalty = max(0, river_width - IDEAL_WIDTH_MAX) * -WIDTH_PENALTY_MULTIPLIER
    cluster_penalty = -100 if has_large_cluster else 0

    raw_reward = (
        water_penalty
        + shore_penalty
        + region_penalty
        + flow_penalty
        + flow_bonus
        + width_penalty
        + cluster_penalty
    )
    total_reward = min(raw_reward, 0.0)

    return total_reward, {
        "water_ratio": round(water_ratio, 3),
        "shore_ratio": round(shore_ratio, 3),
        "flow_quality": round(flow_quality, 3),
        "regions": regions,
        "river_width": round(river_width, 2),
        "has_large_cluster": has_large_cluster,
        "reward": total_reward,
    }