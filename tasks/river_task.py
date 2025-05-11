import numpy as np
from collections import deque
from .utils import calc_num_regions, percent_target_tiles_excluding_excluded_tiles

def get_river_biome(grid: list[list[set[str]]]) -> str:
    water_tiles = {
        "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
        "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
        "shore_bl", "shore_br", "shore_lr", "shore_rl"
    }

    # Calculate basic metrics
    water_cells, shore_cells = 0, 0
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

    # Check connectivity and flow
    is_connected = check_water_connectivity(grid, water_tiles)
    flow_length = measure_river_flow(grid, water_tiles)
    max_dimension = max(len(grid), len(grid[0]))
    flow_quality = flow_length / max_dimension if max_dimension > 0 else 0
    
    has_large_cluster = check_large_water_clusters(grid, water_tiles)

    if (is_connected and
        flow_quality >= 0.75 and 
        0.15 <= water_ratio <= 0.35 and 
        shore_ratio <= 0.3 and
        not has_large_cluster):
        return "river"
    return "unknown"

def check_water_connectivity(grid: list[list[set[str]]], water_tiles: set[str]) -> bool:
    """Check if all water tiles form a single connected region"""
    h, w = len(grid), len(grid[0])
    water_map = np.zeros((h, w), dtype=bool)
    
    # Find all water tiles and starting point
    start = None
    water_count = 0
    for y in range(h):
        for x in range(w):
            cell = grid[y][x]
            if len(cell) == 1 and next(iter(cell)).lower() in water_tiles:
                water_map[y, x] = True
                water_count += 1
                if start is None:
                    start = (y, x)

    # If no water or single tile, return True
    if water_count <= 1:
        return True

    # Flood fill to count connected water tiles
    visited = np.zeros_like(water_map)
    queue = deque([start])
    visited[start] = True
    connected_count = 1

    while queue:
        y, x = queue.popleft()
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if (0 <= ny < h and 0 <= nx < w and 
                water_map[ny, nx] and not visited[ny, nx]):
                visited[ny, nx] = True
                connected_count += 1
                queue.append((ny, nx))

    return connected_count == water_count

def check_large_water_clusters(grid: list[list[set[str]]], water_tiles: set[str]) -> bool:
    """Check for 3x3 or larger pure water clusters"""
    h, w = len(grid), len(grid[0])
    water_map = np.zeros((h, w), dtype=bool)
    
    for y in range(h):
        for x in range(w):
            cell = grid[y][x]
            if len(cell) == 1 and next(iter(cell)).lower() in water_tiles:
                water_map[y, x] = True

    for y in range(h - 2):
        for x in range(w - 2):
            if water_map[y:y+3, x:x+3].all():
                return True
    return False

def measure_river_flow(grid: list[list[set[str]]], water_tiles: set[str]) -> float:
    """Measure the longest continuous stretch of water tiles"""
    h, w = len(grid), len(grid[0])
    water_map = np.zeros((h, w), dtype=bool)
    
    for y in range(h):
        for x in range(w):
            cell = grid[y][x]
            if len(cell) == 1 and next(iter(cell)).lower() in water_tiles:
                water_map[y, x] = True

    def measure_direction(axis: int) -> float:
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

    return max(measure_direction(0), measure_direction(1))

def river_reward(grid: list[list[set[str]]]) -> tuple[float, dict]:
    water_tiles = {
        "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
        "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
        "shore_bl", "shore_br", "shore_lr", "shore_rl"
    }

    h, w = len(grid), len(grid[0])
    if h == 0 or w == 0:
        return -float('inf'), {}

    # Create water and shore maps
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

    # Calculate basic metrics
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
    is_connected = check_water_connectivity(grid, water_tiles)
    has_large_cluster = check_large_water_clusters(grid, water_tiles)
    river_width = calculate_river_width(water_map, flow_quality, h, w)

    # Reward parameters
    IDEAL_WATER_RATIO_MIN = 0.15
    IDEAL_WATER_RATIO_MAX = 0.35
    IDEAL_SHORE_RATIO = 0.3
    IDEAL_WIDTH_MAX = 2.5
    MIN_FLOW_QUALITY = 0.75

    # Calculate penalties and bonuses
    water_penalty = (
        (IDEAL_WATER_RATIO_MIN - water_ratio) * -200 if water_ratio < IDEAL_WATER_RATIO_MIN else
        (water_ratio - IDEAL_WATER_RATIO_MAX) * -200 if water_ratio > IDEAL_WATER_RATIO_MAX else 0
    )
    shore_penalty = max(0, (shore_ratio - IDEAL_SHORE_RATIO)) * -100
    width_penalty = max(0, river_width - IDEAL_WIDTH_MAX) * -50
    connectivity_penalty = -200 if not is_connected else 0
    cluster_penalty = -100 if has_large_cluster else 0
    flow_bonus = 30 if flow_quality >= MIN_FLOW_QUALITY else 0

    # Combine rewards
    total_reward = min(
        water_penalty +
        shore_penalty +
        width_penalty +
        connectivity_penalty +
        cluster_penalty +
        flow_bonus,
        0.0  # Cap at 0 (perfect score)
    )

    return total_reward, {
        "water_ratio": round(water_ratio, 3),
        "shore_ratio": round(shore_ratio, 3),
        "flow_quality": round(flow_quality, 3),
        "regions": regions,
        "river_width": round(river_width, 2),
        "is_connected": is_connected,
        "has_large_cluster": has_large_cluster,
        "reward": total_reward,
    }

def calculate_river_width(water_map: np.ndarray, flow_quality: float, h: int, w: int) -> float:
    """Calculate average river width perpendicular to main flow direction"""
    if flow_quality < 0.5:
        return float('inf')
    
    # Determine primary flow direction
    horizontal_flow = 0
    for y in range(h):
        current = 0
        for x in range(w):
            if water_map[y, x]:
                current += 1
                horizontal_flow = max(horizontal_flow, current)
            else:
                current = 0

    vertical_flow = 0
    for x in range(w):
        current = 0
        for y in range(h):
            if water_map[y, x]:
                current += 1
                vertical_flow = max(vertical_flow, current)
            else:
                current = 0

    is_horizontal = horizontal_flow > vertical_flow
    
    # Measure width perpendicular to flow direction
    widths = []
    if is_horizontal:
        for x in range(w):
            current_width = 0
            max_width = 0
            for y in range(h):
                if water_map[y, x]:
                    current_width += 1
                    max_width = max(max_width, current_width)
                else:
                    current_width = 0
            if max_width > 0:
                widths.append(max_width)
    else:
        for y in range(h):
            current_width = 0
            max_width = 0
            for x in range(w):
                if water_map[y, x]:
                    current_width += 1
                    max_width = max(max_width, current_width)
                else:
                    current_width = 0
            if max_width > 0:
                widths.append(max_width)
    
    return np.mean(widths) if widths else float('inf')