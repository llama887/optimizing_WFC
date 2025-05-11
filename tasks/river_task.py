import numpy as np
from collections import deque
from typing import List, Set, Tuple, Dict
from .utils import calc_num_regions, percent_target_tiles_excluding_excluded_tiles

WATER_TILES = {
    "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
    "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
    "shore_bl", "shore_br", "shore_lr", "shore_rl"
}

# Constants for river biome classification
IDEAL_WATER_RATIO_MIN = 0.15
IDEAL_WATER_RATIO_MAX = 0.35
IDEAL_SHORE_RATIO = 0.3
IDEAL_WIDTH_MAX = 2.5
MIN_FLOW_QUALITY = 0.75


def get_river_biome(grid: List[List[Set[str]]]) -> str:
    """Determine if the given grid represents a river biome."""
    water_cells, shore_cells = count_water_and_shore_tiles(grid)
    total_cells = len(grid) * len(grid[0]) if grid else 0

    if total_cells == 0:
        return "unknown"

    water_ratio = water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0

    # Check river characteristics
    is_connected = check_water_connectivity(grid)
    flow_length = measure_river_flow(grid)
    max_dimension = max(len(grid), len(grid[0]))
    flow_quality = flow_length / max_dimension if max_dimension > 0 else 0
    has_large_cluster = check_large_water_clusters(grid)

    if (is_connected and
        flow_quality >= MIN_FLOW_QUALITY and 
        IDEAL_WATER_RATIO_MIN <= water_ratio <= IDEAL_WATER_RATIO_MAX and 
        shore_ratio <= IDEAL_SHORE_RATIO and
        not has_large_cluster):
        return "river"
    return "unknown"


def count_water_and_shore_tiles(grid: List[List[Set[str]]]) -> Tuple[int, int]:
    """Count water and shore tiles in the grid."""
    water_cells, shore_cells = 0, 0
    
    for row in grid:
        for cell in row:
            if len(cell) == 1:
                tile = next(iter(cell)).lower()
                if tile in WATER_TILES:
                    water_cells += 1
                    if "shore" in tile:
                        shore_cells += 1
                        
    return water_cells, shore_cells


def check_water_connectivity(grid: List[List[Set[str]]]) -> bool:
    """Check if all water tiles form a single connected region."""
    water_map, water_count, start = create_water_map(grid)
    
    if water_count <= 1:
        return True

    connected_count = flood_fill_water(water_map, start)
    return connected_count == water_count


def create_water_map(grid: List[List[Set[str]]]) -> Tuple[np.ndarray, int, Tuple[int, int]]:
    """Create a boolean map of water tiles and count them."""
    h, w = len(grid), len(grid[0])
    water_map = np.zeros((h, w), dtype=bool)
    start = None
    water_count = 0
    
    for y in range(h):
        for x in range(w):
            cell = grid[y][x]
            if len(cell) == 1 and next(iter(cell)).lower() in WATER_TILES:
                water_map[y, x] = True
                water_count += 1
                if start is None:
                    start = (y, x)
                    
    return water_map, water_count, start


def flood_fill_water(water_map: np.ndarray, start: Tuple[int, int]) -> int:
    """Perform flood fill to count connected water tiles."""
    h, w = water_map.shape
    visited = np.zeros_like(water_map)
    queue = deque([start])
    visited[start] = True
    connected_count = 1

    while queue:
        y, x = queue.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if (0 <= ny < h and 0 <= nx < w and 
                water_map[ny, nx] and not visited[ny, nx]):
                visited[ny, nx] = True
                connected_count += 1
                queue.append((ny, nx))
                
    return connected_count


def check_large_water_clusters(grid: List[List[Set[str]]]) -> bool:
    """Check for 3x3 or larger pure water clusters."""
    water_map = create_water_map(grid)[0]
    h, w = water_map.shape

    for y in range(h - 2):
        for x in range(w - 2):
            if water_map[y:y+3, x:x+3].all():
                return True
    return False


def measure_river_flow(grid: List[List[Set[str]]]) -> float:
    """Measure the longest continuous stretch of water tiles."""
    water_map = create_water_map(grid)[0]
    h, w = water_map.shape

    def measure_direction(axis: int) -> float:
        """Measure flow in either horizontal (0) or vertical (1) direction."""
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


def river_reward(grid: List[List[Set[str]]]) -> Tuple[float, Dict[str, float]]:
    """Calculate reward for how well the grid represents a river biome."""
    h, w = len(grid), len(grid[0])
    if h == 0 or w == 0:
        return -float('inf'), {}

    water_map, shore_map = create_water_and_shore_maps(grid)
    metrics = calculate_river_metrics(grid, water_map, shore_map, h, w)
    reward = calculate_reward_from_metrics(metrics)
    
    return reward, metrics


def create_water_and_shore_maps(grid: List[List[Set[str]]]) -> Tuple[np.ndarray, np.ndarray]:
    """Create boolean maps for water and shore tiles."""
    h, w = len(grid), len(grid[0])
    water_map = np.zeros((h, w), dtype=bool)
    shore_map = np.zeros((h, w), dtype=bool)
    
    for y in range(h):
        for x in range(w):
            cell = grid[y][x]
            if len(cell) == 1:
                tile = next(iter(cell)).lower()
                if tile in WATER_TILES:
                    water_map[y, x] = True
                    if "shore" in tile:
                        shore_map[y, x] = True
                        
    return water_map, shore_map


def calculate_river_metrics(
    grid: List[List[Set[str]]], 
    water_map: np.ndarray, 
    shore_map: np.ndarray,
    h: int, 
    w: int
) -> Dict[str, float]:
    """Calculate various metrics about the river."""
    water_ratio = percent_target_tiles_excluding_excluded_tiles(
        grid,
        is_target_tiles=lambda t: t.lower() in WATER_TILES,
        exclude_prefixes=["path"],
    )
    
    shore_ratio = (shore_map.sum() / water_map.sum()) if water_map.sum() > 0 else 0.0
    flow_length = measure_river_flow(grid)
    max_dimension = max(h, w)
    flow_quality = flow_length / max_dimension if max_dimension > 0 else 0.0
    regions = calc_num_regions(water_map.astype(np.int8))
    is_connected = check_water_connectivity(grid)
    has_large_cluster = check_large_water_clusters(grid)
    river_width = calculate_river_width(water_map, flow_quality, h, w)
    
    hill_ratio = percent_target_tiles_excluding_excluded_tiles(
        grid,
        is_target_tiles=lambda t: t.lower().startswith("hill"),
        exclude_prefixes=["path"]
    )
    
    return {
        "water_ratio": round(water_ratio, 3),
        "shore_ratio": round(shore_ratio, 3),
        "flow_quality": round(flow_quality, 3),
        "regions": regions,
        "river_width": round(river_width, 2),
        "is_connected": is_connected,
        "has_large_cluster": has_large_cluster,
        "hill_ratio": round(hill_ratio, 3),
    }


def calculate_reward_from_metrics(metrics: Dict[str, float]) -> float:
    """Calculate reward based on river metrics."""
    water_ratio = metrics["water_ratio"]
    shore_ratio = metrics["shore_ratio"]
    flow_quality = metrics["flow_quality"]
    is_connected = metrics["is_connected"]
    has_large_cluster = metrics["has_large_cluster"]
    river_width = metrics["river_width"]
    hill_ratio = metrics["hill_ratio"]

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
    hill_penalty = -hill_ratio * 200

    raw_reward = (
        water_penalty + shore_penalty + width_penalty + 
        connectivity_penalty + cluster_penalty + flow_bonus + hill_penalty
    )
    
    return min(raw_reward, 0)


def calculate_river_width(water_map: np.ndarray, flow_quality: float, h: int, w: int) -> float:
    """Calculate average river width perpendicular to main flow direction."""
    if flow_quality < 0.5:
        return float('inf')
    
    horizontal_flow = measure_flow_direction(water_map, h, w, horizontal=True)
    vertical_flow = measure_flow_direction(water_map, h, w, horizontal=False)
    is_horizontal = horizontal_flow > vertical_flow
    
    widths = measure_widths_perpendicular_to_flow(water_map, h, w, is_horizontal)
    return np.mean(widths) if widths else float('inf')


def measure_flow_direction(water_map: np.ndarray, h: int, w: int, horizontal: bool) -> int:
    """Measure flow in either horizontal or vertical direction."""
    max_flow = 0
    
    if horizontal:
        for y in range(h):
            current = 0
            for x in range(w):
                if water_map[y, x]:
                    current += 1
                    max_flow = max(max_flow, current)
                else:
                    current = 0
    else:
        for x in range(w):
            current = 0
            for y in range(h):
                if water_map[y, x]:
                    current += 1
                    max_flow = max(max_flow, current)
                else:
                    current = 0
                    
    return max_flow


def measure_widths_perpendicular_to_flow(
    water_map: np.ndarray, 
    h: int, 
    w: int, 
    is_horizontal: bool
) -> List[int]:
    """Measure widths perpendicular to the main flow direction."""
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
                
    return widths