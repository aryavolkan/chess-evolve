extends RefCounted
class_name ChessMapElites

## MAP-Elites quality-diversity archive for chess evolution.
## Maintains a 2D grid of behavioral niches, each storing the
## highest-fitness solution found for that behavior region.
##
## Behavior dimensions:
##   X — Aggressiveness (captures per move)
##   Y — Game length (moves per game)

var archive: Dictionary = {}  ## Vector2i → {solution, fitness, behavior: Vector2}
var grid_size: int = 20

# Behavior space bounds (adaptive — expands as we see new extremes)
var behavior_mins: Vector2 = Vector2.ZERO
var behavior_maxs: Vector2 = Vector2(1.0, 200.0)  # Initial estimates


func _init(p_grid_size: int = 20) -> void:
	grid_size = p_grid_size


func add(solution, behavior: Vector2, fitness: float) -> bool:
	## Try to insert a solution into the archive.
	## Returns true if the solution was added (new niche or better fitness).
	## The caller is responsible for passing a cloned/copied solution.
	
	# Expand bounds if behavior exceeds current range
	_expand_bounds(behavior)
	
	var bin = _behavior_to_bin(behavior)
	if not archive.has(bin) or archive[bin].fitness < fitness:
		archive[bin] = {solution = solution, fitness = fitness, behavior = behavior}
		return true
	return false


func get_elite(bin: Vector2i):
	## Get the elite solution at a specific bin. Returns null if empty.
	if archive.has(bin):
		return archive[bin]
	return null


func sample(count: int) -> Array:
	## Sample random elites from the archive.
	## Returns up to count entries (fewer if archive is smaller).
	if archive.is_empty():
		return []
	var keys = archive.keys()
	var result: Array = []
	var n = mini(count, keys.size())
	# Shuffle keys and take first n
	var shuffled = keys.duplicate()
	shuffled.shuffle()
	for i in n:
		result.append(archive[shuffled[i]])
	return result


func get_coverage() -> float:
	## Fraction of bins that are occupied (0.0 to 1.0).
	return float(archive.size()) / (grid_size * grid_size)


func get_occupied_count() -> int:
	## Number of occupied bins.
	return archive.size()


func get_best_fitness() -> float:
	## Highest fitness across all archive entries.
	var best: float = -INF
	for entry in archive.values():
		best = maxf(best, entry.fitness)
	return best if best != -INF else 0.0


func get_average_fitness() -> float:
	## Mean fitness across all occupied bins.
	if archive.is_empty():
		return 0.0
	var total: float = 0.0
	for entry in archive.values():
		total += entry.fitness
	return total / archive.size()


func get_stats() -> Dictionary:
	return {
		"grid_size": grid_size,
		"occupied": archive.size(),
		"total_bins": grid_size * grid_size,
		"coverage": get_coverage(),
		"best_fitness": get_best_fitness(),
		"avg_fitness": get_average_fitness(),
		"behavior_mins": behavior_mins,
		"behavior_maxs": behavior_maxs,
	}


func get_archive_grid() -> Array:
	## Return a 2D grid (Array of Arrays) representing the archive.
	## grid[x][y] is either null (empty cell) or {fitness: float, behavior: Vector2}.
	## Grid dimensions are grid_size × grid_size.
	var grid: Array = []
	grid.resize(grid_size)
	for x in grid_size:
		var col: Array = []
		col.resize(grid_size)
		col.fill(null)
		for y in grid_size:
			var bin := Vector2i(x, y)
			if archive.has(bin):
				col[y] = {
					"fitness": archive[bin].fitness,
					"behavior": archive[bin].behavior,
				}
		grid[x] = col
	return grid


func clear() -> void:
	archive.clear()


static func calculate_behavior(game_stats: Dictionary) -> Vector2:
	## Compute 2D behavior descriptor from game statistics.
	## 
	## X (Aggressiveness): captures per move
	## Y (Game Length): total moves in the game
	##
	## game_stats should contain:
	##   - captures (int): number of pieces captured
	##   - moves (int): total moves in the game
	##   - win (bool): whether the player won
	
	var captures: int = game_stats.get("captures", 0)
	var moves: int = maxi(game_stats.get("moves", 1), 1)
	
	# Aggressiveness: captures per move (0 = very passive, higher = aggressive)
	var aggressiveness: float = float(captures) / float(moves)
	
	# Game length: number of moves
	var game_length: float = float(moves)
	
	return Vector2(aggressiveness, game_length)


func _behavior_to_bin(behavior: Vector2) -> Vector2i:
	## Map a continuous behavior vector to a discrete grid bin.
	var range_x: float = behavior_maxs.x - behavior_mins.x
	var range_y: float = behavior_maxs.y - behavior_mins.y
	if range_x <= 0:
		range_x = 1.0
	if range_y <= 0:
		range_y = 1.0
	var norm_x: float = clampf((behavior.x - behavior_mins.x) / range_x, 0.0, 1.0)
	var norm_y: float = clampf((behavior.y - behavior_mins.y) / range_y, 0.0, 1.0)
	var bx: int = mini(int(norm_x * grid_size), grid_size - 1)
	var by: int = mini(int(norm_y * grid_size), grid_size - 1)
	return Vector2i(bx, by)


func _expand_bounds(behavior: Vector2) -> void:
	## Expand behavior bounds if a new observation falls outside.
	## Adds 10% headroom so the point isn't right on the edge.
	if behavior.x > behavior_maxs.x:
		behavior_maxs.x = behavior.x * 1.1
	if behavior.y > behavior_maxs.y:
		behavior_maxs.y = behavior.y * 1.1
	if behavior.x < behavior_mins.x:
		behavior_mins.x = behavior.x
	if behavior.y < behavior_mins.y:
		behavior_mins.y = behavior.y
