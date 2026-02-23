class_name ScoreManager
extends RefCounted

## Manages score calculation for AI training episodes.
## All constants include design notes explaining the balance rationale.

## DESIGN NOTE: Increased from 1000 to 1500. Killing one pawn (1500 pts) is now
## worth 3s of survival, creating strong selection pressure for combat behavior.
const KILL_MULTIPLIER: int = 1500

## DESIGN NOTE: Increased from 5000 to 6000. Encourages spatial navigation
## and goal-directed behavior. Powerups enable kills; collecting them is a
## virtuous cycle that accelerates learning.
const POWERUP_COLLECT_BONUS: int = 6000

## Movement distance tracking for exploration reward
var _last_position: Vector2 = Vector2.ZERO
var _cached_enemies: Array = []
var _cached_enemies_frame: int = -1


func calculate_exploration_bonus(delta: float, player: CharacterBody2D) -> float:
	## Reward AI for exploring the arena (covering distance).
	## DESIGN NOTE: Dense exploration reward breaks the "stand still" degenerate
	## strategy that dominates early generations. Small signal (max ~15 pts/sec)
	## that doesn't dominate once agents learn to kill. Follows the "auxiliary
	## reward" pattern from Jaderberg et al. (2017).
	if _last_position == Vector2.ZERO:
		_last_position = player.position
		return 0.0
	var dist := player.position.distance_to(_last_position)
	_last_position = player.position
	return minf(dist * 0.05, 15.0 * delta)


func calculate_engagement_bonus(delta: float, player: CharacterBody2D, scene: Node2D) -> float:
	## Reward AI for staying in combat range of enemies (400–1200 px).
	## DESIGN NOTE: Bridges the gap between "survival" and "first kill." Without
	## this, agents must randomly reach shooting range AND fire simultaneously —
	## a very sparse conjunction. The sweet spot (400–1200px) is close enough to
	## shoot but far enough to dodge. Inspired by potential-based reward shaping
	## (Wiewiora et al., 2003).
	var frame := Engine.get_physics_frames()
	if frame != _cached_enemies_frame:
		_cached_enemies = scene.get_tree().get_nodes_in_group("enemy")
		_cached_enemies_frame = frame
	var count := 0
	for enemy in _cached_enemies:
		if not is_instance_valid(enemy):
			continue
		var dist := player.position.distance_to(enemy.position)
		if dist >= 400.0 and dist <= 1200.0:
			count += 1
	return minf(count, 3) * 20.0 * delta


func reset_exploration_tracking() -> void:
	_last_position = Vector2.ZERO
