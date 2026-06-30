"""The real multi-task metric: a static chain of go-to subgoals scored out of N.

For now the chain is a fixed list (stand-in for the LLM that will eventually
string subgoals together); score = how many legs the navigator reaches in one
episode, pose persisting leg-to-leg. Top score = len(chain).

Shared by the in-training TB metric (agent.chain_eval) and the offline harness
(chained_eval.py) so both report the identical number.
"""

# get_trash >> go_to_fridge >> go_to_human >> go_to_door >> go_to_human
DEFAULT_CHAIN = ["collect_trash", "go_to_fridge", "go_to_human",
                 "go_to_door", "go_to_human"]


def world_state(base) -> dict:
    """Snapshot the task-relevant world state -- the honest ground truth for
    whether a leg accomplished anything. Shared by the planner's WorldModel and
    the offline chain harness so both read identical fields."""
    info = base._task_manager.get_info(base._robot)
    return {
        "carrying": base._robot.carrying,
        "trash_remaining": info["trash_remaining"],
        "drink_delivered": info["drink_delivered"],
        "package_delivered": info["package_delivered"],
        "robot_xy": (float(base._robot.x), float(base._robot.y)),
    }


def leg_succeeded(name, before, after, arrived) -> bool:
    """Honest leg success: did the task actually happen (a world-state delta) --
    not merely 'the robot got near the coordinate'. `arrived` is the fallback
    only for pure-navigation legs that have no task effect."""
    if name == "collect_trash":
        return after["trash_remaining"] < before["trash_remaining"]
    if name == "go_to_fridge":
        return after["carrying"] == "drink" and before["carrying"] != "drink"
    if name == "go_to_door":
        return after["carrying"] == "package" and before["carrying"] != "package"
    if name in ("go_to_human", "deliver_to_human", "deliver_drink", "deliver_package"):
        newly_delivered = (
            (after["drink_delivered"] and not before["drink_delivered"])
            or (after["package_delivered"] and not before["package_delivered"]))
        return newly_delivered if before["carrying"] is not None else arrived
    return arrived


def resolve_goal(base, name):
    """Map a chain subgoal name to pixel (x, y). go_to_human -> the recliner
    (the human's seat); every other name goes through the env goal registry.
    Resolve all legs up front (before stepping) so incidental trash pickup can't
    empty trash_positions mid-chain."""
    if name == "go_to_human":
        col, row = base._map.fixtures["recliner"]
        return tuple(float(v) for v in base._map.tile_to_pixel(col, row))
    return tuple(float(v) for v in base.goal_to_coordinates(name))
