"""A rule-based reference driver for the dynamic-waypoint task (not learned).

It reads the true simulator state, so it is a feasibility check and a yardstick,
not a fair competitor: overtake in a neighbouring lane for FRONT, otherwise join
the slot lane, then steer speed toward the slot. Gap checks are deliberately crude.
"""
import numpy as np

LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = range(5)


def lane_is_clear(env, lane_id, ahead=12.0, behind=6.0):
    ego = env.highway.vehicle
    return all(not (car.lane_index[2] == lane_id and -behind < car.position[0] - ego.position[0] < ahead)
               for car in env.highway.road.vehicles if car is not ego)


def leader_gap(env, lane_id):
    ego = env.highway.vehicle
    gaps = [car.position[0] - ego.position[0] for car in env.highway.road.vehicles
            if car is not ego and car.lane_index[2] == lane_id and car.position[0] > ego.position[0]]
    return min(gaps, default=np.inf)


def scripted_action(env):
    ego, anchor = env.highway.vehicle, env.anchor
    current = ego.target_lane_index[2]
    target = env.goal_lane
    if env.slot == "front" and anchor.position[0] + 10 > ego.position[0]:
        # Still behind or beside the anchor: pass it in a neighbouring lane first.
        neighbours = [lane for lane in (anchor.lane_index[2] - 1, anchor.lane_index[2] + 1)
                      if 0 <= lane < env.env_config["lanes_count"]]
        target = current if current in neighbours else min(neighbours, key=lambda lane: abs(lane - current))
    if target != current and ego.lane_index[2] == current:
        step = -1 if target < current else 1
        if lane_is_clear(env, current + step):
            return LANE_LEFT if step < 0 else LANE_RIGHT
    desired = 25.0
    if target == env.goal_lane:
        lane = env.highway.road.network.get_lane(env.slot_lane_index)
        ego_s, _ = lane.local_coordinates(ego.position)
        slot_s, _ = lane.local_coordinates(env.goal_position)
        desired = anchor.speed + 0.5 * (slot_s - ego_s)
    gap = leader_gap(env, ego.lane_index[2])
    if gap < 12 + 0.8 * ego.speed:
        desired = min(desired, anchor.speed if gap > 9 else 10.0)
    desired = float(np.clip(desired, 10.0, 25.0))
    if ego.target_speed < desired - 0.5:
        return FASTER
    if ego.target_speed > desired + 0.5:
        return SLOWER
    return IDLE
