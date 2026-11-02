"""Fifteen deterministic driving examples with live, state-derived DFA graphs.

Run: python mainV4.py --scenario 3
Verify every example: python mainV4.py --verify
The scene controllers use highway-env's existing steering/speed controllers;
only the vocabulary in README_V4.md can advance a scenario DFA.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import math
import os
from pathlib import Path

import numpy as np

import mainV2 as v2
from simulator_dashboard import Dashboard, ACCENT, BG, BORDER, CARD, MUTED, TEXT
from highway_env.vehicle.controller import ControlledVehicle


@dataclass(frozen=True)
class Geometry:
    d0: float = .8
    headway: float = .35
    front_horizon: float = .9
    back_horizon: float = .75
    velocity_epsilon: float = .45

    def half_length(self, car):
        length = car.length if hasattr(car,"length") else car.LENGTH
        return length / 2 + self.d0 + self.headway * np.linalg.norm(car.velocity)

    def front_length(self, car):
        return self.front_horizon * float(np.linalg.norm(car.velocity))

    def back_length(self, car):
        return self.back_horizon * float(np.linalg.norm(car.velocity))


@dataclass(frozen=True)
class Rectangle:
    center: tuple[float, float]
    half_length: float
    half_width: float

    def contains(self, position):
        return (abs(position[0] - self.center[0]) <= self.half_length
                and abs(position[1] - self.center[1]) <= self.half_width)


def waypoint_regions(waypoint):
    """Name instances of the existing waypoint predicate for scene geometry."""
    if waypoint is None:
        return ()
    return tuple(waypoint.items()) if isinstance(waypoint,dict) else (("W",waypoint),)


def relative_position(ego, car):
    forward = np.array([math.cos(car.heading), math.sin(car.heading)])
    right = np.array([-forward[1], forward[0]])
    delta = ego.position - car.position
    return float(delta @ forward), float(delta @ right)


def evaluate_vocabulary(snapshot, cars, lanes, waypoint=None, zone=None, geometry=Geometry()):
    """Evaluate the supplied point-based geometry, with no hysteresis or extra APs.

    `cars` pins names to snapshot IDs, and `lanes` pins names to physical lanes.
    They never rebind as ego moves. Lane boundaries use highway-env.on_lane.
    """
    ego = snapshot.ego
    facts = {}
    by_id = {car.uid: car for car in snapshot.vehicles}
    for name, uid in cars.items():
        car = by_id[uid]
        s, d = relative_position(ego, car)
        lane = snapshot.road.network.get_lane(car.lane_index)
        width = lane.width_at(lane.local_coordinates(car.position)[0])
        half = geometry.half_length(car)
        facts.update({
            f"Car {name} personal": abs(s) <= half and abs(d) <= width / 2,
            f"Car {name} left": abs(s) <= half and -3 * width / 2 <= d < -width / 2,
            f"Car {name} right": abs(s) <= half and width / 2 < d <= 3 * width / 2,
            f"Car {name} front": half < s <= half + geometry.front_length(car) and abs(d) <= width / 2,
            f"Car {name} back": -half - geometry.back_length(car) <= s < -half and abs(d) <= width / 2,
            f"Car {name} velocity": np.linalg.norm(ego.velocity - car.velocity) < geometry.velocity_epsilon,
        })
    for name, index in lanes.items():
        facts[f"In lane {name}"] = snapshot.road.network.get_lane(index).on_lane(ego.position)
    for name, region in waypoint_regions(waypoint):
        facts[f"Waypoint {name}"] = region.contains(ego.position)
    if zone is not None:
        facts["Avoid zone Z"] = not zone.contains(ego.position)
    return {name: bool(value) for name, value in facts.items()}


@dataclass(frozen=True)
class Guard:
    """A conjunction of supplied predicates, optionally negated. No expression eval."""
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()

    @property
    def label(self):
        return " AND ".join((*self.positive, *(f"NOT {p}" for p in self.negative)))

    def holds(self, facts):
        # Missing atoms are errors, including negated atoms: never accept on
        # a missing car or a misspelled grounding.
        return all(facts[p] for p in self.positive) and all(not facts[p] for p in self.negative)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    guard: Guard


@dataclass(frozen=True)
class Scenario:
    number: int
    title: str
    stages: tuple[tuple[str, str], ...]
    edges: tuple[Edge, ...]
    note: str
    initial: Guard
    failure: Guard | None = None
    state_vocabulary: tuple[tuple[str, Guard], ...] = ()
    expected_walk: tuple[str, ...] = ()

    @property
    def states(self):
        return tuple(name for name, _ in self.stages) + (("q_fail",) if self.failure else ())

    def vocabulary_for(self, state):
        """Every node has an exact vocabulary witness, never an action/phase."""
        declared = dict(self.state_vocabulary)
        if state in declared:
            return declared[state]
        if state == "q0":
            return self.initial
        if state == "q_fail" and self.failure:
            return self.failure
        incoming = {edge.guard for edge in self.edges if edge.target == state}
        if len(incoming) != 1:
            raise ValueError(f"{self.number}/{state}: must have one vocabulary condition")
        return next(iter(incoming))

    @property
    def expected_path(self):
        return self.expected_walk or tuple(name for name, _ in self.stages)

    def __post_init__(self):
        for state in self.states:
            if not self.vocabulary_for(state).label:
                raise ValueError(f"{self.number}/{state}: no vocabulary; cannot be a state")


def _edge(source, target, *positive, negative=()):
    return Edge(source, target, Guard(positive, negative))


def _overtake(number, side, canceled=False):
    atom = f"Car A {side}"
    if canceled:
        # These are the actual vocabulary states, reused on cancellation and
        # retry. B's braking belongs to the traffic script, not another node.
        return Scenario(
            number, f"Overtake {side} canceled, then completed",
            (("q0", "Car A back"), ("q1", atom), ("q_accept", "Car A front")),
            (_edge("q0", "q1", atom), _edge("q1", "q0", "Car A back"),
             _edge("q1", "q_accept", "Car A front", negative=("Car A back",))),
            "B brakes ahead. Ego brakes and returns to Car A back, then retries the same side-to-front path.",
            Guard(("Car A back",)),
            state_vocabulary=(("q0", Guard(("Car A back",))), ("q1", Guard((atom,))),
                              ("q_accept", Guard(("Car A front",)))),
            expected_walk=("q0", "q1", "q0", "q1", "q_accept"))
    else:
        stages = (("q0", "Close behind A"), ("q1", f"Beside A on the {side}"),
                  ("q_accept", "Ahead in the original lane"))
        edges = (_edge("q0", "q1", atom),
                 _edge("q1", "q_accept", "Car A front", "In lane original"))
    return Scenario(number, f"Overtake {side}" + (" canceled, then completed" if canceled else ""),
                    stages, edges,
                    "Start close behind A; pass nearby and return just ahead at A's speed."
                    + (" B brakes ahead; ego falls back, recovers, and waits for room before retrying." if canceled else ""),
                    Guard(("Car A back","In lane original")))


SCENARIOS = (
    _overtake(1, "left"), _overtake(2, "right"),
    _overtake(3, "left", True), _overtake(4, "right", True),
    Scenario(5, "Go from lane 1 to lane 3",
             (("q0", "Lane 1"), ("q1", "Lane 2"), ("q_accept", "Lane 3")),
             (_edge("q0", "q1", "In lane 2"), _edge("q1", "q_accept", "In lane 3")),
             "Reach lane 3, finish steering, then hold the final condition for 1.2 s before acceptance.",
             Guard(("In lane 1",))),
    Scenario(6, "Brake and match Car A's velocity",
             (("q0", "Matching behind A"), ("q1", "Velocity mismatch behind A"),
              ("q_accept", "Matching restored behind A")),
             (_edge("q0", "q1", "Car A back", negative=("Car A velocity","Car A personal")),
              _edge("q1", "q_accept", "Car A velocity", "Car A back", negative=("Car A personal",))),
             "A slows after 2 seconds. Compare measured following distance with a 1 s switching reference.",
             Guard(("Car A back","Car A velocity"),("Car A personal",))),
    Scenario(7, "React to avoid a personal-space violation",
             (("q0", "Outside A's near-back region"), ("q1", "Close behind; velocity mismatch"),
              ("q_accept", "Matching close behind A")),
             (_edge("q0", "q1", "Car A back", negative=("Car A personal", "Car A velocity")),
              _edge("q1", "q_accept", "Car A back", "Car A velocity", negative=("Car A personal",))),
             "Ego follows close. The near-back extension is 0.14 s of A's speed; ego brakes near personal space.",
             Guard(("Car A velocity",),("Car A back","Car A personal")),
             Guard(("Car A personal",))),
    Scenario(8, "Merge onto the highway",
             (("q0", "In the entry lane"), ("q1", "At the merge waypoint"),
              ("q_accept", "In the main lane, outside Z and B")),
             (_edge("q0", "q1", "Waypoint W", "Avoid zone Z"),
              _edge("q1", "q_accept", "In lane main", "Avoid zone Z", negative=("Car B personal",))),
             "The entry lane ends at Z. Reach W, join the main lane, and finish aligning before acceptance.",
             Guard(("In lane 3","Avoid zone Z")), Guard(negative=("Avoid zone Z",))),
    Scenario(9, "Overtake A, then overtake B",
             (("q0", "Close behind A"), ("q1", "Left of A"), ("q2", "Ahead of A in its lane"),
              ("q3", "Close behind B"), ("q4", "Left of B"), ("q_accept", "Ahead of B in its lane")),
             (_edge("q0", "q1", "Car A left"), _edge("q1", "q2", "Car A front", "In lane original"),
              _edge("q2", "q3", "Car B back", "In lane original"), _edge("q3", "q4", "Car B left"),
              _edge("q4", "q_accept", "Car B front", "In lane original")),
             "Approach each car closely before passing. The same compact regions apply to A and B.",
             Guard(("Car A back","In lane original"))),
    Scenario(10, "Reach a waypoint while avoiding a zone",
             (("q0", "In lane 1 outside Z"), ("q1", "In lane 2 outside Z"),
              ("q_accept", "At W in lane 2 outside Z")),
             (_edge("q0", "q1", "In lane 2", "Avoid zone Z"),
              _edge("q1", "q_accept", "Waypoint W", "In lane 2", "Avoid zone Z")),
             "Detour around the red rectangle, reach the green waypoint rectangle, then confirm for 1.2 s.",
             Guard(("In lane 1","Avoid zone Z")), Guard(negative=("Avoid zone Z",))),
    Scenario(11, "Yield and merge behind a faster car",
             (("q0", "Car A left"), ("q1", "Car A back"), ("q_accept", "Car A velocity")),
             (_edge("q0", "q1", "Car A back", negative=("Car A personal",)),
              _edge("q1", "q_accept", "Car A back", "Car A velocity", negative=("Car A personal",))),
             "Yield beside faster A, join behind it, then match its speed at a close following distance.",
             Guard(("Car A left",)), Guard(("Car A personal",)),
             state_vocabulary=(("q1",Guard(("Car A back",))),
                               ("q_accept",Guard(("Car A velocity",))))),
    Scenario(12, "Work-zone detour and return",
             (("q0", "In lane 1"), ("q1", "In lane 2"), ("q_accept", "Waypoint W")),
             (_edge("q0", "q1", "In lane 2", "Avoid zone Z"),
              _edge("q1", "q_accept", "Waypoint W", "In lane 1", "Avoid zone Z")),
             "Leave lane 1 before Z, pass the work zone in lane 2, and return to the waypoint in lane 1.",
             Guard(("In lane 1","Avoid zone Z")), Guard(negative=("Avoid zone Z",))),
    Scenario(13, "Slalom overtakes around a work zone",
             (("q0", "Car A back"), ("q1", "Car A left"), ("q2", "Car A front"),
              ("q3", "Car B back"), ("q4", "Car B right"), ("q_accept", "Car B front")),
             (_edge("q0", "q1", "Car A left", "Avoid zone Z"),
              _edge("q1", "q2", "Car A front", "In lane original", "Avoid zone Z"),
              _edge("q2", "q3", "Car B back", "In lane original", "Avoid zone Z"),
              _edge("q3", "q4", "Car B right", "Avoid zone Z"),
              _edge("q4", "q_accept", "Car B front", "In lane original", "Avoid zone Z")),
             "Pass A on the left, return, then pass B on the right to stay clear of the work zone.",
             Guard(("Car A back","Avoid zone Z")), Guard(negative=("Avoid zone Z",)),
             state_vocabulary=tuple((name,Guard((atom,))) for name,atom in
                                    (("q0","Car A back"),("q1","Car A left"),("q2","Car A front"),
                                     ("q3","Car B back"),("q4","Car B right"),("q_accept","Car B front")))),
    Scenario(14, "Match speed beside A, then merge ahead",
             (("q0", "Car A left"), ("q1", "Car A velocity"), ("q_accept", "Car A front")),
             (_edge("q0", "q1", "Car A left", "Car A velocity", negative=("Car A personal",)),
              _edge("q1", "q_accept", "Car A front", "In lane 2", negative=("Car A personal",))),
             "First synchronize beside A. Then accelerate, move just ahead, and finish merging into A's lane.",
             Guard(("Car A left",),("Car A velocity",)), Guard(("Car A personal",)),
             state_vocabulary=(("q0",Guard(("Car A left",))),
                               ("q1",Guard(("Car A velocity",))),
                               ("q_accept",Guard(("Car A front",))))),
    Scenario(15, "Two waypoints around a forbidden zone",
             (("q0", "In lane 1"), ("q1", "Waypoint W1"), ("q_accept", "Waypoint W2")),
             (_edge("q0", "q1", "Waypoint W1", "Avoid zone Z"),
              _edge("q1", "q_accept", "Waypoint W2", "In lane 3", "Avoid zone Z")),
             "Visit W1 in lane 2, then W2 in lane 3. The red zone blocks continuing straight after W1.",
             Guard(("In lane 1","Avoid zone Z")), Guard(negative=("Avoid zone Z",))),
)

SCENARIO_PAGE_SIZE = 5


@dataclass(frozen=True)
class Event:
    time: float
    edge: Edge
    facts: frozenset[str]


@dataclass
class DemoDFA:
    """Fixed scenario monitor with an explicit finishing confirmation gate."""
    scenario: Scenario
    state: str = "q0"
    history: list[Event] = field(default_factory=list)
    last_edge: Edge | None = None
    last_time: float = -math.inf
    confirmation_since: float | None = None
    confirmation_duration: float = 1.2

    def confirmation_elapsed(self, time):
        return 0. if self.confirmation_since is None else max(0.,time-self.confirmation_since)

    @property
    def terminal(self):
        return self.state in ("q_accept", "q_fail")

    def outgoing(self, state):
        if state in ("q_accept", "q_fail"):
            return ()
        edges = tuple(e for e in self.scenario.edges if e.source == state)
        if self.scenario.failure:
            return (Edge(state, "q_fail", self.scenario.failure), *edges)
        return edges

    def stay_label(self, state):
        guards = [e.guard.label for e in self.outgoing(state)]
        # Terminal states are absorbing; express their tautology using a real AP.
        if not guards:
            atom = self.scenario.edges[0].guard
            name = (atom.positive + atom.negative)[0]
            return f"{name} OR NOT {name}"
        return "NOT (" + " OR ".join(f"({g})" if " AND " in g else g for g in guards) + ")"

    def update(self, facts, time, *, maneuver_complete=False):
        # Exactly one transition per measured physics tick. Failure guards and
        # progress guards are disjoint; simultaneous progress edges are bugs.
        matches = [e for e in self.outgoing(self.state) if e.guard.holds(facts)]
        if len(matches) > 1:
            raise AssertionError(f"Overlapping DFA guards: {matches}")
        final = next((e for e in self.outgoing(self.state) if e.target == "q_accept"),None)
        if final is not None:
            if final.guard.holds(facts) and maneuver_complete:
                if self.confirmation_since is None:
                    self.confirmation_since = time
            else:
                self.confirmation_since = None
            matches = [e for e in matches if e.target != "q_accept" or
                       self.confirmation_elapsed(time)+1e-9 >= self.confirmation_duration]
        elif not self.terminal:
            self.confirmation_since = None
        if matches:
            edge = matches[0]
            self.state, self.last_edge, self.last_time = edge.target, edge, time
            self.history.append(Event(time, edge, frozenset(p for p, value in facts.items() if value)))
        return self.state


class TrafficCar(ControlledVehicle):
    """Existing lane/speed controller with deterministic acceleration limits."""
    def act(self, action=None):
        super().act(action)
        self.action["acceleration"] = float(np.clip(self.action["acceleration"], -4., 2.))


class Simulator(v2.Simulator):
    tick_duration = 1. / 30
    step_duration = .2

    def __init__(self, scenario=1, seed=7, geometry=Geometry()):
        if scenario not in range(1, len(SCENARIOS)+1):
            raise ValueError(f"Choose a scenario from 1 through {len(SCENARIOS)}")
        self.scenario = SCENARIOS[scenario - 1]
        self.base_geometry = geometry
        self.geometry = geometry
        super().__init__(seed=seed)

    def reset(self, new_seed=False):
        if new_seed:
            self.seed += 1
        h = self.env.unwrapped
        self.geometry = (replace(self.base_geometry,back_horizon=.14)
                         if self.scenario.number == 7 else self.base_geometry)
        # One physics tick per step: monitor every simulated position, rather
        # than sampling only at UI/label frequency. No unused normalized data.
        h.configure({"lanes_count": 3, "vehicles_count": 1, "duration": 60,
                     "simulation_frequency": 30, "policy_frequency": 30,
                     "observation": {"type": "Kinematics", "vehicles_count": 3,
                                     "normalize": False, "absolute": True},
                     "screen_width": 1100, "screen_height": 330,
                     "scaling": 4.5, "centering_position": [.28, .5]})
        # Zoom into the maneuvers; map scenes retain the wider view so their
        # waypoint and forbidden rectangle fit alongside the approaching ego.
        if self.scenario.number not in (8,10,12,15):
            h.configure({"screen_width":760,"screen_height":240,
                         "scaling":7.,"centering_position":[.4,.5]})
        elif self.scenario.number in (10,12,15):
            h.configure({"centering_position":[.10,.5]})
        self.env.reset(seed=self.seed)
        self.waypoint = self.zone = None
        self.dfa = DemoDFA(self.scenario)
        self.done, self.steps, self.motion_ticks = False, 0, 0
        self.error = None
        self.state_entered = 0.
        self.control_phase = ""
        self.b_brake_started = None
        self.traffic_events = []
        self.returning_car = None
        self.cancel_phase = "first"
        self.cancellations = []
        n = self.scenario.number
        start_lane = 0 if n in (5,10,11,12,14,15) else 2 if n == 8 else 1
        speed = 22. if n == 6 else 26. if n == 7 else 20. if n in (5,8,10) else 18.
        ego = h.vehicle
        ego.position = np.array([100., start_lane * 4.])
        ego.heading, ego.speed = 0., speed
        ego.lane_index = ego.target_lane_index = ("0", "1", start_lane)
        ego.lane = h.road.network.get_lane(ego.lane_index)
        ego.target_speed = speed
        ego.color = ACCENT
        h.road.vehicles = [ego]
        self.actors = {}
        if n in (1,2,3,4,9,13):
            half = ego.LENGTH/2+self.geometry.d0+self.geometry.headway*16.
            self._add_car("A",100.+half+3.,1,16.)
        if n in (3, 4):
            self._add_car("B",124.,0 if n == 3 else 2,18.)
        elif n == 6:
            self._add_car("A",100.+self.geometry.half_length(ego)+5.,1,22.)
        elif n == 7:
            half = self.geometry.half_length(ego)
            self._add_car("A", 100.+half+5., 1, 26.)
        elif n == 8:
            self._add_car("B", 160., 1, 22.)
            h.road.network.get_lane(("0", "1", 2)).length = 260.
            self.waypoint = Rectangle((145.,8.),10.,1.8)
            self.zone = Rectangle((253.,8.),8.,2.)
        elif n in (9,13):
            self._add_car("B",self.actors["A"].position[0]+64.,1,16.)
            if n == 13:
                self.zone = Rectangle((350.,0.),32.,1.8)
        elif n == 10:
            self.waypoint = Rectangle((300.,4.),18.,1.8)
            self.zone = Rectangle((210.,0.),8.,1.8)
        elif n in (11,14):
            self._add_car("A",100.,1,22.)
        elif n == 12:
            self.waypoint = Rectangle((290.,0.),22.,1.8)
            self.zone = Rectangle((210.,0.),20.,1.8)
        elif n == 15:
            self.waypoint = {"W1":Rectangle((165.,4.),18.,1.8),
                             "W2":Rectangle((270.,8.),22.,1.8)}
            self.zone = Rectangle((218.,4.),15.,1.8)
        self.lanes = {str(i + 1): ("0", "1", i) for i in range(3)}
        self.lanes["original"] = ("0", "1", start_lane)
        if n == 8:
            self.lanes["main"] = ("0", "1", 1)
        h._semantic_ids = {}
        self.snapshot = v2.extract_snapshot(self.env)
        self.cars = {name: h._semantic_ids[car] for name, car in self.actors.items()}
        self.labels = set()
        self.refresh()
        self.reference_x, self.reference_speed = 100., speed
        self.reference_acceleration, self.reference_sample = 0., -1
        self.measurements = []
        self._record()

    def _add_car(self, name, x, lane, speed):
        h = self.env.unwrapped
        car = TrafficCar(h.road, [x, lane * 4.], speed=speed, target_speed=speed)
        car.color = (100, 178, 255) if name == "A" else (255, 191, 91)
        h.road.vehicles.append(car)
        self.actors[name] = car

    @property
    def speed(self):
        return float(np.linalg.norm(self.snapshot.ego.velocity))

    def select(self, number):
        if number not in range(1, len(SCENARIOS)+1):
            raise ValueError(f"Choose a scenario from 1 through {len(SCENARIOS)}")
        self.scenario = SCENARIOS[number - 1]
        self.reset()

    def refresh(self):
        self.facts = evaluate_vocabulary(self.snapshot, self.cars, self.lanes,
                                         self.waypoint, self.zone, self.geometry)
        self.labels = {name for name, value in self.facts.items() if value}

    def _pass(self, name, side, passing_speed=24.):
        car = next(v for v in self.snapshot.vehicles if v.uid == self.cars[name])
        s, _ = relative_position(self.snapshot.ego, car)
        # This controller clearance is an execution detail, never a DFA atom.
        # Return early enough to remain inside the finite front rectangle.
        half = self.geometry.half_length(car)
        if self.returning_car == name or s > half+3.:
            self.returning_car = name
            # Match the passed car instead of racing out of its compact front
            # region while steering and confirming the completed maneuver.
            reference_speed = float(np.linalg.norm(car.velocity))
            speed = reference_speed+float(np.clip(.8*(half+4.-s),-4.,4.))
            return 1,speed,f"Return close ahead of {name}; match its speed"
        _, lateral = relative_position(self.snapshot.ego,car)
        speed = 18. if abs(lateral) < 2. else passing_speed
        return side,speed,f"Pass close beside {name}"

    def _follow_target(self, name):
        car, ego = self.actors[name], self.env.unwrapped.vehicle
        gap = car.position[0]-ego.position[0]
        desired = self.geometry.half_length(car)+4.
        acceleration = .7*(gap-desired)+1.6*(car.speed-ego.speed)
        return max(0.,ego.speed+acceleration/ego.KP_A)

    def _script_braking_car(self):
        state, time = self.dfa.state, self.snapshot.time
        car = self.actors["B"]
        if self.cancel_phase == "first" and state == "q1" and time-self.state_entered >= .7:
            if self.b_brake_started is None:
                self.b_brake_started = time
                self.traffic_events.append((time,"B brakes in the overtaking lane"))
            car.target_speed = 8.
        elif self.cancel_phase == "brake":
            car.target_speed = 8.
        elif self.cancel_phase in ("recover", "retry"):
            if car.target_speed != 30.:
                self.traffic_events.append((time,"B accelerates away; retry can clear"))
            car.target_speed = 30.

    def _canceled_overtake(self, side):
        """Script the maneuver without turning controller phases into DFA nodes."""
        if self.cancel_phase == "brake" and self.dfa.state == "q0":
            self.cancel_phase = "recover"
        self._script_braking_car()
        if (self.cancel_phase in ("first", "retry") and self.dfa.state == "q1"
                and self.b_brake_started is not None
                and self.facts["Car B back"] and not self.facts["Car B velocity"]
                and self.actors["B"].speed+self.geometry.velocity_epsilon < self.speed):
            self.cancel_phase = "brake"
            self.cancellations.append((self.snapshot.time, frozenset(self.labels)))
            self.traffic_events.append((self.snapshot.time, "Ego brakes and cancels the overtake"))
        if self.cancel_phase == "brake":
            self.returning_car = None
            car = self.actors["A"]
            behind = car.position[0]-self.snapshot.ego.position[0] > self.geometry.half_length(car)+3.
            return (1 if behind else side), 8., "Brake, fall behind A, and return"
        if self.cancel_phase == "recover":
            car = self.actors["A"]
            gap = car.position[0]-self.snapshot.ego.position[0]
            ready = (self.actors["B"].position[0]-self.snapshot.ego.position[0] >= 32.
                     and self.actors["B"].speed >= 22. and self.facts["Car A back"])
            if ready:
                self.cancel_phase = "retry"
            else:
                desired = self.geometry.half_length(car)+4.
                return 1, float(np.clip(car.speed+.7*(gap-desired),8.,24.)), "Stay close behind A; wait for B"
        return self._pass("A", side)

    def _control(self):
        n, state = self.scenario.number, self.dfa.state
        lane, speed, phase = 1, self.speed, "Cruise"
        if n in (1, 2, 3, 4):
            side = 0 if n in (1, 3) else 2
            if n in (3,4):
                lane, speed, phase = self._canceled_overtake(side)
            else:
                lane, speed, phase = self._pass("A", side)
        elif n == 5:
            lane, speed, phase = (1 if state == "q0" else 2), 20., "Move toward target lane"
        elif n in (6, 7):
            car = self.actors["A"]
            car.target_speed = ((12. if self.snapshot.time >= 2. else 22.) if n == 6
                                else (14. if self.snapshot.time >= 1. else 26.))
            if state == "q0":
                speed = 22. if n == 6 else 26.
                phase = "Follow close; react at the near-back region" if n == 7 else "Hold initial speed"
            else:
                speed = car.speed
                if n == 7:
                    half = self.geometry.half_length(car)
                    gap = car.position[0]-self.snapshot.ego.position[0]
                    speed += .5*(gap-half-.9*self.geometry.back_length(car))
                phase = "Brake near personal space, then match A" if n == 7 else "Match A's measured velocity"
        elif n == 8:
            lane, speed, phase = (2 if state == "q0" else 1), 20., "Approach W" if state == "q0" else "Join main lane"
        elif n in (9,13):
            if state in ("q0", "q1"):
                lane, speed, phase = self._pass("A", 0)
            elif state == "q2":
                lane, speed, phase = 1, 24., "Approach B"
            else:
                lane, speed, phase = self._pass("B", 2 if n == 13 else 0)
        elif n == 10:
            lane, speed, phase = 1, 20., "Detour around Z toward W"
        elif n == 11:
            if state == "q0":
                car = self.actors["A"]
                gap = car.position[0]-self.snapshot.ego.position[0]
                lane = 1 if gap > self.geometry.half_length(car)+3. else 0
                speed, phase = 16., "Yield to A, then merge behind"
            else:
                lane, speed, phase = 1, self._follow_target("A"), "Match A and stabilize the following gap"
        elif n == 12:
            cleared = self.snapshot.ego.position[0] > self.zone.center[0]+self.zone.half_length+5.
            lane, speed = (0 if state != "q0" and cleared else 1), 18.
            phase = "Return to lane 1 and W" if cleared else "Detour around the work zone"
        elif n == 14:
            if state == "q0":
                lane, speed, phase = 0, self.actors["A"].speed, "Synchronize while staying beside A"
            else:
                lane, speed, phase = self._pass("A",0,passing_speed=26.)
        elif n == 15:
            lane, speed, phase = (1 if state == "q0" else 2), 18., "Visit W1, then W2 via lane 3"
        self.control_phase = phase
        ego = self.env.unwrapped.vehicle
        ego.target_lane_index = ("0", "1", lane)
        # Reuse the proportional controller and V3's bounded setpoint method.
        if n == 7 and state != "q0":
            acceleration = float(np.clip(3.*(speed-ego.speed),-10.,2.))
            ego.target_speed = max(0.,ego.speed+acceleration/ego.KP_A)
        else:
            ego.target_speed = float(np.clip(speed, max(0., ego.speed - 4. / ego.KP_A), ego.speed + 2. / ego.KP_A))
        if n in (3,4) and self.b_brake_started is None:
            self.actors["B"].target_speed = ego.target_speed

    def maneuver_complete(self):
        ego = self.env.unwrapped.vehicle
        lane = self.snapshot.road.network.get_lane(ego.target_lane_index)
        longitudinal, lateral = lane.local_coordinates(self.snapshot.ego.position)
        heading_error = (self.snapshot.ego.heading-lane.heading_at(longitudinal)+math.pi) % (2*math.pi)-math.pi
        aligned = (self.snapshot.ego.lane_index == ego.target_lane_index
                   and abs(lateral) <= .15 and abs(heading_error) <= .02)
        if self.scenario.number in (6,7):
            lead = self.actors["A"]
            return aligned and abs(lead.speed-lead.target_speed) <= .25
        return aligned

    def _record(self):
        if self.scenario.number not in (6, 7):
            return
        car = self.actors["A"]
        if self.scenario.number == 6 and self.snapshot.time > 0:
            sample = int(self.snapshot.time)
            if sample != self.reference_sample:
                self.reference_sample = sample
                self.reference_acceleration = 2. if self.reference_speed < car.speed else -4.
            self.reference_x += self.reference_speed * self.tick_duration
            self.reference_speed = max(0., self.reference_speed + self.reference_acceleration * self.tick_duration)
        self.measurements.append((self.snapshot.time, car.position[0] - self.snapshot.ego.position[0],
                                  car.position[0] - self.reference_x, self.speed, car.speed))

    def advance(self):
        if self.done:
            return False
        self._control()
        _, _, terminated, truncated, _ = self.env.step(self.actions["IDLE"])
        self.motion_ticks += 1
        self.steps = self.motion_ticks // 6
        self.snapshot = v2.extract_snapshot(self.env)
        self.refresh()
        old = self.dfa.state
        self.dfa.update(self.facts, self.snapshot.time,maneuver_complete=self.maneuver_complete())
        if old != self.dfa.state:
            self.state_entered = self.snapshot.time
            self.control_phase = dict(self.scenario.stages).get(self.dfa.state,"Violation")
        self._record()
        self.done = self.dfa.terminal or bool(terminated or truncated)
        if self.done and not self.dfa.terminal:
            self.error = "Simulation ended before the DFA reached a terminal state"
        return True

    def step(self):
        if self.done:
            return False
        for _ in range(6):
            if not self.advance() or self.done:
                break
        return True


class Visualizer(Dashboard, v2.Visualizer):
    """Reuse the project's road renderer, camera, cards, arrows and text cache."""
    def __init__(self, simulator, playback=1., size=None):
        import pygame
        self.pg, self.sim = pygame, simulator
        pygame.init()
        self.camera_follow = True
        from types import SimpleNamespace
        self.camera_anchor = SimpleNamespace(position=simulator.snapshot.ego.position.copy())
        self.frame = self.render_frame(reset_camera=True)
        desktop = pygame.display.get_desktop_sizes()[0]
        size = size or (max(1160, min(1460, desktop[0] - 40)), max(780, min(920, desktop[1] - 90)))
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption(f"Highway V4 - {len(SCENARIOS)} driving scenarios and live DFAs")
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small = pygame.font.SysFont("segoeui", 15)
        self.graph_font = pygame.font.SysFont("segoeui", 14)
        self.node_font = pygame.font.SysFont("consolas", 13)
        self.clock = pygame.time.Clock()
        self.play, self.playback = False, playback
        self.buttons, self.edge_hits = [], []
        self.accumulator = 0.
        self.graph_scroll = 0.
        self.graph_focus = None
        self.graph_scroll_max = 0.
        self.scenario_page = (simulator.scenario.number-1)//SCENARIO_PAGE_SIZE
        self.selector_scenario = simulator.scenario.number
        self.update_layout()

    def update_layout(self):
        width, height = self.screen.get_size()
        left = int(width * .46)
        self.road_rect = self.pg.Rect(14, 64, left - 20, min(296, int(height*.32)))
        self.road_image_rect = self.road_rect.inflate(-16, -40)
        self.road_image_rect.y = self.road_rect.y + 32
        self.graph_rect = self.pg.Rect(left + 6, 64, width - left - 20, height - 78)
        self.left_width = self.road_rect.width

    def render_frame(self, reset_camera=False):
        h = self.sim.env.unwrapped
        if h.viewer is None:
            self.sim.env.render()
        # highway-env disables its drawing under SDL dummy by default. Its
        # offscreen surfaces can still draw normally for visual verification.
        if h.viewer.offscreen:
            h.viewer.enabled = True
        # Rebuild the cached surface after a selector changes scene zoom.
        viewer = h.viewer
        desired_size = (h.config["screen_width"],h.config["screen_height"])
        if viewer.sim_surface.get_size() != desired_size:
            from highway_env.road.graphics import WorldSurface
            viewer.sim_surface = WorldSurface(desired_size,0,self.pg.Surface(desired_size))
        viewer.sim_surface.scaling = h.config["scaling"]
        viewer.sim_surface.centering_position = h.config["centering_position"]
        return v2.Visualizer.render_frame(self, reset_camera)

    @staticmethod
    def _text_lines(text, width, font):
        lines, line = [], ""
        for word in text.split():
            candidate = (line + " " + word).strip()
            if line and font.size(candidate)[0] > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        return lines

    def _wrapped(self, text, x, y, width, color=MUTED, font=None):
        font = font or self.small
        for line in self._text_lines(text,width,font):
            self._label(line, x, y, color, font)
            y += font.get_linesize()
        return y

    def _draw_scene(self):
        pg, sim = self.pg, self.sim
        reference = next(iter(sim.actors.items()),None)
        note = (f"{reference[0]} front {sim.geometry.front_length(reference[1]):.1f} m / back {sim.geometry.back_length(reference[1]):.1f} m"
                if reference else "Rectangular map regions")
        self._card(self.road_rect, "Driving environment", note)
        picture = pg.surfarray.make_surface(self.frame.swapaxes(0, 1))
        pg.draw.rect(self.screen, (65, 123, 67), self.road_image_rect)
        self.screen.blit(pg.transform.smoothscale(picture, self.road_image_rect.size), self.road_image_rect)
        viewer = sim.env.unwrapped.viewer
        if viewer is None:
            return
        def project(point):
            px, py = viewer.sim_surface.vec2pix(np.array(point))
            return (self.road_image_rect.x + px * self.road_image_rect.width / self.frame.shape[1],
                    self.road_image_rect.y + py * self.road_image_rect.height / self.frame.shape[0])
        old_clip = self.screen.get_clip()
        self.screen.set_clip(self.road_image_rect)
        layer = pg.Surface(self.screen.get_size(), pg.SRCALPHA)
        by_id = {car.uid: car for car in sim.snapshot.vehicles}
        for name, uid in sim.cars.items():
            car = by_id[uid]
            half = sim.geometry.half_length(car)
            lane = sim.snapshot.road.network.get_lane(car.lane_index)
            width = lane.width_at(lane.local_coordinates(car.position)[0])
            forward = np.array([math.cos(car.heading), math.sin(car.heading)])
            right = np.array([-forward[1], forward[0]])
            # Draw precisely the evaluated rectangles, including front/back and
            # the side strips, in the car's heading-aligned frame.
            regions = ((-half, half, -width/2, width/2, (255, 105, 111)),
                       (-half, half, -1.5*width, -width/2, (100, 178, 255)),
                       (-half, half, width/2, 1.5*width, (100, 178, 255)),
                       (half, half+sim.geometry.front_length(car), -width/2, width/2, (85, 211, 164)),
                       (-half-sim.geometry.back_length(car), -half, -width/2, width/2, (255, 191, 91)))
            for s0, s1, d0, d1, color in regions:
                points = [project(car.position + s*forward + d*right) for s, d in ((s0,d0),(s1,d0),(s1,d1),(s0,d1))]
                pg.draw.polygon(layer, (*color, 28), points)
                pg.draw.polygon(layer, (*color, 170), points, 1)
        markers = [(region,(79,226,178),name) for name,region in waypoint_regions(sim.waypoint)]
        markers.append((sim.zone,(255,105,111),"Z"))
        for region, color, caption in markers:
            if region:
                center = project(region.center)
                rx = abs(project((region.center[0]+region.half_length,region.center[1]))[0]-center[0])
                ry = abs(project((region.center[0],region.center[1]+region.half_width))[1]-center[1])
                box = pg.Rect(round(center[0]-rx),round(center[1]-ry),max(2,round(2*rx)),max(2,round(2*ry)))
                pg.draw.rect(layer, (*color, 65), box)
                pg.draw.rect(layer, (*color, 255), box, 2)
                self._label(caption, center[0] - 5, center[1] - 9, color, self.font)
        self.screen.blit(layer, (0, 0))
        for name, car, color in [("EGO", sim.snapshot.ego, ACCENT)] + [(name, by_id[uid], (100,178,255) if name == "A" else (255,191,91)) for name,uid in sim.cars.items()]:
            x, y = project(car.position)
            self._label(name, x - 12, y - 30, color, self.small)
        self.screen.set_clip(old_clip)

    def _draw_graph(self):
        pg, sim, rect = self.pg, self.sim, self.graph_rect
        dfa, stages = sim.dfa, sim.scenario.stages
        visited = {"q0"} | {event.edge.target for event in dfa.history}
        progress = sum(name in visited for name, _ in stages)
        self._card(rect, "Scenario state graph", f"{progress} / {len(stages)} stages")
        self._label("Each node shows its entry vocabulary · green dot = true now",
                    rect.x+14,rect.y+35,MUTED,self.graph_font)
        track = pg.Rect(rect.x+14,rect.y+61,rect.width-28,3)
        pg.draw.rect(self.screen,BORDER,track,border_radius=2)
        pg.draw.rect(self.screen,ACCENT,(track.x,track.y,round(track.width*progress/len(stages)),3))
        self.graph_view = pg.Rect(rect.x+10,rect.y+79,rect.width-20,rect.height-242)
        node_width = rect.width-(204 if sim.scenario.failure else 92)
        left = rect.x+55
        positions = {}
        self.state_labels = {}
        self.state_truth = {}
        cursor = 27.
        for index,(name,_) in enumerate(stages):
            vocabulary = sim.scenario.vocabulary_for(name)
            self.state_labels[name] = vocabulary.label
            self.state_truth[name] = vocabulary.holds(sim.facts)
            condition_lines = self._text_lines(vocabulary.label,node_width-60,self.graph_font)
            node_height = 38+len(condition_lines)*self.graph_font.get_linesize()
            positions[name] = pg.Rect(left,round(cursor),node_width,node_height)
            if index < len(stages)-1:
                next_name = stages[index+1][0]
                edge = next(e for e in sim.scenario.edges if e.source == name and e.target == next_name)
                lines = self._text_lines(edge.guard.label,node_width-57,self.graph_font)
                gap = max(34,len(lines)*self.graph_font.get_linesize()+12)
                if next_name == "q_accept":
                    gap += 21
                cursor += node_height+gap
            else:
                cursor += node_height+15
        if cursor < self.graph_view.height-24:
            extra = (self.graph_view.height-24-cursor)/max(1,len(stages)-1)
            for index,(name,_) in enumerate(stages):
                positions[name].y += round(index*extra)
            cursor += extra*(len(stages)-1)
        if sim.scenario.failure:
            condition = sim.scenario.vocabulary_for("q_fail")
            self.state_labels["q_fail"] = condition.label
            self.state_truth["q_fail"] = condition.holds(sim.facts)
            fail_height = 38+len(self._text_lines(condition.label,94,self.graph_font))*self.graph_font.get_linesize()
            positions["q_fail"] = pg.Rect(rect.right-122,positions["q_accept"].y,110,fail_height)
            cursor = max(cursor,positions["q_fail"].bottom+15)
        self.graph_scroll_max = max(0.,cursor-self.graph_view.height)
        key = (sim.scenario.number,dfa.state,dfa.confirmation_since is not None)
        if key != self.graph_focus:
            target = positions["q_accept" if dfa.confirmation_since is not None else dfa.state]
            if target.bottom-self.graph_scroll > self.graph_view.height-12:
                self.graph_scroll = target.bottom-self.graph_view.height+12
            if target.top-self.graph_scroll < 12:
                self.graph_scroll = target.top-12
            self.graph_focus = key
        self.graph_scroll = float(np.clip(self.graph_scroll,0.,self.graph_scroll_max))
        nodes = {name:box.move(0,round(self.graph_view.y-self.graph_scroll))
                 for name,box in positions.items()}
        self.node_rects, self.edge_hits = nodes, []
        traversed = {event.edge for event in dfa.history}
        hot = dfa.last_edge if sim.snapshot.time-dfa.last_time < .85 else None

        def color_for(edge):
            return ((255,213,93) if edge == hot else (100,189,163)
                    if edge in traversed else (74,95,119))

        old_clip = self.screen.get_clip()
        self.screen.set_clip(self.graph_view)
        first = nodes["q0"]
        self._arrow((first.x+21,first.y-23),(first.x+21,first.y-3),TEXT,2)
        self._label("START",first.x+35,first.y-23,MUTED,self.node_font)
        for edge in sim.scenario.edges:
            source, target = nodes[edge.source],nodes[edge.target]
            color = color_for(edge)
            if target.y > source.y:
                rail = source.x+21
                self._arrow((rail,source.bottom),(rail,target.top-2),color,2)
                lines = self._text_lines(edge.guard.label,node_width-57,self.graph_font)
                label_height = len(lines)*self.graph_font.get_linesize()+8
                if edge.target == "q_accept":
                    label_height += 21
                label = pg.Rect(source.x+41,source.bottom+(target.top-source.bottom-label_height)//2,
                                node_width-45,label_height)
                pg.draw.rect(self.screen,(25,38,54),label,border_radius=5)
                label_bottom = self._wrapped(edge.guard.label,label.x+6,label.y+4,label.width-12,
                                             color if edge in traversed else TEXT,self.graph_font)
                if edge.target == "q_accept":
                    seconds = min(dfa.confirmation_duration,dfa.confirmation_elapsed(sim.snapshot.time))
                    confirming = dfa.state == edge.source and dfa.confirmation_since is not None
                    caption = (f"Confirming {seconds:.1f} / 1.2 s" if confirming else "Align, then confirm for 1.2 s")
                    self._label(caption,label.x+6,label_bottom+1,(255,200,103),self.node_font)
                    ratio = seconds/dfa.confirmation_duration if confirming or dfa.state == "q_accept" else 0.
                    pg.draw.rect(self.screen,(255,200,103),(label.x,label.bottom-2,round(label.width*ratio),2))
                self.edge_hits.append((label,edge.guard.label))
            else:
                # Cancellation is an actual back edge, on its own outside rail.
                rail = rect.x+32
                pg.draw.lines(self.screen,color,False,
                              [(source.left,source.centery),(rail,source.centery),
                               (rail,target.centery),(target.left-3,target.centery)],2)
                self._arrow((rail,target.centery),(target.left-3,target.centery),color,2)
                caption = pg.transform.rotate(self.node_font.render(edge.guard.label,True,TEXT),90)
                label = caption.get_rect(midright=(rail-6,(source.centery+target.centery)//2))
                self.screen.blit(caption,label)
                self.edge_hits.append((label,edge.guard.label))
        if sim.scenario.failure:
            fail = nodes["q_fail"]
            rail = fail.centerx
            for name,_ in stages[:-1]:
                source = nodes[name]
                edge = Edge(name,"q_fail",sim.scenario.failure)
                color = color_for(edge) if edge in traversed else (136,72,89)
                pg.draw.lines(self.screen,color,False,
                              [(source.right+2,source.centery),(rail,source.centery),
                               (rail,fail.top-3)],1)
            self._arrow((rail,fail.top-15),(rail,fail.top-2),(255,118,135),2)
            caption = pg.transform.rotate(self.node_font.render(
                sim.scenario.failure.label,True,(255,146,159)),90)
            label = caption.get_rect(midright=(rail-6,(first.centery+fail.top)//2))
            self.screen.blit(caption,label)
            self.edge_hits.append((label,sim.scenario.failure.label))

        for index,(name,description) in enumerate(stages):
            node, active = nodes[name],name == dfa.state
            complete = name in visited and not active
            color = ACCENT if name == "q_accept" or complete else (111,182,255)
            fill = (30,58,75) if active else (23,43,48) if complete else (25,35,49)
            if active:
                pg.draw.rect(self.screen,color,node.inflate(6,6),1,border_radius=11)
            pg.draw.rect(self.screen,fill,node,border_radius=8)
            pg.draw.rect(self.screen,color if active or complete or name == "q_accept" else BORDER,
                         node,2 if active else 1,border_radius=8)
            if name == "q_accept":
                pg.draw.rect(self.screen,ACCENT,node.inflate(-8,-8),1,border_radius=5)
            badge = (node.x+21,node.centery)
            pg.draw.circle(self.screen,color if active or complete else (50,66,85),badge,12)
            marker = "OK" if name == "q_accept" else str(index+1)
            self._label(marker,badge[0]-self.node_font.size(marker)[0]/2,badge[1]-7,BG,self.node_font)
            title = name if description == self.state_labels[name] else f"{name}   {description}"
            self._label(title,node.x+43,node.y+6,TEXT,self.graph_font,max_width=node.width-64)
            truth_color = (126,231,192) if self.state_truth[name] else MUTED
            pg.draw.circle(self.screen,truth_color if self.state_truth[name] else (77,94,114),
                           (node.right-12,node.y+15),3)
            self._wrapped(self.state_labels[name],node.x+43,node.y+28,node.width-60,truth_color,self.graph_font)
            # Fold complementary self-loop labels into the persistent inspector.
            loop = pg.Rect(node.right-18,node.top-9,22,16)
            pg.draw.arc(self.screen,MUTED,loop,.1,5.5,1)
            self._arrow((loop.right-3,loop.centery),(loop.right-5,loop.bottom-1),MUTED)
            self.edge_hits.append((loop,dfa.stay_label(name)))
            self.edge_hits.append((node,self.state_labels[name]))
        if sim.scenario.failure:
            node = nodes["q_fail"]
            color = (255,115,135)
            pg.draw.rect(self.screen,(65,32,46),node,border_radius=8)
            pg.draw.rect(self.screen,color,node,3 if dfa.state == "q_fail" else 1,border_radius=8)
            self._label("q_fail",node.x+10,node.y+6,color,self.node_font)
            pg.draw.circle(self.screen,ACCENT if self.state_truth["q_fail"] else (77,94,114),
                           (node.right-12,node.y+15),3)
            self._wrapped(self.state_labels["q_fail"],node.x+8,node.y+28,node.width-16,color,self.graph_font)
            self.edge_hits.append((node,self.state_labels["q_fail"]))
        self.screen.set_clip(old_clip)
        if self.graph_scroll_max:
            scroll_track = pg.Rect(self.graph_view.right-3,self.graph_view.y+3,2,self.graph_view.height-6)
            pg.draw.rect(self.screen,BORDER,scroll_track)
            thumb_height = max(24,scroll_track.height*self.graph_view.height/cursor)
            thumb_y = scroll_track.y+(scroll_track.height-thumb_height)*self.graph_scroll/self.graph_scroll_max
            pg.draw.rect(self.screen,MUTED,(scroll_track.x-1,round(thumb_y),4,round(thumb_height)),border_radius=2)

        # Persistent guard inspector and predicate chips never cover the graph.
        footer = rect.bottom-152
        pg.draw.line(self.screen,BORDER,(rect.x+14,footer-9),(rect.right-14,footer-9))
        heading, detail = "NEXT TRANSITION", ""
        outgoing = [e for e in dfa.outgoing(dfa.state) if e.target != "q_fail"]
        detail = " OR ".join(e.guard.label for e in outgoing) or dfa.stay_label(dfa.state)
        mouse = pg.mouse.get_pos()
        if self.graph_view.collidepoint(mouse):
            for hit,label in self.edge_hits:
                if hit.collidepoint(mouse):
                    heading,detail = "GUARD INSPECTOR",label
                    break
        accepting = next((e for e in outgoing if e.target == "q_accept"),None)
        if accepting and accepting.guard.holds(sim.facts):
            detail = accepting.guard.label
            seconds = min(dfa.confirmation_duration,dfa.confirmation_elapsed(sim.snapshot.time))
            heading = (f"CONFIRM FINISH  ·  {seconds:.1f} / {dfa.confirmation_duration:.1f} s"
                       if sim.maneuver_complete() else "FINISH STEERING BEFORE CONFIRMING")
        if dfa.state == "q_accept":
            heading = "ACCEPTED  ·  MANEUVER FINISHED"
        self._label(heading,rect.x+14,footer,ACCENT if dfa.state == "q_accept" else (255,200,103),self.graph_font)
        self._wrapped(detail,rect.x+14,footer+21,rect.width-28,TEXT,self.graph_font)
        self._label("TRUE NOW",rect.x+14,footer+76,MUTED,self.node_font)
        x,y = rect.x+14,footer+96
        for predicate in sorted(sim.labels):
            chip_width = self.graph_font.size(predicate)[0]+14
            if x+chip_width > rect.right-14:
                x,y = rect.x+14,y+24
            chip = pg.Rect(x,y,chip_width,20)
            pg.draw.rect(self.screen,(26,55,51),chip,border_radius=5)
            self._label(predicate,x+7,y+2,(138,233,201),self.graph_font)
            x += chip_width+6

    def _draw_gap_plot(self, rect):
        pg, rows = self.pg, self.sim.measurements
        if len(rows) < 2:
            return
        self._card(rect, "Following distance (m)")
        data = np.array(rows)
        series = ((1, ACCENT, "Measured ego"), (2,(255,191,91),"1 s switching reference")) if self.sim.scenario.number == 6 else ((1,ACCENT,"Measured ego"),)
        if self.sim.scenario.number == 7:
            boundary = self.sim.actors["A"].LENGTH/2+self.sim.geometry.d0+self.sim.geometry.headway*data[:,4]
            data = np.column_stack((data,boundary))
            series += ((5,(255,115,135),"Personal-space boundary"),)
        plot = rect.inflate(-28,-66)
        plot.y += 14
        low, high = data[:,[i for i,_,_ in series]].min()-3, data[:,[i for i,_,_ in series]].max()+3
        for i,color,caption in series:
            points = [(plot.x+t/max(data[-1,0],.01)*plot.width,plot.bottom-(v-low)/(high-low)*plot.height) for t,v in data[:,[0,i]]]
            pg.draw.lines(self.screen,color,False,points,2)
        self._label(f"{low:.0f}",plot.x,plot.bottom-10,MUTED,self.graph_font)
        self._label(f"{high:.0f}",plot.x,plot.y,MUTED,self.graph_font)
        for j,(_,color,caption) in enumerate(series):
            self._label(caption,rect.x+12+j*185,rect.bottom-23,color,self.graph_font)

    def draw(self):
        pg, sim = self.pg, self.sim
        self.screen.fill(BG)
        self.buttons = []
        if self.selector_scenario != sim.scenario.number:
            self.scenario_page = (sim.scenario.number-1)//SCENARIO_PAGE_SIZE
            self.selector_scenario = sim.scenario.number
        self._label(f"{sim.scenario.number:02d}  {sim.scenario.title}",16,13,TEXT,self.font)
        status = "ACCEPTED" if sim.dfa.state == "q_accept" else "FAILED" if sim.dfa.state == "q_fail" else "ENDED" if sim.done else "PLAYING" if self.play else "PAUSED"
        self._label(f"{status} | t={sim.snapshot.time:.2f}s | ego {sim.speed:.1f} m/s",16,37,ACCENT if sim.dfa.state == "q_accept" else MUTED,self.small)
        self._draw_scene()
        width = self.left_width
        controls_y = self.road_rect.bottom + 11
        for i,(name,caption) in enumerate((("Play","Pause [P]" if self.play else "Play [P]"),("Step","Step [Space]"),("Reset","Replay [R]"),("Slower","Speed -"),("Faster","Speed +"))):
            self._button(pg.Rect(14+i*(width//5+1),controls_y,width//5-5,30),name,caption,not(sim.done and name in ("Play","Step")))
        self._label(f"Playback {self.playback:g}x | {sim.control_phase or 'Ready'}",16,controls_y+38,MUTED,self.small,max_width=width)
        y = self._wrapped(sim.scenario.note,16,controls_y+66,width-8,MUTED,self.small)
        self._label("Scenarios",16,y+9,TEXT,self.small)
        page_count = math.ceil(len(SCENARIOS)/SCENARIO_PAGE_SIZE)
        tab_width = (width-94)//page_count
        for page in range(page_count):
            first = page*SCENARIO_PAGE_SIZE+1
            last = min(first+SCENARIO_PAGE_SIZE-1,len(SCENARIOS))
            self._button(pg.Rect(102+page*tab_width,y+5,tab_width-6,26),
                         f"Page {page}",f"{first}–{last}",active=page==self.scenario_page)
        y += 42
        start = self.scenario_page*SCENARIO_PAGE_SIZE
        visible_scenarios = SCENARIOS[start:start+SCENARIO_PAGE_SIZE]
        for i,scenario in enumerate(visible_scenarios):
            button = pg.Rect(14,y+i*34,width-2,29)
            shortcut = f"  [F{scenario.number-10}]" if scenario.number > 10 else ""
            self._button(button,f"Scenario {scenario.number}",f"{scenario.number}. {scenario.title}{shortcut}",
                         active=scenario.number==sim.scenario.number)
        y += 181
        if sim.scenario.number in (6,7):
            self._draw_gap_plot(pg.Rect(14,y,width,max(115,self.screen.get_height()-y-14)))
        else:
            self._label("Observed DFA path",16,y,TEXT,self.small)
            path = "q0" + "".join(f" -> {event.edge.target}" for event in sim.dfa.history)
            self._wrapped(path,16,y+25,width,ACCENT,self.graph_font)
            if sim.dfa.last_edge:
                event = sim.dfa.history[-1]
                self._wrapped(f"t={event.time:.2f}s: {event.edge.guard.label}",16,y+72,width,TEXT,self.graph_font)
        self._draw_graph()
        pg.display.flip()

    def command(self, name):
        if name == "Play":
            self.play = not self.play and not self.sim.done
            self.accumulator = 0.
        elif name == "Step":
            self.play = False
            self.sim.step()
        elif name == "Reset":
            self.play = False
            self.sim.reset()
            self.accumulator = 0.
            self.graph_scroll,self.graph_focus = 0.,None
        elif name.startswith("Scenario "):
            self.sim.select(int(name.split()[-1]))
            self.play = False
            self.accumulator = 0.
            self.graph_scroll,self.graph_focus = 0.,None
            self.scenario_page = (self.sim.scenario.number-1)//SCENARIO_PAGE_SIZE
            self.selector_scenario = self.sim.scenario.number
        elif name.startswith("Page ") or name in ("Previous page","Next page"):
            page_count = math.ceil(len(SCENARIOS)/SCENARIO_PAGE_SIZE)
            page = (int(name.split()[-1]) if name.startswith("Page ") else
                    self.scenario_page+(-1 if name == "Previous page" else 1))
            self.scenario_page = page % page_count
        elif name in ("Slower", "Faster"):
            self.playback = float(np.clip(self.playback*(.5 if name == "Slower" else 2),.25,4.))
        self.frame = self.render_frame(reset_camera=name == "Reset" or name.startswith("Scenario "))

    def handle_event(self, event):
        pg = self.pg
        if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
            return False
        if event.type == pg.VIDEORESIZE:
            self.screen = pg.display.set_mode((max(1160,event.w),max(780,event.h)),pg.RESIZABLE)
            self.update_layout()
        elif event.type == pg.KEYDOWN and not getattr(event,"repeat",False):
            keys = {pg.K_p:"Play",pg.K_SPACE:"Step",pg.K_r:"Reset",pg.K_MINUS:"Slower",pg.K_EQUALS:"Faster",
                    pg.K_LEFTBRACKET:"Previous page",pg.K_RIGHTBRACKET:"Next page"}
            keys.update({getattr(pg,f"K_F{i+1}"):f"Scenario {11+i}" for i in range(5)})
            if event.key in keys:
                self.command(keys[event.key])
            elif getattr(event,"unicode","") in "1234567890" and getattr(event,"unicode",""):
                self.command(f"Scenario {10 if event.unicode == '0' else int(event.unicode)}")
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            for rect,name,enabled in self.buttons:
                if enabled and rect.collidepoint(event.pos):
                    self.command(name)
                    break
        elif event.type == pg.MOUSEWHEEL and self.graph_rect.collidepoint(pg.mouse.get_pos()):
            self.graph_scroll = float(np.clip(self.graph_scroll-event.y*52,0.,self.graph_scroll_max))
        return True

    def run(self):
        running = True
        while running:
            elapsed = min(self.clock.tick(60)/1000.,.1)
            for event in self.pg.event.get():
                if not self.handle_event(event):
                    running = False
            if self.play and running:
                self.accumulator += elapsed*self.playback
                while self.accumulator >= self.sim.tick_duration and not self.sim.done:
                    self.sim.advance()
                    self.accumulator -= self.sim.tick_duration
                self.play = not self.sim.done
                self.frame = self.render_frame()
            self.draw()


def verify_all(screenshots=None):
    """Execute every real scene, verify all guards and optionally render graphs."""
    if screenshots:
        os.environ.setdefault("SDL_VIDEODRIVER","dummy")
        os.environ.setdefault("SDL_AUDIODRIVER","dummy")
        Path(screenshots).mkdir(parents=True, exist_ok=True)
    sim = Simulator()
    view = Visualizer(sim,size=(1460,920)) if screenshots else None
    def capture(suffix):
        if view:
            view.frame = view.render_frame(reset_camera=True)
            view.draw()
            view.pg.image.save(view.screen,str(Path(screenshots)/f"scenario-{sim.scenario.number:02d}-{suffix}.png"))
    try:
        for scenario in SCENARIOS:
            sim.select(scenario.number)
            assert scenario.initial.holds(sim.facts), "Initial state has no matching vocabulary"
            capture("q0")
            captured_confirmation = False
            while not sim.done and sim.snapshot.time < 50.:
                old_state = sim.dfa.state
                sim.advance()
                if sim.dfa.state != old_state:
                    capture(sim.dfa.state)
                if (not captured_confirmation and not sim.dfa.terminal
                        and sim.dfa.confirmation_elapsed(sim.snapshot.time) >= .6):
                    capture("confirming")
                    captured_confirmation = True
            if sim.dfa.state != "q_accept":
                raise AssertionError(f"Scenario {scenario.number}: {sim.dfa.state} at {sim.snapshot.time:.2f}s; {sim.error}")
            expected = list(scenario.expected_path)
            observed = ["q0",*(event.edge.target for event in sim.dfa.history)]
            if observed != expected:
                raise AssertionError(f"Scenario {scenario.number} took {observed}; expected {expected}")
            for event in sim.dfa.history:
                facts = {name: name in event.facts for name in sim.facts}
                assert event.edge.guard.holds(facts), event
                assert scenario.vocabulary_for(event.edge.target).holds(facts), event
            assert sim.maneuver_complete(), "Accepted before the maneuver finished"
            assert (sim.dfa.confirmation_elapsed(sim.snapshot.time)+1e-9
                    >= sim.dfa.confirmation_duration), "Accepted before the confirmation hold"
            assert not any(car.crashed for car in sim.snapshot.vehicles), "Physical collision during demonstration"
            print(f"{scenario.number:02d} PASS {sim.snapshot.time:5.2f}s | {' -> '.join(observed)}")
    finally:
        sim.close()
        if view:
            view.pg.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=int, choices=range(1,len(SCENARIOS)+1), default=1)
    parser.add_argument("--list", action="store_true", help="List all demos")
    parser.add_argument("--verify", action="store_true", help="Run every demo and check its observed DFA path")
    parser.add_argument("--screenshots", type=Path, help="With --verify, render every scene and graph to this directory")
    parser.add_argument("--headless-steps", type=int, help="Run this many 0.2 s steps without opening a window")
    parser.add_argument("--play", action="store_true", help="Start playback immediately")
    args = parser.parse_args()
    if args.list:
        for scenario in SCENARIOS:
            print(f"{scenario.number:2d}. {scenario.title}")
        return
    if args.screenshots and not args.verify:
        parser.error("--screenshots requires --verify")
    if args.verify:
        verify_all(args.screenshots)
        return
    sim = Simulator(args.scenario)
    try:
        if args.headless_steps is not None:
            for _ in range(max(0,args.headless_steps)):
                if not sim.step():
                    break
            print(f"Scenario {args.scenario}: {sim.dfa.state}, t={sim.snapshot.time:.2f}s, speed={sim.speed:.2f} m/s")
            for event in sim.dfa.history:
                print(f"  {event.time:.2f}s {event.edge.source} -> {event.edge.target}: {event.edge.guard.label}")
            if sim.done and sim.dfa.state != "q_accept":
                raise SystemExit(1)
        else:
            view = Visualizer(sim)
            view.play = args.play
            view.run()
    finally:
        sim.close()
        import pygame
        pygame.quit()


if __name__ == "__main__":
    main()
