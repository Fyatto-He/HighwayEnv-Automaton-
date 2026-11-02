"""Geometry boundaries, deterministic paths, failure guards and rendered demos."""
import itertools
import math
import os
from dataclasses import replace
from types import SimpleNamespace
import unittest

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np

import mainV2 as v2
import mainV4 as m


class Vocabulary(unittest.TestCase):
    def setUp(self):
        self.geometry = m.Geometry()
        self.car = v2.Vehicle(2, np.array([0., 4.]), np.array([16., 0.]),
                              0., 5., 2., ("0", "1", 1))
        self.ego = replace(self.car, uid=1)
        lane = SimpleNamespace(width_at=lambda s: 4., local_coordinates=lambda p: (p[0], p[1]-4.),
                               on_lane=lambda p: abs(p[1]-4.) <= 2.)
        self.road = SimpleNamespace(network=SimpleNamespace(get_lane=lambda index: lane))

    def evaluate(self, s, d=0., velocity=None, heading=0., waypoint=None, zone=None, car_speed=16.):
        forward = np.array([math.cos(heading), math.sin(heading)])
        right = np.array([-forward[1], forward[0]])
        car = replace(self.car, heading=heading, velocity=car_speed*forward)
        ego = replace(self.ego, position=car.position+s*forward+d*right,
                      velocity=car.velocity if velocity is None else np.array(velocity))
        snapshot = v2.Snapshot(0.,ego,(ego,car),self.road)
        return m.evaluate_vocabulary(snapshot,{"A":2},{"original":car.lane_index},waypoint,zone,self.geometry)

    def test_exact_longitudinal_and_lateral_boundaries(self):
        h = self.geometry.half_length(self.car)
        for s in (-h,0.,h):
            for d in (-2.,2.):
                facts = self.evaluate(s,d)
                self.assertTrue(facts["Car A personal"])
                self.assertFalse(facts["Car A left"])
                self.assertFalse(facts["Car A right"])
        for d in (-6.,-2.00001):
            self.assertTrue(self.evaluate(0.,d)["Car A left"])
        for d in (2.00001,6.):
            self.assertTrue(self.evaluate(0.,d)["Car A right"])
        self.assertFalse(self.evaluate(0.,-6.00001)["Car A left"])
        self.assertFalse(self.evaluate(0.,6.00001)["Car A right"])
        self.assertFalse(self.evaluate(h)["Car A front"])
        self.assertTrue(self.evaluate(h+.00001)["Car A front"])
        self.assertTrue(self.evaluate(h+self.geometry.front_length(self.car))["Car A front"])
        self.assertFalse(self.evaluate(h+self.geometry.front_length(self.car)+.00001)["Car A front"])
        self.assertFalse(self.evaluate(-h)["Car A back"])
        self.assertTrue(self.evaluate(-h-.00001)["Car A back"])
        self.assertTrue(self.evaluate(-h-self.geometry.back_length(self.car))["Car A back"])
        self.assertFalse(self.evaluate(-h-self.geometry.back_length(self.car)-.00001)["Car A back"])

    def test_front_and_back_extend_from_personal_space_in_proportion_to_speed(self):
        for speed in (0.,8.,16.,32.):
            car = replace(self.car,velocity=np.array([speed,0.]))
            half = self.geometry.half_length(car)
            front,back = self.geometry.front_length(car),self.geometry.back_length(car)
            self.assertAlmostEqual(front,.9*speed)
            self.assertAlmostEqual(back,.75*speed)
            self.assertFalse(self.evaluate(half,car_speed=speed)["Car A front"])
            self.assertFalse(self.evaluate(-half,car_speed=speed)["Car A back"])
            self.assertFalse(self.evaluate(half+front+.001,car_speed=speed)["Car A front"])
            self.assertFalse(self.evaluate(-half-back-.001,car_speed=speed)["Car A back"])
            if speed:
                self.assertTrue(self.evaluate(half+.001,car_speed=speed)["Car A front"])
                self.assertTrue(self.evaluate(-half-.001,car_speed=speed)["Car A back"])
        self.assertLess(self.geometry.front_length(self.car),20.)
        self.assertLess(self.geometry.back_length(self.car),20.)

    def test_heading_frame_and_vector_velocity_matching(self):
        for heading in (0.,math.pi/2,math.pi,-math.pi/3):
            with self.subTest(heading=heading):
                self.assertTrue(self.evaluate(0.,-4.,heading=heading)["Car A left"])
                self.assertTrue(self.evaluate(0.,4.,heading=heading)["Car A right"])
                self.assertTrue(self.evaluate(15.,heading=heading)["Car A front"])
        epsilon = self.geometry.velocity_epsilon
        self.assertTrue(self.evaluate(0.,velocity=(16.,epsilon-.00001))["Car A velocity"])
        self.assertFalse(self.evaluate(0.,velocity=(16.,epsilon))["Car A velocity"])
        self.assertFalse(self.evaluate(0.,velocity=(-16.,0.))["Car A velocity"])
        faster = replace(self.car, velocity=np.array([0.,30.]))
        self.assertAlmostEqual(self.geometry.half_length(faster), 5./2+.8+.35*30.)
        self.assertLess(self.geometry.half_length(faster), .5*(5./2+2.+30.))

    def test_rectangular_boundaries_and_point_lane_membership(self):
        rectangle = m.Rectangle((0.,4.),10.,2.)
        for s in (-10.,10.):
            for d in (-2.,2.):
                facts = self.evaluate(s,d,waypoint=rectangle,zone=rectangle)
                self.assertTrue(facts["Waypoint W"])
                self.assertFalse(facts["Avoid zone Z"])
        facts = self.evaluate(10.00001,waypoint=rectangle,zone=rectangle)
        self.assertFalse(facts["Waypoint W"])
        self.assertTrue(facts["Avoid zone Z"])
        self.assertFalse(self.evaluate(0.,2.00001,waypoint=rectangle)["Waypoint W"])
        self.assertTrue(self.evaluate(0.,2.)["In lane original"])
        self.assertFalse(self.evaluate(0.,2.00001)["In lane original"])

    def test_named_waypoints_reuse_the_same_rectangle_predicate(self):
        regions = {"W1":m.Rectangle((0.,4.),10.,2.),"W2":m.Rectangle((30.,8.),6.,1.8)}
        facts = self.evaluate(0.,waypoint=regions)
        self.assertTrue(facts["Waypoint W1"])
        self.assertFalse(facts["Waypoint W2"])
        self.assertNotIn("Waypoint W",facts)
        facts = self.evaluate(30.,4.,waypoint=regions)
        self.assertFalse(facts["Waypoint W1"])
        self.assertTrue(facts["Waypoint W2"])


class DFA(unittest.TestCase):
    def test_every_state_has_a_vocabulary_witness(self):
        for scenario in m.SCENARIOS:
            self.assertNotIn("q_finish",scenario.states)
            self.assertNotIn("q_retry",scenario.states)
            for state in scenario.states:
                self.assertTrue(scenario.vocabulary_for(state).label)
            with self.assertRaisesRegex(ValueError,"vocabulary"):
                replace(scenario,stages=scenario.stages+(("q_wait","Wait for a timer"),))

    def test_every_boolean_valuation_is_deterministic(self):
        self.assertEqual([scenario.number for scenario in m.SCENARIOS],list(range(1,16)))
        allowed = {f"Car {car} {region}" for car in ("A","B")
                   for region in ("personal","left","right","front","back","velocity")}
        allowed |= {f"In lane {lane}" for lane in ("1","2","3","original","main")}
        allowed |= {"Waypoint W","Waypoint W1","Waypoint W2","Avoid zone Z"}
        for scenario in m.SCENARIOS:
            dfa = m.DemoDFA(scenario)
            for state in scenario.states:
                edges = dfa.outgoing(state)
                atoms = sorted({p for edge in edges for p in edge.guard.positive+edge.guard.negative})
                self.assertTrue(set(atoms) <= allowed)
                for values in itertools.product((False,True),repeat=len(atoms)):
                    facts = dict(zip(atoms,values))
                    matches = [e for e in edges if e.guard.holds(facts)]
                    self.assertLessEqual(len(matches),1,(scenario.number,state,facts))
                    monitor = m.DemoDFA(scenario,state=state,confirmation_since=0.)
                    self.assertEqual(monitor.update(facts,2.,maneuver_complete=True),
                                     matches[0].target if matches else state)

    def test_retry_cancellation_is_a_real_loop(self):
        for number,side in ((3,"left"),(4,"right")):
            scenario = m.SCENARIOS[number-1]
            self.assertEqual(scenario.states,("q0","q1","q_accept"))
            self.assertEqual([scenario.vocabulary_for(state).label for state in scenario.states],
                             ["Car A back",f"Car A {side}","Car A front"])
            dfa = m.DemoDFA(scenario,state="q1")
            facts = {f"Car A {side}":False,"Car A front":False,"Car A back":True}
            self.assertEqual(dfa.update(facts,1.),"q0")
            self.assertEqual(dfa.update(facts,2.),"q0")
            facts[f"Car A {side}"] = True
            facts["Car A back"] = False
            self.assertEqual(dfa.update(facts,3.),"q1")
            facts[f"Car A {side}"] = False
            facts["Car A front"] = True
            self.assertEqual(dfa.update(facts,4.),"q1")
            self.assertEqual(dfa.update(facts,5.,maneuver_complete=True),"q1")
            self.assertEqual(dfa.update(facts,6.2,maneuver_complete=True),"q_accept")

    def test_acceptance_waits_for_alignment_and_an_uninterrupted_hold(self):
        for scenario in m.SCENARIOS:
            edge = next(e for e in scenario.edges if e.target == "q_accept")
            wait_state = edge.source
            dfa = m.DemoDFA(scenario,state=wait_state)
            facts = {p:False for e in dfa.outgoing(wait_state)
                     for p in e.guard.positive+e.guard.negative}
            facts.update(dict.fromkeys(edge.guard.positive,True))
            self.assertEqual(dfa.update(facts,1.),wait_state)
            self.assertIsNone(dfa.confirmation_since)
            self.assertEqual(dfa.update(facts,2.,maneuver_complete=True),wait_state)
            self.assertEqual(dfa.update(facts,3.1,maneuver_complete=True),wait_state)
            # Losing physical alignment resets the confirmation, even though
            # every vocabulary predicate in the final condition is still true.
            self.assertEqual(dfa.update(facts,3.2,maneuver_complete=False),wait_state)
            self.assertIsNone(dfa.confirmation_since)
            self.assertEqual(dfa.update(facts,4.,maneuver_complete=True),wait_state)
            self.assertEqual(dfa.update(facts,5.19,maneuver_complete=True),wait_state)
            self.assertEqual(dfa.update(facts,5.2,maneuver_complete=True),"q_accept")

    def test_losing_the_final_condition_restarts_confirmation(self):
        dfa = m.DemoDFA(m.SCENARIOS[0],state="q1")
        facts = {"Car A front":True,"In lane original":True}
        dfa.update(facts,0.,maneuver_complete=True)
        facts["Car A front"] = False
        self.assertEqual(dfa.update(facts,1.,maneuver_complete=True),"q1")
        self.assertIsNone(dfa.confirmation_since)
        facts["Car A front"] = True
        self.assertEqual(dfa.update(facts,2.,maneuver_complete=True),"q1")
        self.assertEqual(dfa.update(facts,3.19,maneuver_complete=True),"q1")
        self.assertEqual(dfa.update(facts,3.2,maneuver_complete=True),"q_accept")

    def test_unsafe_measurement_fails_even_with_progress_conditions(self):
        for number in (7,8,10,11,12,13,14,15):
            scenario = m.SCENARIOS[number-1]
            for state,_ in scenario.stages[:-1]:
                dfa = m.DemoDFA(scenario,state=state)
                atoms = {p for edge in dfa.outgoing(state) for p in edge.guard.positive+edge.guard.negative}
                facts = dict.fromkeys(atoms,True)
                if number not in (7,11,14):
                    facts["Avoid zone Z"] = False
                self.assertEqual(dfa.update(facts,1.),"q_fail")
                self.assertEqual(dfa.update(dict.fromkeys(atoms,False),2.),"q_fail")

    def test_two_waypoint_dfa_requires_the_configured_order(self):
        dfa = m.DemoDFA(m.SCENARIOS[14])
        facts = {"Waypoint W1":False,"Waypoint W2":True,"In lane 3":True,"Avoid zone Z":True}
        self.assertEqual(dfa.update(facts,0.,maneuver_complete=True),"q0")
        facts.update({"Waypoint W1":True,"Waypoint W2":False})
        self.assertEqual(dfa.update(facts,1.,maneuver_complete=True),"q1")
        self.assertEqual(dfa.update(facts,2.,maneuver_complete=True),"q1")
        facts.update({"Waypoint W1":False,"Waypoint W2":True})
        self.assertEqual(dfa.update(facts,3.,maneuver_complete=True),"q1")
        self.assertEqual(dfa.update(facts,4.19,maneuver_complete=True),"q1")
        self.assertEqual(dfa.update(facts,4.2,maneuver_complete=True),"q_accept")


class Demonstrations(unittest.TestCase):
    def test_all_fifteen_real_scenes_accept_and_replay_deterministically(self):
        sim = m.Simulator()
        try:
            for number in range(1,len(m.SCENARIOS)+1):
                baseline = None
                for replay in range(2):
                    with self.subTest(number=number,replay=replay):
                        sim.select(number)
                        self.assertEqual(sim.dfa.state,"q0")
                        self.assertEqual(sim.dfa.history,[])
                        self.assertTrue(sim.scenario.initial.holds(sim.facts))
                        if number in (1,2,3,4,9):
                            self.assertTrue(sim.facts["Car A back"])
                            gap = sim.actors["A"].position[0]-sim.snapshot.ego.position[0]
                            self.assertLessEqual(gap-sim.geometry.half_length(sim.actors["A"]),3.01)
                        near_margins = []
                        cancel_speeds = []
                        observed_lanes = {sim.snapshot.ego.lane_index[-1]}
                        while not sim.done and sim.snapshot.time < 50.:
                            sim.advance()
                            observed_lanes.add(sim.snapshot.ego.lane_index[-1])
                            if number == 7:
                                self.assertFalse(sim.facts["Car A personal"])
                                near_margins.append(sim.actors["A"].position[0]-sim.snapshot.ego.position[0]
                                                    -sim.geometry.half_length(sim.actors["A"]))
                            if number in (3,4):
                                self.assertFalse(sim.facts["Car A personal"])
                                if sim.cancel_phase == "brake":
                                    cancel_speeds.append(sim.speed)
                            if number in (8,10,12,13,15):
                                self.assertTrue(sim.facts["Avoid zone Z"])
                            if number in (11,13,14):
                                for car in sim.cars:
                                    self.assertFalse(sim.facts[f"Car {car} personal"])
                        self.assertEqual(sim.dfa.state,"q_accept",sim.error)
                        self.assertFalse(any(car.crashed for car in sim.snapshot.vehicles))
                        self.assertTrue(sim.maneuver_complete())
                        self.assertGreaterEqual(sim.dfa.confirmation_elapsed(sim.snapshot.time)+1e-9,1.2)
                        self.assertEqual(["q0",*(e.edge.target for e in sim.dfa.history)],
                                         list(sim.scenario.expected_path))
                        for event in sim.dfa.history:
                            facts = {name:name in event.facts for name in sim.facts}
                            self.assertTrue(event.edge.guard.holds(facts))
                            self.assertTrue(sim.scenario.vocabulary_for(event.edge.target).holds(facts))
                        trace = [(e.time,e.edge,e.facts) for e in sim.dfa.history]
                        if baseline is None:
                            baseline = trace
                        else:
                            self.assertEqual(trace,baseline)
                        if number in (3,4):
                            first, returned = sim.dfa.history[:2]
                            self.assertEqual(len(sim.cancellations),1)
                            cancel_time,cancel_facts = sim.cancellations[0]
                            self.assertGreater(cancel_time-first.time,.3)
                            self.assertLess(first.time,sim.b_brake_started)
                            self.assertLess(sim.b_brake_started,cancel_time)
                            self.assertLess(cancel_time,returned.time)
                            self.assertIn("Car B back",cancel_facts)
                            self.assertNotIn("Car B velocity",cancel_facts)
                            self.assertNotIn("Car B personal",cancel_facts)
                            self.assertLess(min(cancel_speeds),10.)
                            self.assertGreater(sim.actors["B"].speed,24.)
                        if number == 7:
                            self.assertLess(min(near_margins),3.)
                            reaction = sim.dfa.history[0]
                            self.assertGreater(reaction.time,1.5)
                        if number == 6:
                            self.assertLess(sim.speed,13.)
                            data = np.array(sim.measurements)
                            self.assertGreater(data[:,3].max()-data[:,3].min(),8.)
                        if number in (11,14):
                            self.assertEqual(observed_lanes,{0,1})
                            self.assertEqual(sim.snapshot.ego.lane_index[-1],1)
                            if number == 11:
                                self.assertTrue(sim.facts["Car A back"])
                                self.assertTrue(sim.facts["Car A velocity"])
                            else:
                                self.assertIn("Car A left",sim.dfa.history[0].facts)
                                self.assertIn("Car A velocity",sim.dfa.history[0].facts)
                                self.assertTrue(sim.facts["Car A front"])
                        if number == 12:
                            self.assertEqual(observed_lanes,{0,1})
                            self.assertEqual(sim.snapshot.ego.lane_index[-1],0)
                            self.assertTrue(sim.facts["Waypoint W"])
                        if number == 13:
                            self.assertEqual(observed_lanes,{0,1,2})
                            self.assertIn("Car A left",sim.dfa.history[0].facts)
                            self.assertIn("Car B right",sim.dfa.history[3].facts)
                            self.assertNotIn("Car B left",sim.dfa.history[3].facts)
                        if number == 15:
                            self.assertEqual(observed_lanes,{0,1,2})
                            self.assertIn("Waypoint W1",sim.dfa.history[0].facts)
                            self.assertTrue(sim.facts["Waypoint W2"])
                        self.assertFalse(sim.advance())
        finally:
            sim.close()

    def test_render_and_ui_controls_do_not_advance_the_dfa(self):
        sim = m.Simulator()
        view = m.Visualizer(sim,size=(1160,780))
        try:
            for number in range(1,len(m.SCENARIOS)+1):
                view.command(f"Scenario {number}")
                snapshot = sim.snapshot
                view.draw()
                view.frame = view.render_frame()
                view.draw()
                self.assertIs(sim.snapshot,snapshot)
                self.assertEqual(sim.dfa.history,[])
                self.assertEqual(set(view.node_rects),set(sim.scenario.states))
                self.assertEqual(view.state_labels,{state:sim.scenario.vocabulary_for(state).label
                                                   for state in sim.scenario.states})
                self.assertEqual(view.scenario_page,(number-1)//m.SCENARIO_PAGE_SIZE)
                visible = {name for _,name,_ in view.buttons if name.startswith("Scenario ")}
                first = view.scenario_page*m.SCENARIO_PAGE_SIZE+1
                self.assertEqual(visible,{f"Scenario {i}" for i in range(first,first+m.SCENARIO_PAGE_SIZE)})
                for button,name,_ in view.buttons:
                    if name.startswith("Scenario "):
                        self.assertTrue(view.screen.get_rect().contains(button))
                # Detect the black-frame regression in highway-env's dummy SDL
                # renderer, not just a successfully created surface.
                self.assertGreater(float(view.frame.std()),15.)
                view.command("Step")
                self.assertGreater(sim.snapshot.time,0.)
                view.command("Reset")
                self.assertEqual(sim.snapshot.time,0.)
                self.assertEqual(sim.scenario.number,number)
                self.assertEqual(sim.dfa.history,[])
            event = view.pg.event.Event(view.pg.VIDEORESIZE,w=1460,h=920)
            self.assertTrue(view.handle_event(event))
            view.draw()
            # The canceled examples now have just three vocabulary nodes.
            # Check their compact graph at the smaller window size as well.
            self.assertTrue(view.handle_event(view.pg.event.Event(
                view.pg.VIDEORESIZE,w=1160,h=780)))
            view.command("Scenario 3")
            view.draw()
            self.assertEqual(len(view.node_rects),3)
            self.assertEqual(set(view.state_labels.values()),
                             {"Car A back","Car A left","Car A front"})
            view.graph_scroll = view.graph_scroll_max
            view.draw()
            self.assertTrue(view.graph_view.colliderect(view.node_rects["q_accept"]))
        finally:
            sim.close()
            view.pg.quit()

    def test_scenario_pages_mouse_and_function_keys_select_all_new_demos(self):
        sim = m.Simulator()
        view = m.Visualizer(sim,size=(1160,780))
        try:
            for page in range(3):
                snapshot = sim.snapshot
                view.command(f"Page {page}")
                view.draw()
                self.assertIs(sim.snapshot,snapshot)
                self.assertEqual(sim.dfa.history,[])
            for number in range(11,16):
                key = getattr(view.pg,f"K_F{number-10}")
                self.assertTrue(view.handle_event(view.pg.event.Event(view.pg.KEYDOWN,key=key,unicode="")))
                view.draw()
                self.assertEqual(sim.scenario.number,number)
                self.assertEqual(view.scenario_page,2)
                self.assertEqual(sim.snapshot.time,0.)
            # Browsing away from the running scene leaves it loaded.
            view.command("Page 0")
            view.draw()
            page_button = next(rect for rect,name,_ in view.buttons if name == "Page 2")
            self.assertTrue(view.handle_event(view.pg.event.Event(
                view.pg.MOUSEBUTTONDOWN,button=1,pos=page_button.center)))
            view.draw()
            scene_button = next(rect for rect,name,_ in view.buttons if name == "Scenario 13")
            self.assertTrue(view.handle_event(view.pg.event.Event(
                view.pg.MOUSEBUTTONDOWN,button=1,pos=scene_button.center)))
            self.assertEqual(sim.scenario.number,13)
            snapshot = sim.snapshot
            view.command("Next page")
            view.draw()
            self.assertEqual(view.scenario_page,0)
            self.assertIs(sim.snapshot,snapshot)
            view.command("Previous page")
            self.assertEqual(view.scenario_page,2)
        finally:
            sim.close()
            view.pg.quit()


if __name__ == "__main__":
    unittest.main()
