"""Annotated native highway-env rendering, including intermediate physics frames."""
import math

import pygame
from highway_env.envs.common.graphics import EnvViewer
from highway_env.road.graphics import RoadGraphics


class TrafficViewer(EnvViewer):
    """Use the simulator's camera and traffic renderer; add only explanations."""

    def __init__(self, env, task):
        super().__init__(env)
        # Offscreen surfaces still draw under SDL's dummy driver.
        self.enabled = True
        self.task = task
        self.font = pygame.font.Font(None, 24)
        self.frame_callback = None

    def display(self):
        if not self.enabled:
            return
        surface = self.sim_surface
        surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, surface)
        RoadGraphics.display_road_objects(self.env.road, surface, offscreen=self.offscreen)
        RoadGraphics.display_traffic(self.env.road, surface,
                                    simulation_frequency=self.env.config["simulation_frequency"],
                                    offscreen=self.offscreen)
        self._draw_vehicle_labels()
        self._draw_front_gap()
        self._draw_hud()
        if not self.offscreen:
            self.screen.blit(surface, (0, 0))
            self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()
        # Native highway-env calls display() between policy decisions too. The
        # callback records these real physics frames without altering the policy.
        if self.frame_callback is not None:
            self.frame_callback(self.get_image())

    def _draw_vehicle_labels(self):
        ego = self.env.vehicle
        x, y = self.sim_surface.pos2pix(*ego.position)
        self.sim_surface.blit(self.font.render("EGO", True, (90, 210, 255)), (x - 15, y - 33))
        lead = getattr(self.task, "lead_vehicle", None)
        if lead is not None:
            x, y = self.sim_surface.pos2pix(*lead.position)
            caption = "INITIAL CAR" if self.task.scenario == "quickstart" else "SLOW LEAD"
            text = self.font.render(caption, True, (250, 130, 230))
            self.sim_surface.blit(text, (x - text.get_width() // 2, y + 15))

    def _draw_front_gap(self):
        front, _ = self.env.road.neighbour_vehicles(self.env.vehicle, self.env.vehicle.lane_index)
        gap = self.task.closest_front_gap()
        if front is None or gap is None or not math.isfinite(gap):
            return
        color = (70, 220, 130) if gap >= 15 else (255, 195, 85) if gap >= 5 else (255, 90, 90)
        start = self.sim_surface.pos2pix(*self.env.vehicle.position)
        end = self.sim_surface.pos2pix(*front.position)
        pygame.draw.line(self.sim_surface, color, start, end, 2)
        label = self.font.render(f"{gap:.1f} m gap", True, color)
        self.sim_surface.blit(label, ((start[0] + end[0] - label.get_width()) // 2, start[1] - 25))

    def _draw_hud(self):
        task, surface = self.task, self.sim_surface
        metrics = task.metrics
        gap = task.closest_front_gap()
        gap_label = f"{gap:.1f} m" if gap is not None and math.isfinite(gap) else "clear"
        color = (255, 95, 95) if self.env.vehicle.crashed else (110, 240, 150)
        seconds = self.env.steps / self.env.config["simulation_frequency"]
        scenario = "Native traffic / quickstart DQN" if task.scenario == "quickstart" else "Slower traffic / historical DQN"
        lines = [
            (f"{task.status}    Speed: {self.env.vehicle.speed:.1f} m/s    "
             f"Front gap: {gap_label}    Time: {seconds:.1f}/{self.env.config['duration']:.0f} s", color),
            (f"Encounters: {metrics['encounter_count']}    Overtakes: {metrics['overtake_count']}    "
             f"Lane changes: {metrics['lane_changes']}    Following brakes: {metrics['following_brake_steps']}", (225, 230, 245)),
            (f"EGO: BLUE    Traffic: AMBER    Initial reference car: MAGENTA    {scenario}", (180, 200, 220)),
        ]
        pygame.draw.rect(surface, (20, 27, 40), (0, 0, surface.get_width(), 88))
        for index, (text, text_color) in enumerate(lines):
            surface.blit(self.font.render(text, True, text_color), (14, 8 + index * 25))
