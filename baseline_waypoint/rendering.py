"""A small EnvViewer extension; never edits the installed simulator."""
import pygame

from highway_env.envs.common.graphics import EnvViewer
from highway_env.road.graphics import RoadGraphics


class GoalViewer(EnvViewer):
    def __init__(self, env, task):
        super().__init__(env)
        self.task = task
        self.font = pygame.font.Font(None, 25)
        self.title_font = pygame.font.Font(None, 32)

    def display(self):
        if not self.enabled:
            return
        # The same WorldSurface transform positions both cars and the goal.
        surface = self.sim_surface
        surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, surface)
        RoadGraphics.display_road_objects(self.env.road, surface, offscreen=self.offscreen)
        RoadGraphics.display_traffic(self.env.road, surface,
                                    simulation_frequency=self.env.config["simulation_frequency"],
                                    offscreen=self.offscreen)
        self._draw_goal()
        self._draw_hud()
        if not self.offscreen:
            self.screen.blit(surface, (0, 0))
            self.clock.tick(self.env.config["policy_frequency"])
            pygame.display.flip()

    def _draw_goal(self):
        surface, task = self.sim_surface, self.task
        center = surface.pos2pix(*task.goal_position)
        ego_center = surface.pos2pix(*self.env.vehicle.position)
        green = (65, 245, 125)
        radius = max(1, surface.pix(task.goal_config["radius"]))
        pygame.draw.line(surface, (55, 160, 105), ego_center, center, 2)
        # This ring is the actual success tolerance, not a decorative radius.
        pygame.draw.circle(surface, green, center, radius, 3)
        pygame.draw.circle(surface, green, center, 4)
        for dx, dy in ((1, 0), (0, 1)):
            pygame.draw.line(surface, green,
                             (center[0] - 18 * dx, center[1] - 18 * dy),
                             (center[0] + 18 * dx, center[1] + 18 * dy), 2)
        text = self.title_font.render("GOAL", True, green)
        surface.blit(text, (center[0] + radius + 6, center[1] - radius - 22))

    def _draw_hud(self):
        task, surface = self.task, self.sim_surface
        pygame.draw.rect(surface, (20, 27, 40), (0, 0, surface.get_width(), 85))
        color = (65, 245, 125) if task.status == "GOAL REACHED" else (255, 210, 110)
        if task.status == "COLLISION":
            color = (255, 90, 90)
        x, y = task.goal_position
        lines = [
            (f"{task.status}    Distance: {task.distance_to_goal():.1f} m    "
             f"Time: {self.env.time:.1f}/{task.env_config['duration']:.0f} s", color),
            (f"Goal: ({x:.1f}, {y:.1f}) m    Lane ID: {task.goal_lane}    "
             f"Tolerance: {task.goal_config['radius']:.1f} m    Ego: BLUE / Traffic: AMBER", (220, 230, 245)),
            ("Reward: " + "  ".join(f"{name}={value:+.3f}" for name, value in task.last_components.items()), (180, 200, 220)),
        ]
        for index, (text, text_color) in enumerate(lines):
            surface.blit(self.font.render(text, True, text_color), (15, 7 + index * 25))
