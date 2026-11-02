"""GoalViewer with the moving slot box, the anchor car and a hold-progress bar."""
import numpy as np
import pygame

from baseline_waypoint.rendering import GoalViewer

GREEN = (65, 245, 125)
MAGENTA = (240, 90, 230)


class DynamicGoalViewer(GoalViewer):
    def _world_polygon(self, lane, s_range, lateral_range):
        surface = self.sim_surface
        corners = [lane.position(s, lateral) for s, lateral in
                   ((s_range[0], lateral_range[0]), (s_range[1], lateral_range[0]),
                    (s_range[1], lateral_range[1]), (s_range[0], lateral_range[1]))]
        return [surface.pos2pix(*corner) for corner in corners]

    def _draw_goal(self):
        surface, task = self.sim_surface, self.task
        anchor = task.anchor
        # Anchor outline: the car whose slot is the target.
        direction = np.array([np.cos(anchor.heading), np.sin(anchor.heading)])
        normal = np.array([-direction[1], direction[0]])
        half_length, half_width = anchor.LENGTH / 2 + 0.6, anchor.WIDTH / 2 + 0.6
        outline = [surface.pos2pix(*(anchor.position + a * half_length * direction + b * half_width * normal))
                   for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        pygame.draw.polygon(surface, MAGENTA, outline, 3)
        # Slot box: the actual success tolerance, in the slot lane's frame.
        lane = task.highway.road.network.get_lane(task.slot_lane_index)
        slot_s, _ = lane.local_coordinates(task.goal_position)
        radius, lateral = task.goal_config["radius"], task.goal_config["lateral_tolerance"]
        box = self._world_polygon(lane, (slot_s - radius, slot_s + radius), (-lateral, lateral))
        pygame.draw.polygon(surface, GREEN, box, 4 if task.status == "HOLDING" else 2)
        center = surface.pos2pix(*task.goal_position)
        ego_center = surface.pos2pix(*task.highway.vehicle.position)
        pygame.draw.line(surface, (55, 160, 105), ego_center, center, 2)
        pygame.draw.circle(surface, GREEN, center, 4)
        text = self.title_font.render(task.slot.upper(), True, GREEN)
        top = min(point[1] for point in box)
        surface.blit(text, (center[0] - text.get_width() // 2, top - 26))

    def _draw_hud(self):
        task, surface = self.task, self.sim_surface
        pygame.draw.rect(surface, (20, 27, 40), (0, 0, surface.get_width(), 85))
        color = {"GOAL REACHED": GREEN, "COLLISION": (255, 90, 90), "HOLDING": GREEN}.get(task.status, (255, 210, 110))
        relative_speed = task.anchor.speed - task.highway.vehicle.speed
        lines = [
            (f"{task.status}    Distance: {task.distance_to_goal():.1f} m    "
             f"Time: {self.env.time:.1f}/{task.env_config['duration']:.0f} s", color),
            (f"Target: {task.slot.upper()} of car #{task.anchor_id} (magenta)    Slot lane ID: {task.goal_lane}    "
             f"Anchor speed - ego speed: {relative_speed:+.1f} m/s", (220, 230, 245)),
            ("Reward: " + "  ".join(f"{name}={value:+.3f}" for name, value in task.last_components.items()), (180, 200, 220)),
        ]
        for index, (text, text_color) in enumerate(lines):
            surface.blit(self.font.render(text, True, text_color), (15, 7 + index * 25))
        # Hold progress: fills while ego stays in the box, empties when it leaves.
        width, x, y = 220, surface.get_width() - 245, 12
        hold_seconds = task.hold_steps / task.env_config["policy_frequency"]
        fraction = min(1.0, task.hold_steps / task.hold_required)
        pygame.draw.rect(surface, (60, 70, 90), (x, y + 22, width, 14))
        pygame.draw.rect(surface, GREEN, (x, y + 22, int(width * fraction), 14))
        label = f"Hold {hold_seconds:.1f}/{task.goal_config['hold_seconds']:.1f} s"
        surface.blit(self.font.render(label, True, (220, 230, 245)), (x, y))
