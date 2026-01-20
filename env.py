import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math


class SimpleArtilleryEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # ---------- Action & Observation ----------
        self.action_space = spaces.Box(
            low=np.array([-10.0, 40.0], dtype=np.float32),
            high=np.array([10.0, 140.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -40.0], dtype=np.float32),
            high=np.array([90.0, 900.0, 700.0, 40.0], dtype=np.float32),
            dtype=np.float32
        )

        # ---------- Physics ----------
        self.default_v = 80.0
        self.g = 9.81
        self.ground_y = 600.0
        self.max_steps = 20

        # ---------- Barrel animation ----------
        self.barrel_angle = 45.0
        self.target_barrel_angle = 45.0
        self.barrel_rotate_speed = 2.0  # degrees per frame
        self.barrel_length = 60

        # ---------- Trajectory history for dotted line ----------
        self.trajectory_history = []
        self.max_trajectory_points = 100

        # ---------- Pygame ----------
        pygame.init()
        pygame.mixer.init()

        self.screen = pygame.display.set_mode((900, 700))
        pygame.display.set_caption("Artillery RL - Enhanced")
        self.clock = pygame.time.Clock()

        # Load or create sounds
        self.shot_sound = self.create_shot_sound()
        self.blast_sound = self.create_blast_sound()

        self.reset()

    # =====================================================
    def create_shot_sound(self):
        """Create a cannon shot sound effect"""
        try:
            # Try to load from file first
            return pygame.mixer.Sound("assets/sounds/cannon.wav")
        except:
            # Generate a simple cannon sound
            sample_rate = 22050
            duration = 0.3
            frequency = 100
            
            samples = int(sample_rate * duration)
            wave = np.zeros(samples)
            
            # Create explosion-like sound
            for i in range(samples):
                t = i / sample_rate
                # Decaying sine wave with noise
                envelope = np.exp(-8 * t)
                wave[i] = envelope * (
                    np.sin(2 * np.pi * frequency * t) * 0.5 +
                    np.random.uniform(-0.3, 0.3)
                )
            
            # Convert to 16-bit
            wave = np.int16(wave * 32767)
            sound = pygame.sndarray.make_sound(wave)
            return sound

    def create_blast_sound(self):
        """Create an explosion sound effect"""
        try:
            return pygame.mixer.Sound("assets/sounds/blast.wav")
        except:
            # Generate a simple explosion sound
            sample_rate = 22050
            duration = 0.4
            
            samples = int(sample_rate * duration)
            wave = np.zeros(samples)
            
            # Create explosion with white noise burst
            for i in range(samples):
                t = i / sample_rate
                envelope = np.exp(-10 * t)
                wave[i] = envelope * np.random.uniform(-1.0, 1.0)
            
            wave = np.int16(wave * 32767)
            sound = pygame.sndarray.make_sound(wave)
            return sound

    # =====================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.gun_angle = 45.0
        self.barrel_angle = 45.0
        self.target_barrel_angle = 45.0

        self.shelter_x = float(self.np_random.integers(450, 750))
        self.shelter_y = self.ground_y
        self.wind = float(self.np_random.uniform(-20.0, 20.0))

        self.trajectory = []
        self.trajectory_history = []
        self.shell_pos = None
        self.last_hit = False
        self.current_step = 0

        obs = np.array(
            [self.gun_angle, self.shelter_x, self.shelter_y, self.wind],
            dtype=np.float32
        )
        return obs, {}

    # =====================================================
    def step(self, action):
        self.current_step += 1

        angle_change = float(action[0])
        muzzle_v = float(action[1])
##############
        # for _ in range(4):              # 4 small animation frames
        #     self.gun_angle = self.target_barrel_angle
        #     pygame.time.delay(20)         # 20 ms per frame (~80 ms total)
################
        # Update target angle (animation target)
        self.target_barrel_angle = np.clip(self.target_barrel_angle + angle_change, 0.0, 90.0)
        self.gun_angle = self.target_barrel_angle

        # Calculate trajectory
        angle_rad = math.radians(self.gun_angle)
        vx = muzzle_v * math.cos(angle_rad) + self.wind
        vy = muzzle_v * math.sin(angle_rad)

        t_flight = max(0.1, 2 * vy / self.g)
        dt = 0.03
        times = np.arange(0, t_flight, dt)

        self.trajectory = []
        hit = False
        hit_radius = 30.0

        # Play shot sound
        try:
            self.shot_sound.play()
        except:
            pass  # Ignore sound errors

        # Compute trajectory
        for t in times:
            x = 100 + vx * t
            y = self.ground_y - (vy * t - 0.5 * self.g * t * t)

            self.trajectory.append((x, y))
            self.shell_pos = (x, y)

            # Check for hit
            if math.hypot(x - self.shelter_x, y - self.shelter_y) < hit_radius:
                hit = True
                try:
                    self.blast_sound.play()
                except:
                    pass  # Ignore sound errors
                break

        # Add trajectory to history for dotted path
        if len(self.trajectory) > 0:
            self.trajectory_history.extend(self.trajectory[::3])  # Every 3rd point
            if len(self.trajectory_history) > self.max_trajectory_points:
                self.trajectory_history = self.trajectory_history[-self.max_trajectory_points:]

        self.last_hit = hit

        terminated = hit
        truncated = self.current_step >= self.max_steps

        # Calculate reward
        if hit:
            reward = 500.0
        else:
            if self.shell_pos:
                dist = math.hypot(
                    self.shell_pos[0] - self.shelter_x,
                    self.shell_pos[1] - self.shelter_y
                )
                reward = -dist
            else:
                reward = -1000.0

        obs = np.array(
            [self.gun_angle, self.shelter_x, self.shelter_y, self.wind],
            dtype=np.float32
        )

        # Always include trajectory and hit in info
        info = {
            "trajectory": self.trajectory.copy(),  # Make a copy to avoid reference issues
            "hit": hit,
            "wind": self.wind
        }

        return obs, reward, terminated, truncated, info

    # =====================================================
    def update_barrel_animation(self):
        """Smoothly animate barrel rotation"""
        if abs(self.barrel_angle - self.target_barrel_angle) > self.barrel_rotate_speed:
            direction = np.sign(self.target_barrel_angle - self.barrel_angle)
            self.barrel_angle += direction * self.barrel_rotate_speed
        else:
            self.barrel_angle = self.target_barrel_angle

    # =====================================================
    def draw_barrel(self, screen):
        """Draw artillery barrel with pixel art style"""
        base_x = 100
        base_y = int(self.ground_y)
        
        # Draw base/platform
        pygame.draw.circle(screen, (60, 60, 60), (base_x, base_y), 25)
        pygame.draw.circle(screen, (80, 80, 80), (base_x, base_y), 20)
        
        # Draw wheels
        wheel_offset = 15
        pygame.draw.circle(screen, (40, 40, 40), (base_x - wheel_offset, base_y + 10), 8)
        pygame.draw.circle(screen, (40, 40, 40), (base_x + wheel_offset, base_y + 10), 8)
        
        # Draw barrel
        angle_rad = math.radians(self.barrel_angle)
        end_x = base_x + self.barrel_length * math.cos(angle_rad)
        end_y = base_y - self.barrel_length * math.sin(angle_rad)
        
        # Barrel shadow/outline
        pygame.draw.line(screen, (20, 20, 20), (base_x, base_y), (end_x, end_y), 10)
        # Main barrel
        pygame.draw.line(screen, (100, 100, 100), (base_x, base_y), (end_x, end_y), 8)
        # Barrel highlight
        h_end_x = base_x + (self.barrel_length - 5) * math.cos(angle_rad)
        h_end_y = base_y - (self.barrel_length - 5) * math.sin(angle_rad)
        pygame.draw.line(screen, (140, 140, 140), (base_x, base_y), (h_end_x, h_end_y), 4)

    # =====================================================
    def draw_trajectory_path(self, screen):
        """Draw dotted trajectory path"""
        if len(self.trajectory_history) > 1:
            for i in range(0, len(self.trajectory_history) - 1, 4):  # Every 4th point for dots
                pos = self.trajectory_history[i]
                if 0 <= pos[0] < 900 and 0 <= pos[1] < 700:
                    pygame.draw.circle(screen, (100, 100, 100), 
                                     (int(pos[0]), int(pos[1])), 2)

    # =====================================================
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.update_barrel_animation()

        # Background - Sky gradient
        for y in range(int(self.ground_y)):
            color_val = 135 + int((206 - 135) * (y / self.ground_y))
            pygame.draw.line(self.screen, (color_val, color_val + 50, 235), 
                           (0, y), (900, y))
        
        # Ground
        pygame.draw.rect(
            self.screen, (34, 139, 34),  # Forest green
            (0, self.ground_y, 900, 100)
        )
        
        # Ground details
        for i in range(0, 900, 40):
            pygame.draw.line(self.screen, (20, 100, 20), 
                           (i, self.ground_y), (i + 20, self.ground_y), 2)

        # Draw trajectory path (behind everything)
        self.draw_trajectory_path(self.screen)

        # Shelter/Target
        shelter_width = 50
        shelter_height = 50
        pygame.draw.rect(
            self.screen, (139, 69, 19),  # Brown
            (self.shelter_x - shelter_width/2, self.shelter_y - shelter_height, 
             shelter_width, shelter_height)
        )
        # Shelter roof
        roof_points = [
            (self.shelter_x - shelter_width/2 - 5, self.shelter_y - shelter_height),
            (self.shelter_x, self.shelter_y - shelter_height - 15),
            (self.shelter_x + shelter_width/2 + 5, self.shelter_y - shelter_height)
        ]
        pygame.draw.polygon(self.screen, (120, 50, 10), roof_points)
        
        # Door
        pygame.draw.rect(self.screen, (80, 40, 10),
                        (self.shelter_x - 10, self.shelter_y - 30, 20, 30))

        # Draw artillery
        self.draw_barrel(self.screen)

        # Shell with trail effect
        if self.shell_pos:
            sx, sy = int(self.shell_pos[0]), int(self.shell_pos[1])
            if 0 <= sx < 900 and 0 <= sy < 700:
                # Shell glow
                pygame.draw.circle(self.screen, (255, 150, 0), (sx, sy), 8)
                pygame.draw.circle(self.screen, (255, 200, 100), (sx, sy), 5)
                pygame.draw.circle(self.screen, (255, 255, 200), (sx, sy), 3)

        # HUD
        font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Wind indicator
        wind_text = font.render(f"Wind: {self.wind:+.1f}", True, (255, 255, 255))
        wind_bg = pygame.Surface((150, 30))
        wind_bg.fill((0, 0, 0))
        wind_bg.set_alpha(150)
        self.screen.blit(wind_bg, (10, 10))
        self.screen.blit(wind_text, (15, 12))
        
        # # Angle indicator
        # angle_text = font.render(f"Angle: {self.barrel_angle:.1f}°", True, (255, 255, 255))
        # angle_bg = pygame.Surface((150, 30))
        # angle_bg.fill((0, 0, 0))
        # angle_bg.set_alpha(150)
        # self.screen.blit(angle_bg, (10, 45))
        # self.screen.blit(angle_text, (15, 47))

        # Hit message
        if self.last_hit:
            hit_font = pygame.font.SysFont("monospace", 72, bold=True)
            hit_text = hit_font.render("HIT!", True, (255, 50, 50))
            hit_outline = hit_font.render("HIT!", True, (255, 255, 255))
            text_rect = hit_text.get_rect(center=(450, 100))
            
            # Draw outline
            for dx, dy in [(-2,-2), (-2,2), (2,-2), (2,2)]:
                outline_rect = text_rect.copy()
                outline_rect.center = (450 + dx, 100 + dy)
                self.screen.blit(hit_outline, outline_rect)
            
            self.screen.blit(hit_text, text_rect)

        pygame.display.flip()
        self.clock.tick(60)

    # =====================================================
    def close(self):
        pygame.quit()