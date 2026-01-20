import pygame
import time
import numpy as np
from env import SimpleArtilleryEnv

SCREEN_W, SCREEN_H = 900, 700
GROUND_Y = 600


def draw_scene(screen, env, shell_pos=None, trajectory=None, hit=False):
    """Enhanced scene drawing with trajectory dots"""
    # Sky background with gradient effect
    for y in range(GROUND_Y):
        color_val = 135 + int((206 - 135) * (y / GROUND_Y))
        pygame.draw.line(screen, (color_val, color_val + 50, 235), (0, y), (SCREEN_W, y))
    
    # Ground
    pygame.draw.rect(screen, (34, 139, 34), (0, GROUND_Y, SCREEN_W, SCREEN_H - GROUND_Y))
    
    # Ground texture
    for i in range(0, SCREEN_W, 40):
        pygame.draw.line(screen, (20, 100, 20), (i, GROUND_Y), (i + 20, GROUND_Y), 2)
    
    # Draw dotted trajectory path
    if trajectory and len(trajectory) > 1:
        for i in range(0, len(trajectory), 5):  # Every 5th point
            x, y = trajectory[i]
            if 0 <= x < SCREEN_W and 0 <= y < SCREEN_H:
                # Fade effect based on position in trajectory
                alpha = 100 + int(155 * (i / len(trajectory)))
                pygame.draw.circle(screen, (150, 150, 150), (int(x), int(y)), 3)
                pygame.draw.circle(screen, (200, 200, 200), (int(x), int(y)), 2)
    
    # Target/Shelter
    shelter_w, shelter_h = 50, 50
    shelter_x, shelter_y = env.shelter_x, env.shelter_y
    
    # Shelter body
    pygame.draw.rect(screen, (139, 69, 19), 
                     (shelter_x - shelter_w/2, shelter_y - shelter_h, 
                      shelter_w, shelter_h))
    
    # Shelter roof
    roof_points = [
        (shelter_x - shelter_w/2 - 5, shelter_y - shelter_h),
        (shelter_x, shelter_y - shelter_h - 15),
        (shelter_x + shelter_w/2 + 5, shelter_y - shelter_h)
    ]
    pygame.draw.polygon(screen, (120, 50, 10), roof_points)
    
    # Door
    pygame.draw.rect(screen, (80, 40, 10), 
                     (shelter_x - 10, shelter_y - 30, 20, 30))
    
    # Artillery base
    base_x, base_y = 100, GROUND_Y
    
    # Platform
    pygame.draw.circle(screen, (60, 60, 60), (base_x, base_y), 25)
    pygame.draw.circle(screen, (80, 80, 80), (base_x, base_y), 20)
    
    # Wheels
    pygame.draw.circle(screen, (40, 40, 40), (base_x - 15, base_y + 10), 8)
    pygame.draw.circle(screen, (40, 40, 40), (base_x + 15, base_y + 10), 8)
    
    # Barrel
    barrel_len = 60
    angle_rad = -env.gun_angle * np.pi / 180.0
    end_x = base_x + barrel_len * np.cos(angle_rad)
    end_y = base_y + barrel_len * np.sin(angle_rad)
    
    # Barrel shadow
    pygame.draw.line(screen, (20, 20, 20), (base_x, base_y), (end_x, end_y), 10)
    # Main barrel
    pygame.draw.line(screen, (100, 100, 100), (base_x, base_y), (end_x, end_y), 8)
    # Barrel highlight
    h_end_x = base_x + (barrel_len - 5) * np.cos(angle_rad)
    h_end_y = base_y + (barrel_len - 5) * np.sin(angle_rad)
    pygame.draw.line(screen, (140, 140, 140), (base_x, base_y), (h_end_x, h_end_y), 4)
    
    # Shell with glow effect
    if shell_pos is not None:
        sx, sy = int(shell_pos[0]), int(shell_pos[1])
        if 0 <= sx < SCREEN_W and 0 <= sy < SCREEN_H:
            # Outer glow
            pygame.draw.circle(screen, (255, 150, 0), (sx, sy), 10)
            # Middle layer
            pygame.draw.circle(screen, (255, 200, 100), (sx, sy), 6)
            # Core
            pygame.draw.circle(screen, (255, 255, 200), (sx, sy), 3)
    
    # HUD
    font = pygame.font.SysFont("monospace", 20, bold=True)
    
    # Wind indicator with background
    wind_text = font.render(f"Wind: {env.wind:+.1f}", True, (255, 255, 255))
    wind_bg = pygame.Surface((150, 30))
    wind_bg.fill((0, 0, 0))
    wind_bg.set_alpha(150)
    screen.blit(wind_bg, (10, 10))
    screen.blit(wind_text, (15, 12))
    
    # Angle indicator
    angle_text = font.render(f"Angle: {env.gun_angle:.1f}°", True, (255, 255, 255))
    angle_bg = pygame.Surface((150, 30))
    angle_bg.fill((0, 0, 0))
    angle_bg.set_alpha(150)
    screen.blit(angle_bg, (10, 45))
    screen.blit(angle_text, (15, 47))
    
    # Hit message with outline
    if hit:
        hit_font = pygame.font.SysFont("monospace", 72, bold=True)
        hit_text = hit_font.render("HIT!", True, (255, 50, 50))
        hit_outline = hit_font.render("HIT!", True, (255, 255, 255))
        text_rect = hit_text.get_rect(center=(SCREEN_W // 2, 100))
        
        # Draw outline
        for dx, dy in [(-2,-2), (-2,2), (2,-2), (2,2)]:
            outline_rect = text_rect.copy()
            outline_rect.center = (SCREEN_W // 2 + dx, 100 + dy)
            screen.blit(hit_outline, outline_rect)
        
        screen.blit(hit_text, text_rect)
    
    pygame.display.flip()


def animate_trajectory(screen, env, trajectory, hit):
    """Animate shell movement along trajectory with dotted path"""
    clock = pygame.time.Clock()
    
    if not trajectory:
        return True
    
    # Animate shell movement
    for i, pos in enumerate(trajectory):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        # Draw scene with trajectory up to current point
        draw_scene(screen, env, shell_pos=pos, trajectory=trajectory[:i+1], hit=False)
        clock.tick(40)  # Smooth animation
    
    # Final frame with hit indicator
    draw_scene(screen, env, shell_pos=trajectory[-1], trajectory=trajectory, hit=hit)
    pygame.time.wait(800 if hit else 600)
    
    return True


def interactive_demo():
    """Interactive demo with keyboard controls"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Artillery Demo - Arrow keys: angle/speed, SPACE: fire")
    clock = pygame.time.Clock()
    
    env = SimpleArtilleryEnv()
    obs, _ = env.reset()
    
    running = True
    last_fire_time = 0.0
    fire_cooldown_ms = 250
    current_speed = env.default_v
    
    # Instructions
    inst_font = pygame.font.SysFont("monospace", 16)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Angle control
        angle_change = 0.0
        if keys[pygame.K_LEFT]:
            angle_change = -2.0
        elif keys[pygame.K_RIGHT]:
            angle_change = 2.0
        
        # Speed control
        if keys[pygame.K_UP]:
            current_speed = min(140.0, current_speed + 1.0)
        elif keys[pygame.K_DOWN]:
            current_speed = max(40.0, current_speed - 1.0)
        
        # Fire
        fired = keys[pygame.K_SPACE]
        if fired and (pygame.time.get_ticks() - last_fire_time) > fire_cooldown_ms:
            last_fire_time = pygame.time.get_ticks()
            obs, reward, terminated, truncated, info = env.step([angle_change, current_speed])
            
            ok = animate_trajectory(screen, env, info["trajectory"], info["hit"])
            if not ok:
                running = False
                break
            
            if info["hit"]:
                time.sleep(0.3)
                obs, _ = env.reset()
                current_speed = env.default_v
        else:
            # Just update angle without firing
            if abs(angle_change) > 0.0:
                env.gun_angle = float(np.clip(env.gun_angle + angle_change, 0.0, 90.0))
            
            draw_scene(screen, env, shell_pos=None, trajectory=None, hit=False)
            
            # Draw instructions
            instructions = [
                "LEFT/RIGHT: Adjust angle",
                "UP/DOWN: Adjust speed",
                "SPACE: Fire",
                f"Speed: {current_speed:.0f}"
            ]
            
            y_offset = SCREEN_H - 120
            for inst in instructions:
                inst_bg = pygame.Surface((250, 25))
                inst_bg.fill((0, 0, 0))
                inst_bg.set_alpha(150)
                screen.blit(inst_bg, (SCREEN_W - 260, y_offset))
                
                inst_text = inst_font.render(inst, True, (255, 255, 255))
                screen.blit(inst_text, (SCREEN_W - 255, y_offset + 3))
                y_offset += 30
            
            pygame.display.flip()
            clock.tick(30)
    
    pygame.quit()


def test_random_render(n_episodes=3):
    """Test with random actions"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Artillery Random Test")
    
    for ep in range(n_episodes):
        env = SimpleArtilleryEnv()
        obs, _ = env.reset()
        done = False
        shots = 0
        
        while not done and shots < 20:
            action = [
                float(np.random.uniform(-3.0, 3.0)), 
                float(np.random.uniform(60.0, 120.0))
            ]
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ok = animate_trajectory(screen, env, info["trajectory"], info["hit"])
            if not ok:
                pygame.quit()
                return
            
            shots += 1
            pygame.time.wait(120)
        
        print(f"[Random] Episode {ep+1}: Hit={info['hit']}, Shots={shots}")
        pygame.time.wait(500)
    
    pygame.quit()


if __name__ == "__main__":
    print("\n🎮 Artillery Demo")
    print("=" * 50)
    print("1. Interactive Demo (recommended)")
    print("2. Random Agent Test")
    print("=" * 50)
    
    choice = input("Select option (1 or 2): ").strip()
    
    if choice == "1":
        interactive_demo()
    elif choice == "2":
        test_random_render()
    else:
        print("Running interactive demo by default...")
        interactive_demo()