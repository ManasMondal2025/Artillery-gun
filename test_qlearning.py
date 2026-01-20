import pickle
import numpy as np
import pygame
import time
from env import SimpleArtilleryEnv
from main_demo import animate_trajectory

class TestQLearning:
    def __init__(self):
        # Initialize environment
        self.env = SimpleArtilleryEnv()
        np.random.seed(0)
        import random
        random.seed(0)
        self.env.reset(seed=0)

        # Load trained Q-table and actions
        try:
            with open("simple_qtable.pkl", "rb") as f:
                self.q_table = pickle.load(f)
            with open("simple_actions.pkl", "rb") as f:
                self.action_list = pickle.load(f)
            
            # Try to load metadata for verification
            try:
                with open("simple_metadata.pkl", "rb") as f:
                    metadata = pickle.load(f)
                    print("✅ Loaded Q-table metadata")
            except FileNotFoundError:
                print("⚠️  No metadata file found (training with old version)")
                
        except FileNotFoundError:
            print("❌ ERROR: Q-table files not found!")
            print("   Please run: python qlearning.py")
            raise

        # Ensure action list length matches Q-table
        assert len(self.action_list) == self.q_table.shape[3], \
            f"Action list length {len(self.action_list)} must match Q-table actions {self.q_table.shape[3]}"

        # Parameters must match training
        self.n_angle_bins = self.q_table.shape[0]
        self.n_dist_bins = self.q_table.shape[1]
        self.n_wind_bins = self.q_table.shape[2]
        self.angle_space = np.linspace(0.0, 90.0, self.n_angle_bins)
        self.dist_space = np.linspace(0.0, 800.0, self.n_dist_bins)
        self.wind_space = np.arange(-20, 21)  # Must match training exactly

        print(f"\n📊 Q-table loaded:")
        print(f"   State dimensions: {self.n_angle_bins}×{self.n_dist_bins}×{self.n_wind_bins}")
        print(f"   Actions: {len(self.action_list)}")
        print(f"   Total Q-values: {self.q_table.size:,}\n")

        # Initialize Pygame for visualization
        pygame.init()
        self.screen = pygame.display.set_mode((900, 700))
        pygame.display.set_caption("Q-Learning Test Agent")

    def discretize(self, obs):
        """Discretize observation - must match training exactly"""
        gun_angle = float(obs[0])
        shelter_x = float(obs[1])
        wind = float(obs[3]) if len(obs) > 3 else 0.0
        dist = max(0.0, shelter_x - 100.0)

        angle_idx = int(np.digitize(gun_angle, self.angle_space) - 1)
        angle_idx = np.clip(angle_idx, 0, self.n_angle_bins - 1)

        dist_idx = int(np.digitize(dist, self.dist_space) - 1)
        dist_idx = np.clip(dist_idx, 0, self.n_dist_bins - 1)

        wind_idx = int(np.digitize(wind, self.wind_space) - 1)
        wind_idx = np.clip(wind_idx, 0, self.n_wind_bins - 1)

        return angle_idx, dist_idx, wind_idx

    def run(self, n_episodes=10, verbose=True):
        one_shot_hits = 0
        total_hits = 0
        total_steps = 0
        episode_details = []
        
        if verbose:
            print("🎯 Running Q-Learning Test Episodes...\n")
        
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=ep)
            done = False
            steps = 0
            total_reward = 0.0
            episode_trajectory = []

            while not done and steps < 20:
                # Handle Pygame events (so window stays responsive)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        if verbose:
                            print("\n✅ Q-Learning Testing Complete (User closed window)")
                        return

                # Discretize state including wind
                angle_idx, dist_idx, wind_idx = self.discretize(obs)

                # Choose greedy action (exploitation only during testing)
                action_idx = int(np.argmax(self.q_table[angle_idx, dist_idx, wind_idx]))
                action_idx = np.clip(action_idx, 0, len(self.action_list) - 1)
                action = self.action_list[action_idx]

                # Debug first shot
                if steps == 0 and verbose and ep < 3:
                    print(f"Episode {ep+1}, Shot 1:")
                    print(f"  Angle: {obs[0]:.1f}°, Target: {obs[1]:.0f}, Wind: {obs[3]:+.1f}")
                    print(f"  Action: angle_change={action[0]:+.1f}, speed={action[1]:.1f}")

                # Step in environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                steps += 1

                # Animate shell trajectory
                traj = info.get("trajectory", [])
                hit = info.get("hit", False)
                animate_trajectory(self.screen, self.env, traj, hit)
                time.sleep(0.1)  # small delay to see movement

            # Track statistics
            hit = info.get('hit', False)
            if hit:
                total_hits += 1
                if steps == 1:
                    one_shot_hits += 1
            
            total_steps += steps
            episode_details.append({
                'episode': ep + 1,
                'steps': steps,
                'hit': hit,
                'reward': total_reward
            })
            
            if verbose:
                status = "✓ HIT" if hit else "✗ MISS"
                one_shot = " (ONE-SHOT!)" if (hit and steps == 1) else ""
                print(f"Episode {ep+1}: {status}{one_shot}, steps={steps}, reward={total_reward:.1f}")
            
            time.sleep(0.5)  # pause between episodes

        # Print summary
        avg_steps = total_steps / n_episodes
        hit_rate = (total_hits / n_episodes) * 100
        one_shot_rate = (one_shot_hits / n_episodes) * 100
        
        if verbose:
            print("\n" + "="*50)
            print("📊 Q-Learning Test Results Summary")
            print("="*50)
            print(f"Total hits:     {total_hits}/{n_episodes} ({hit_rate:.1f}%)")
            print(f"One-shot hits:  {one_shot_hits}/{n_episodes} ({one_shot_rate:.1f}%)")
            print(f"Average steps:  {avg_steps:.2f}")
            print("="*50)
            
            # Show episodes that didn't hit in one shot
            if one_shot_hits < n_episodes:
                print("\n⚠️  Episodes that needed multiple shots:")
                for detail in episode_details:
                    if detail['steps'] > 1:
                        print(f"   Episode {detail['episode']}: {detail['steps']} steps, hit={detail['hit']}")
            print()
        
        pygame.quit()
        if verbose:
            print("✅ Q-Learning Testing Complete")
        
        return {
            'total_hits': total_hits,
            'one_shot_hits': one_shot_hits,
            'hit_rate': hit_rate,
            'one_shot_rate': one_shot_rate,
            'avg_steps': avg_steps,
            'details': episode_details
        }


def run_multiple_tests(n_runs=5, episodes_per_run=5):
    """Run multiple test sessions to check consistency"""
    print("\n" + "="*60)
    print("🔬 Running Multiple Test Sessions for Consistency Check")
    print("="*60)
    
    all_results = []
    
    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Test Session {run + 1}/{n_runs}")
        print(f"{'='*60}")
        
        tester = TestQLearning()
        results = tester.run(n_episodes=episodes_per_run, verbose=True)
        all_results.append(results)
        
        time.sleep(1)  # Small delay between sessions
    
    # Aggregate statistics
    print("\n" + "="*60)
    print("📈 AGGREGATE STATISTICS ACROSS ALL SESSIONS")
    print("="*60)
    
    total_episodes = n_runs * episodes_per_run
    total_hits = sum(r['total_hits'] for r in all_results)
    total_one_shots = sum(r['one_shot_hits'] for r in all_results)
    
    overall_hit_rate = (total_hits / total_episodes) * 100
    overall_one_shot_rate = (total_one_shots / total_episodes) * 100
    avg_hit_rate = np.mean([r['hit_rate'] for r in all_results])
    avg_one_shot_rate = np.mean([r['one_shot_rate'] for r in all_results])
    
    print(f"Total episodes:         {total_episodes}")
    print(f"Total hits:             {total_hits} ({overall_hit_rate:.1f}%)")
    print(f"Total one-shot hits:    {total_one_shots} ({overall_one_shot_rate:.1f}%)")
    print(f"Average hit rate:       {avg_hit_rate:.1f}% (±{np.std([r['hit_rate'] for r in all_results]):.1f})")
    print(f"Average one-shot rate:  {avg_one_shot_rate:.1f}% (±{np.std([r['one_shot_rate'] for r in all_results]):.1f})")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        # Run consistency check with multiple sessions
        run_multiple_tests(n_runs=5, episodes_per_run=5)
    else:
        # Single test session
        tester = TestQLearning()
        tester.run(n_episodes=5)