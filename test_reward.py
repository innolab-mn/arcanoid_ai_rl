from game.arkanoid import ArkanoidGame
from game.settings import SCREEN_HEIGHT, SCREEN_WIDTH

def test_reward():
    print("Testing Reward Logic...")
    game = ArkanoidGame(render_mode=None)
    
    # --- Case 1: Aligned (High Reward) ---
    game.reset()
    # Force positions
    game.paddle.rect.centerx = 300
    game.ball.rect.centerx = 300
    game.ball.rect.top = SCREEN_HEIGHT - 1 # Just about to fall
    game.ball.speed_y = 10 # Ensure it crosses the line
    
    # Step
    _, reward, terminated, _, _ = game.step(0) # Stay
    
    print(f"Case 1 (Aligned): Reward={reward}, Terminated={terminated}")
    # Expected: -1 (penalty) + 1 (bonus) = 0
    # Note: It might be slightly different due to float precision or if lives > 0
    
    # --- Case 2: Far (Low Reward) ---
    game.reset()
    game.paddle.rect.centerx = 0
    game.ball.rect.centerx = 600 # Max distance
    game.ball.rect.top = SCREEN_HEIGHT - 1
    game.ball.speed_y = 10
    
    _, reward, terminated, _, _ = game.step(0)
    
    print(f"Case 2 (Far): Reward={reward}, Terminated={terminated}")
    # Expected: -1 (penalty) + 0 (bonus) = -1
    
    # --- Case 3: Middle (Medium Reward) ---
    game.reset()
    game.paddle.rect.centerx = 0
    game.ball.rect.centerx = 300 # Half screen
    game.ball.rect.top = SCREEN_HEIGHT - 1
    game.ball.speed_y = 10
    
    _, reward, terminated, _, _ = game.step(0)
    
    print(f"Case 3 (Middle): Reward={reward}, Terminated={terminated}")
    # Expected: -1 (penalty) + 0.5 (bonus) = -0.5

if __name__ == "__main__":
    test_reward()
