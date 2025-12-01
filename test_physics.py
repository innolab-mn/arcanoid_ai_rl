from game.arkanoid import ArkanoidGame
from game.settings import SCREEN_HEIGHT, SCREEN_WIDTH

def test_physics():
    print("Testing Paddle Physics...")
    game = ArkanoidGame(render_mode=None)
    
    # --- Case 1: Paddle moves RIGHT ---
    game.reset()
    game.paddle.rect.centerx = 300
    game.ball.rect.centerx = 300
    game.ball.rect.bottom = game.paddle.rect.top + 1 # Force collision
    game.ball.speed_y = 5
    game.ball.speed_x = 0 # Start with 0 horizontal speed
    
    # Step with action RIGHT (2)
    # We need to manually trigger collision logic which happens in step()
    # But step() updates positions first.
    # So let's place ball slightly above paddle and moving down.
    game.ball.rect.bottom = game.paddle.rect.top - 1
    
    print(f"Initial Speed X: {game.ball.speed_x}")
    game.step(2) # Right
    print(f"After Hit (Right): Speed X: {game.ball.speed_x}")
    
    if game.ball.speed_x > 0:
        print("PASS: Ball moved right.")
    else:
        print("FAIL: Ball did not move right.")
        
    # --- Case 2: Paddle moves LEFT ---
    game.reset()
    game.paddle.rect.centerx = 300
    game.ball.rect.centerx = 300
    game.ball.rect.bottom = game.paddle.rect.top - 1
    game.ball.speed_y = 5
    game.ball.speed_x = 0
    
    print(f"Initial Speed X: {game.ball.speed_x}")
    game.step(1) # Left
    print(f"After Hit (Left): Speed X: {game.ball.speed_x}")
    
    if game.ball.speed_x < 0:
        print("PASS: Ball moved left.")
    else:
        print("FAIL: Ball did not move left.")

if __name__ == "__main__":
    test_physics()
