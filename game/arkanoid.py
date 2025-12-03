import pygame
import numpy as np
from game.settings import *

class Paddle(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([PADDLE_WIDTH, PADDLE_HEIGHT])
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.x = (SCREEN_WIDTH - PADDLE_WIDTH) // 2
        self.rect.y = SCREEN_HEIGHT - PADDLE_Y_OFFSET
        self.speed = PADDLE_SPEED

    def move_left(self):
        self.rect.x -= self.speed
        if self.rect.x < 0:
            self.rect.x = 0

    def move_right(self):
        self.rect.x += self.speed
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

    def update(self, action):
        # Action: 0 = Stay, 1 = Left, 2 = Right
        if action == 1:
            self.move_left()
        elif action == 2:
            self.move_right()

class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([BALL_RADIUS * 2, BALL_RADIUS * 2])
        self.image.fill(WHITE) # Fill with white for background transparency if needed, but we draw a circle
        self.image.set_colorkey(WHITE) # Make white transparent
        pygame.draw.circle(self.image, RED, (BALL_RADIUS, BALL_RADIUS), BALL_RADIUS)
        self.rect = self.image.get_rect()
        self.reset()

    def reset(self):
        self.rect.x = SCREEN_WIDTH // 2
        self.rect.y = SCREEN_HEIGHT // 2
        self.speed_x = BALL_SPEED * np.random.uniform(-3, 3)
        self.speed_y = -BALL_SPEED

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        # Wall collisions
        if self.rect.left <= 0:
            self.rect.left = 0
            self.speed_x *= -1
        if self.rect.right >= SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
            self.speed_x *= -1
        if self.rect.top <= 0:
            self.rect.top = 0
            self.speed_y *= -1

class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.image = pygame.Surface([SCREEN_WIDTH // BRICK_COLS - BRICK_PADDING, BRICK_HEIGHT])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class ArkanoidGame:
    def __init__(self, render_mode='human'):
        pygame.init()
        self.render_mode = render_mode
        if self.render_mode == 'human':
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Arkanoid RL")
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.all_sprites = pygame.sprite.Group()
        self.bricks = pygame.sprite.Group()
        
        self.paddle = Paddle()
        self.all_sprites.add(self.paddle)
        
        self.ball = Ball()
        self.all_sprites.add(self.ball)
        
        self._create_bricks()
        
        self.score = 0
        self.lives = 1
        self.game_over = False
        self.won = False
        
        return self.get_state()

    def _create_bricks(self):
        colors = [RED, ORANGE, YELLOW, GREEN, BLUE]
        brick_width = SCREEN_WIDTH // BRICK_COLS
        
        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                color = colors[row % len(colors)]
                brick = Brick(col * brick_width + BRICK_PADDING // 2, 
                              row * (BRICK_HEIGHT + BRICK_PADDING) + BRICK_OFFSET_TOP, 
                              color)
                self.bricks.add(brick)
                self.all_sprites.add(brick)

    def step(self, action):
        # Action: 0 = Stay, 1 = Left, 2 = Right
        self.paddle.update(action)
        self.ball.update()
        
        reward = 0
        terminated = False
        
        # Ball collision with paddle
        if pygame.sprite.collide_rect(self.ball, self.paddle):
            self.ball.speed_y *= -1
            self.ball.rect.bottom = self.paddle.rect.top
            reward += 0.2 # Reward for hitting the ball
            
            # Add paddle velocity effect
            # If moving left, add leftward velocity. If right, add rightward.
            if action == 1: # Left
                self.ball.speed_x -= 2
            elif action == 2: # Right
                self.ball.speed_x += 2
                
            # Clamp horizontal speed to avoid it getting too fast
            self.ball.speed_x = max(-10, min(10, self.ball.speed_x))
            
        # Ball collision with bricks
        hits = pygame.sprite.spritecollide(self.ball, self.bricks, True)
        if hits:
            self.ball.speed_y *= -1
            self.score += len(hits) * 10
            reward += len(hits) * 0.5 # Increased reward for bricks
            if len(self.bricks) == 0:
                self.won = True
                terminated = True
                reward += reward * 0.1 + 10.0 / len(hits) # Bonus for winning

        # Ball lost
        if self.ball.rect.top > SCREEN_HEIGHT:
            # Calculate distance bonus (reward for being close to the ball even if missed)
            paddle_center = self.paddle.rect.centerx
            ball_center = self.ball.rect.centerx
            dist = abs(paddle_center - ball_center)
            # Normalize distance to 0-1 range (1 is closest)
            # Max possible distance is approx SCREEN_WIDTH
            distance_reward = (1 - (dist / SCREEN_WIDTH))
            reward += 5 * distance_reward # Add bonus (0.0 to 1.0)
            
            self.lives -= 1
            reward -= 1 # Penalty for losing a life
            if self.lives > 0:
                self.ball.reset()
            else:
                self.game_over = True
                terminated = True
                reward -= 10 # Penalty for game over

        self.render()
        
        return self.get_state(), reward, terminated, False, {"score": self.score, "lives": self.lives}

    def render(self):
        self.screen.fill(BLACK)
        self.all_sprites.draw(self.screen)
        
        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(FPS)

    def get_state(self):
        # Get pixel array
        # pygame.surfarray.array3d returns (W, H, 3), we might want (H, W, 3)
        # Also it is transposed, so we need to transpose it back
        view = pygame.surfarray.array3d(self.screen)
        view = view.transpose([1, 0, 2])
        return view

    def close(self):
        pygame.quit()
