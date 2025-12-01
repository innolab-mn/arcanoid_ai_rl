import argparse
import pygame
from game.arkanoid import ArkanoidGame

def manual_play():
    game = ArkanoidGame(render_mode='human')
    running = True
    
    while running:
        action = 0 # Stay
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 1
        elif keys[pygame.K_RIGHT]:
            action = 2
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        obs, reward, terminated, truncated, info = game.step(action)
        
        if terminated:
            print(f"Game Over! Score: {info['score']}")
            game.reset()
            
    game.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    manual_play()
