import numpy as np
import random
import pygame
import sys
import matplotlib.pyplot as plt


class TrafficEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.state = np.random.randint(0, 10, size=4)
        return tuple(int(x) for x in self.state)
    
    def step(self, action):
        if action == 0:  # NS GREEN
            self.state[0] = max(0, self.state[0] - 3)
            self.state[1] = max(0, self.state[1] - 3)
        else:  # EW GREEN
            self.state[2] = max(0, self.state[2] - 3)
            self.state[3] = max(0, self.state[3] - 3)
        
        arrivals = np.random.randint(0, 3, size=4)
        self.state += arrivals
        
        reward = -np.sum(self.state)
        return tuple(int(x) for x in self.state), reward


q_table = {}

def get_q(state):
    state = tuple(int(x) for x in state)
    if state not in q_table:
        q_table[state] = np.zeros(2)
    return q_table[state]


env = TrafficEnv()

alpha, gamma = 0.1, 0.9
epsilon = 1.0

for ep in range(500):
    state = env.reset()
    for _ in range(50):
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(get_q(state))
        
        next_state, reward = env.step(action)
        
        get_q(state)[action] += alpha * (
            reward + gamma * np.max(get_q(next_state)) - get_q(state)[action]
        )
        
        state = next_state
    
    epsilon *= 0.995

print("✅ Training Completed")


pygame.init()
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Traffic Signal Game 🚦")

font = pygame.font.SysFont(None, 30)
clock = pygame.time.Clock()

state = env.reset()
mode = "AI"
action = 0


player_score = 0
ai_score = 0


player_history = []
ai_history = []
steps_history = []
step_count = 0


running = True

while running:
    screen.fill((30, 30, 30))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            
            if event.key == pygame.K_a:
                mode = "AI"
            
            if event.key == pygame.K_m:
                mode = "MANUAL"
            
            if mode == "MANUAL":
                if event.key == pygame.K_UP:
                    action = 0
                if event.key == pygame.K_RIGHT:
                    action = 1
    

    if mode == "AI":
        action = np.argmax(get_q(state))
    
    # Environment step
    state, reward = env.step(action)
    
    step_count += 1
    
   
    if mode == "MANUAL":
        player_score += reward
    else:
        ai_score += reward
    
   
    player_history.append(player_score)
    ai_history.append(ai_score)
    steps_history.append(step_count)
    
 
    pygame.draw.rect(screen, (100, 100, 100), (250, 0, 100, 600))
    pygame.draw.rect(screen, (100, 100, 100), (0, 250, 600, 100))
    
 
    if action == 0:
        ns_color = (0, 255, 0)
        ew_color = (255, 0, 0)
    else:
        ns_color = (255, 0, 0)
        ew_color = (0, 255, 0)
    
 
    pygame.draw.circle(screen, ns_color, (300, 200), 20)
    pygame.draw.circle(screen, ns_color, (300, 400), 20)
    pygame.draw.circle(screen, ew_color, (200, 300), 20)
    pygame.draw.circle(screen, ew_color, (400, 300), 20)
   
    N, S, E, W = state
    
    for i in range(N):
        pygame.draw.rect(screen, (0, 0, 255), (290, 50 + i*15, 20, 10))
    for i in range(S):
        pygame.draw.rect(screen, (0, 0, 255), (290, 450 + i*15, 20, 10))
    for i in range(E):
        pygame.draw.rect(screen, (255, 255, 0), (450 + i*15, 290, 10, 20))
    for i in range(W):
        pygame.draw.rect(screen, (255, 255, 0), (50 + i*15, 290, 10, 20))
    
  
    screen.blit(font.render(f"Mode: {mode}", True, (255,255,255)), (10, 10))
    screen.blit(font.render(f"N:{N} S:{S} E:{E} W:{W}", True, (255,255,255)), (10, 40))
    
  
    screen.blit(font.render(f"Player Score: {player_score}", True, (0,255,0)), (10, 80))
    screen.blit(font.render(f"AI Score: {ai_score}", True, (255,0,0)), (10, 110))
    
    pygame.display.update()
    clock.tick(2)

pygame.quit()


plt.plot(steps_history, player_history, label="Player Score")
plt.plot(steps_history, ai_history, label="AI Score")

plt.xlabel("Steps")
plt.ylabel("Score")
plt.title("Traffic Control Performance")

plt.legend()
plt.show()
