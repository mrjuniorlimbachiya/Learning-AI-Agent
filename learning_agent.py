import tkinter as tk
import random
import pickle
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CELL_SIZE = 25
GRID_SIZE = 20
WIDTH = HEIGHT = CELL_SIZE * GRID_SIZE
QTABLE_FILE = "q_table.pkl"

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# ---------- Q LEARNING AGENT ----------
class QLearningAgent:
    def __init__(self):
        self.q = {}
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def get_state(self, snake, food, direction):
        head = snake[0]
        return (
            food[0] - head[0],
            food[1] - head[1],
            direction
        )

    def choose_action(self, state):
        self.q.setdefault(state, {a: 0 for a in ACTIONS})

        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        return max(self.q[state], key=self.q[state].get)

    def learn(self, state, action, reward, next_state):
        self.q.setdefault(next_state, {a: 0 for a in ACTIONS})

        old = self.q[state][action]
        future = max(self.q[next_state].values())

        self.q[state][action] = old + self.lr * (reward + self.gamma * future - old)

    def save(self):
        with open(QTABLE_FILE, "wb") as f:
            pickle.dump(self.q, f)
        print("✅ Q-table saved")

    def load(self):
        try:
            with open(QTABLE_FILE, "rb") as f:
                self.q = pickle.load(f)
            print("✅ Q-table loaded")
        except FileNotFoundError:
            print("ℹ️ No saved Q-table found")

# ---------- GAME ----------
class SnakeGame:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="#111")
        self.canvas.pack()

        self.agent = QLearningAgent()
        self.agent.load()

        self.rewards = []
        self.episode_reward = 0
        self.episode = 1

        self.reset()
        root.after(80, self.loop)

    def reset(self):
        self.snake = [(10, 10)]
        self.food = self.spawn_food()
        self.direction = "RIGHT"
        self.episode_reward = 0

    def spawn_food(self):
        while True:
            pos = (random.randint(0, 19), random.randint(0, 19))
            if pos not in self.snake:
                return pos

    def move(self, action):
        x, y = self.snake[0]

        if action == "UP": y -= 1
        if action == "DOWN": y += 1
        if action == "LEFT": x -= 1
        if action == "RIGHT": x += 1

        new_head = (x, y)
        reward = -1

        if new_head in self.snake or not (0 <= x < 20 and 0 <= y < 20):
            reward = -100
            self.rewards.append(self.episode_reward)
            self.episode += 1
            self.reset()
            return reward

        self.snake.insert(0, new_head)

        if new_head == self.food:
            reward = 20
            self.food = self.spawn_food()
        else:
            self.snake.pop()

        return reward

    def draw_grid(self):
        for i in range(GRID_SIZE):
            self.canvas.create_line(
                i * CELL_SIZE, 0, i * CELL_SIZE, HEIGHT, fill="#222"
            )
            self.canvas.create_line(
                0, i * CELL_SIZE, WIDTH, i * CELL_SIZE, fill="#222"
            )

    def draw(self):
        self.canvas.delete("all")
        self.draw_grid()

        # Snake body
        for i, (x, y) in enumerate(self.snake):
            color = "#00ff66" if i == 0 else "#00cc55"
            self.canvas.create_oval(
                x * CELL_SIZE + 3,
                y * CELL_SIZE + 3,
                (x + 1) * CELL_SIZE - 3,
                (y + 1) * CELL_SIZE - 3,
                fill=color,
                outline=""
            )

        # Food
        fx, fy = self.food
        self.canvas.create_oval(
            fx * CELL_SIZE + 5,
            fy * CELL_SIZE + 5,
            (fx + 1) * CELL_SIZE - 5,
            (fy + 1) * CELL_SIZE - 5,
            fill="red"
        )

        # HUD
        self.canvas.create_text(
            10, 10,
            anchor="nw",
            fill="white",
            text=f"Episode: {self.episode}"
        )

    def loop(self):
        state = self.agent.get_state(self.snake, self.food, self.direction)
        action = self.agent.choose_action(state)

        reward = self.move(action)
        self.episode_reward += reward

        next_state = self.agent.get_state(self.snake, self.food, action)
        self.agent.learn(state, action, reward, next_state)

        self.direction = action
        self.draw()
        self.root.after(80, self.loop)

# ---------- PLOT ----------
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Snake Learning Progress")
    plt.show()

# ---------- RUN ----------
root = tk.Tk()
root.title("Snake Learning Agent (Python 3.14)")
game = SnakeGame(root)

def on_close():
    game.agent.save()
    plot_rewards(game.rewards)
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
