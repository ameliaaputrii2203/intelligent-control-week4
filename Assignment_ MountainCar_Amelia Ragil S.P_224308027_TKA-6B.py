import gym
import numpy as np
import random
import tensorflow as tf
import cv2  
from collections import deque

# DQN Agent dengan Target Network
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # HYPERPARAMETERS
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05  # Sedikit lebih tinggi untuk MountainCar
        self.epsilon_decay = 0.999  # Perlambat decay agar eksplorasi lebih lama
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_freq = 10  # Update Target Network setiap 10 episode
        self.memory = deque(maxlen=5000)  # Lebih besar untuk MountainCar

        # Q-Network & Target Network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

    def _build_model(self):
        """Membangun Q-Network"""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self):
        """Copy weights dari Q-Network ke Target Network"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Simpan pengalaman ke memory buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-Greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Latih model menggunakan Experience Replay"""
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward  # Jika selesai, tidak ada future Q
            else:
                future_q = np.max(self.target_model.predict(next_state, verbose=0)[0])
                target[0][action] = reward + self.gamma * future_q

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ENVIRONMENT: MountainCar-v0
env = gym.make("MountainCar-v0", render_mode="rgb_array")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

for e in range(100):  
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(200):  # Max 200 steps di MountainCar
        frame = env.render()
        cv2.imshow("MountainCar", frame)
        cv2.waitKey(1)
        
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])

        # Untuk MountainCar, tambahkan reward lebih besar jika berhasil
        if terminated:
            reward = 100  

        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            if e % agent.update_target_freq == 0:
                agent.update_target_network()
            print(f"Episode: {e+1}, Score: {time}, Epsilon: {agent.epsilon:.4f}")
            break

    agent.replay()

env.close()
cv2.destroyAllWindows()