import pygame
import gym
import numpy as np
import cv2  #Tambahkan OpenCV untuk menampilkan animasi
from dqn_agent import DQNAgent

# Cek apakah pygame berhasil diinstal
print("Cek pygame:")
print("SDL Version:", pygame.get_sdl_version())  # Harusnya menampilkan versi SDL
print("-" * 30)

# Buat environment dengan render_mode "rgb_array"
env = gym.make('CartPole-v1', render_mode="rgb_array")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen (gunakan model terlatih jika tersedia)
agent = DQNAgent(state_size, action_size)
agent.epsilon = 0.01  # Minimalkan eksplorasi saat testing

for e in range(100):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    
    for time in range(500):
        frame = env.render()  # Ambil frame dari environment
        cv2.imshow("CartPole", frame)  # Tampilkan menggunakan OpenCV
        cv2.waitKey(1)  #  Delay kecil agar animasi berjalan lancar

        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  
        state = np.reshape(next_state, [1, state_size])

        if done:  # Pindahkan kondisi ini ke dalam loop agar episodenya benar
            print(f"Test Episode: {e+1}, Score: {time}")
            break

env.close()
cv2.destroyAllWindows()  # Tutup jendela OpenCV setelah selesai