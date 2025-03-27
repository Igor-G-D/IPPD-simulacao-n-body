from numba import cuda
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import argparse
import math

DT = 0.000001
MIN = 0.0001

time_per_frame = []  # Lista para armazenar o tempo de cada frame

class Body:
    def __init__(self, pos: np.ndarray, vel: np.ndarray, mass: float):
        self.pos = pos  # vetor 2D de posição
        self.vel = vel  # vetor 2D de velocidade
        self.acc = np.zeros(2)  # vetor 2D de aceleração (inicializado com 0)
        self.mass = mass  # "peso" do corpo

def rand_body() -> Body:
    pos = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
    vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
    mass = random.uniform(0.1, 10.0)
    return Body(pos, vel, mass)

@cuda.jit
def compute_forces_kernel(positions, masses, accelerations, n):
    i = cuda.grid(1)
    if i < n:
        acc_x = 0.0
        acc_y = 0.0
        for j in range(n):
            if i != j:
                r_x = positions[j, 0] - positions[i, 0]
                r_y = positions[j, 1] - positions[i, 1]
                mag_sq = r_x * r_x + r_y * r_y

                mag = math.sqrt(mag_sq) 

                tmp_x = r_x / (max(mag_sq, MIN) * mag)
                tmp_y = r_y / (max(mag_sq, MIN) * mag)
                
                acc_x += masses[j] * tmp_x
                acc_y += masses[j] * tmp_y
        
        accelerations[i, 0] = acc_x
        accelerations[i, 1] = acc_y


# + Kernel para atualizar posições
@cuda.jit
def update_positions_kernel(positions, velocities, accelerations, dt, n):
    i = cuda.grid(1)
    if i < n:
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt
        velocities[i, 0] += accelerations[i, 0] * dt
        velocities[i, 1] += accelerations[i, 1] * dt

class Simulation:
    def __init__(self, seed: int, num_bodies: int, threads: int):
        random.seed(seed)
        self.num_bodies = num_bodies
        self.threads = threads
        self.bodies = [rand_body() for _ in range(num_bodies)]

        # Calcular centro de massa
        self.total_mass = sum(b.mass for b in self.bodies)
        self.vel_com = sum(b.vel * b.mass for b in self.bodies) / self.total_mass
        self.pos_com = sum(b.pos * b.mass for b in self.bodies) / self.total_mass

        # Ajustar velocidades e normalizar posições
        max_mag = max(np.linalg.norm(b.pos) for b in self.bodies)
        for b in self.bodies:
            b.vel -= self.vel_com
            b.pos = (b.pos - self.pos_com) / max_mag

        # Criaa arrays na GPU
        ###########################
        self.d_positions = cuda.to_device(np.array([b.pos for b in self.bodies], dtype=np.float32))
        self.d_velocities = cuda.to_device(np.array([b.vel for b in self.bodies], dtype=np.float32))
        self.d_masses = cuda.to_device(np.array([b.mass for b in self.bodies], dtype=np.float32))
        self.d_accelerations = cuda.device_array_like(self.d_positions)

    def update(self):
        n = self.num_bodies
        threads_per_block = 128
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

        # Computa forças
        compute_forces_kernel[blocks_per_grid, threads_per_block](self.d_positions, self.d_masses, self.d_accelerations, n)

        # Atualiza posições e velocidades
        update_positions_kernel[blocks_per_grid, threads_per_block](self.d_positions, self.d_velocities, self.d_accelerations, DT, n)

    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Simulação de Corpos")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True)

        scatters = [ax.scatter(0, 0, s=b.mass * 50) for b in self.bodies]
        avg_time_text = ax.text(-1.4, 1.4, '', fontsize=12, color='red')

        def update_plot(frame):
            start_time = time.time()

            for _ in range(500):
                self.update()

            # Copiar posições da GPU para a CPU uma única vez por frame
            ######################################
            positions = self.d_positions.copy_to_host()
            for i in range(self.num_bodies):
                scatters[i].set_offsets(positions[i])
            ######################################

            elapsed_time = time.time() - start_time
            time_per_frame.append(elapsed_time)


            if len(time_per_frame) > 0:
                avg_time = np.average(time_per_frame)
                avg_time_text.set_text(f'Média: {(avg_time * 1000):.2f} ms | Frames: {frame + 1}')

            return scatters + [avg_time_text]

        ani = FuncAnimation(fig, update_plot, frames=200, interval=1, blit=True)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-Body Simulation")
    parser.add_argument("--n", type=int, default=20, help="Numero de corpos")
    parser.add_argument("--threads", type=int, default=128, help="Threads por bloco")
    args = parser.parse_args()

    print(f"Iniciando simulação com {args.n} corpos")
    sim = Simulation(1, num_bodies=args.n, threads=args.threads)
    sim.plot()
