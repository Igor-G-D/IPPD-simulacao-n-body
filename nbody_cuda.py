from numba import cuda
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import argparse

DT = 0.000001
MIN = 0.0001

time_per_frame = []  # Lista para armazenar o tempo de cada frame

class Body:
    def __init__(self, pos: np.ndarray, vel: np.ndarray, mass: float):
        self.pos = pos  # vetor 2D de posição
        self.vel = vel  # vetor 2D de velocidade
        self.acc = np.zeros(2)  # vetor 2D de aceleração (inicializado com 0)
        self.mass = mass  # "peso" do corpo

    def update(self, dt: float):
        self.pos += self.vel * dt
        self.vel += self.acc * dt
        self.acc = np.zeros(2)


def rand_body() -> Body:
    pos = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
    vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
    mass = random.uniform(0.1, 10.0)
    return Body(pos, vel, mass)


@cuda.jit
def compute_forces_kernel(positions, masses, accelerations, n): # função que será executada no gpu
    i = cuda.grid(1)
    if i < n:
        for j in range(n):
            if i != j:
                # Computa a diferença das posições manualmente
                r_x = positions[j, 0] - positions[i, 0]
                r_y = positions[j, 1] - positions[i, 1]
                mag_sq = r_x * r_x + r_y * r_y
                mag = mag_sq ** 0.5  # cálculo de raiz manualmente
                tmp_x = r_x / (max(mag_sq, MIN) * mag)
                tmp_y = r_y / (max(mag_sq, MIN) * mag)
                accelerations[i, 0] += masses[j] * tmp_x
                accelerations[i, 1] += masses[j] * tmp_y


class Simulation:
    def __init__(self, seed: int, num_bodies: int, threads: int):
        random.seed(seed)
        self.num_bodies = num_bodies
        self.threads = threads
        self.bodies = []

        for _ in range(self.num_bodies):
            self.bodies.append(rand_body())  # Gera os corpos
            
        # Calcular o centro de massa (posição e velocidade)
        self.total_mass = sum(b.mass for b in self.bodies)
        self.vel_com = sum(b.vel * b.mass for b in self.bodies) / self.total_mass
        self.pos_com = sum(b.pos * b.mass for b in self.bodies) / self.total_mass

        # Ajustar as velocidades em relação ao centro de massa
        for b in self.bodies:
            b.vel -= self.vel_com
            b.pos -= self.pos_com

        # Normalizar posições em relação a magnitude máxima
        max_mag = max(np.linalg.norm(b.pos) for b in self.bodies)
        for b in self.bodies:
            b.pos /= max_mag

    def update(self):
        n = len(self.bodies)
        
        # Peeparando os dados para CUDA
        positions = np.array([b.pos for b in self.bodies], dtype=np.float32)
        masses = np.array([b.mass for b in self.bodies], dtype=np.float32)
        accelerations = np.zeros_like(positions)
        
        # Alocar memória
        d_positions = cuda.to_device(positions)
        d_masses = cuda.to_device(masses)
        d_accelerations = cuda.to_device(accelerations)
        
        # Número de threads e blocks
        threads_per_block = self.threads
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
        
        # kerner que executa no GPU
        compute_forces_kernel[blocks_per_grid, threads_per_block](d_positions, d_masses, d_accelerations, n)
        
        # Copiar os resultados para o host
        accelerations = d_accelerations.copy_to_host()
        
        # Atualizar as acelerações
        for i, b in enumerate(self.bodies):
            b.acc = accelerations[i]
        
        # Atualizar as posições
        for body in self.bodies:
            body.update(DT)

    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Simulação de Corpos")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True)

        scatters = [ax.scatter(b.pos[0], b.pos[1], s=b.mass * 50) for b in self.bodies]
        total_frames = 200
        
        # Adiciona o texto na posição superior esquerda do gráfico
        avg_time_text = ax.text(-1.4, 1.4, '', fontsize=12, color='red')

        def update_plot(frame):
            start_time = time.time()  # Início da contagem de cada frame
            
            for _ in range(500):  # Número de atualizações entre frames
                self.update()
            
            # Atualiza os dados dos corpos
            for i, b in enumerate(self.bodies):
                scatters[i].set_offsets(b.pos)
            
            elapsed_time = time.time() - start_time
            time_per_frame.append(elapsed_time)
            
            # tempo médio por frame
            if len(time_per_frame) > 0:
                avg_time = np.average(time_per_frame)
                avg_time_text.set_text(f'Média de tempo por frame: {(avg_time * 1000):.2f} ms | Frames: {frame + 1}')
            
            return scatters + [avg_time_text]

        ani = FuncAnimation(fig, update_plot, frames=total_frames, interval=1, blit=True)
        plt.show()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-Body Simulation")
    parser.add_argument("--n", type=int, default=20, help="Numero de corpos na simulação")
    parser.add_argument("--threads", type=int, default=64, help="Numero de threads por bloco")
    args = parser.parse_args()
    seed = 1
    print(f"Iniciando simulação com {args.n} corpos")
    sim = Simulation(seed, num_bodies=args.n, threads=args.threads)
    sim.plot()