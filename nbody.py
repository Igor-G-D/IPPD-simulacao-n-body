import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

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


class Simulation:
    def __init__(self, seed: int):
        random.seed(seed)
        self.bodies = []

        n = 5  # Número de corpos na simulação
        for _ in range(n):
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
        # Itera sobre cada corpo na simulação
        for i in range(len(self.bodies)):
            p1 = self.bodies[i].pos  
            m1 = self.bodies[i].mass 

            # Compara esse corpo com todos os outros que ainda não foram comparados
            for j in range(i + 1, len(self.bodies)):
                p2 = self.bodies[j].pos  
                m2 = self.bodies[j].mass 

                # Vetor de deslocamento entre os corpos
                r = p2 - p1
                
                # Calcula a raiz da magnitude da distância
                mag_sq = np.dot(r, r)
                
                # Calcula a magnitude da distâcia
                mag = np.sqrt(mag_sq)
                
                # Usa max(mag_sq, MIN) para evitar divisão por 0, evita que corpos ganhem uma aceleração gigantesca quando se aproximam muito
                # Computa a força usando a equação de gravidade (G = 1)
                tmp = r / (max(mag_sq, MIN) * mag)

                # Atualiza as velocidades com a terceira lei de newton
                self.bodies[i].acc += m2 * tmp  
                self.bodies[j].acc -= m1 * tmp  

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

        # Intervalo de 2ms para atualizar o gráfico
        ani = FuncAnimation(fig, update_plot, frames=total_frames, interval=1, blit=True)
        plt.show()

if __name__ == "__main__":
    seed = random.randint(0, 10000)
    sim = Simulation(seed)
    sim.plot()
