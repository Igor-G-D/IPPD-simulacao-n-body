import numpy as np
import random
import matplotlib.pyplot as plt

class Body:
    def __init__(self, pos: np.ndarray, vel: np.ndarray, mass: float):
        self.pos = pos  # vetor 2D de posição
        self.vel = vel  # vetor 2D de velocidade
        self.acc = np.zeros(2)  # vetor 2D de aceleração (inicializado com 0)
        self.mass = mass  # "peso" do corpo

    def update(self, dt: float):
        # atualiza a posição e velocidade
        self.pos += self.vel * dt
        self.vel += self.acc * dt
        self.acc = np.zeros(2)  # reseta aceleração para 0


def rand_body() -> Body:
    # gera um corpo de posição e massa aleatória
    pos = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])  # posição aleatória entre -1 e 1
    vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])  # velocidade aleatória entre -1 e 1
    mass = random.uniform(0.1, 10.0)  # Peso aleatório entre 0.1 e 10
    return Body(pos, vel, mass)


class Simulation:
    def __init__(self, seed: int):
        random.seed(seed)
        self.bodies = []

        n = 10  # Número de corpos na simulação
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

    def __repr__(self):
        return "\n".join(repr(b) for b in self.bodies)

    def plot(self):
        plt.figure(figsize=(6, 6))
        for b in self.bodies:
            plt.scatter(b.pos[0], b.pos[1], s=b.mass * 50, label=f"Mass={b.mass:.2f}")

        # Centro de massa
        plt.scatter(self.pos_com[0], self.pos_com[1], s=200, marker="x", color="black", label="Center of Mass")

        plt.title("Posições iniciais dos corpos e Centro de Massa")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.show()


if __name__ == "__main__":
    seed = random.randint(0,10000)
    sim = Simulation(seed)
    sim.plot()