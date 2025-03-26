# IPPD-simulacao-n-body
Trabalho final de implementação de Introdução a Processamento Paralelo e Distribuido

Integrantes:
- Igor Dutra
- Alexandre Cardoso
- Leonardo Melo
- Eloisa Barros

baseado nos materiais:
- https://medium.com/swlh/create-your-own-n-body-simulation-with-python-f417234885e9
- https://youtu.be/L9N7ZbGSckk?si=7Pi-G1EuPXtibarb
- https://numba.readthedocs.io/en/stable/cuda/index.html
- https://www.youtube.com/watch?v=9bBsvpg-Xlk

Para rodar:
- python nbody.py --n {número de corpos}
- python nbody_cuda.py --n {número de corpos} --threads {número de threads por bloco}
- python nbody_cuda_shared_memory.py --n {número de corpos} --threads {número de threads por bloco, max 128}



