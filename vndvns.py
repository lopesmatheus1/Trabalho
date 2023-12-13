import random
import numpy as np
import time

pontosEntrega = [
    {"id": 1, "coordenadas": (565.0, 575.0)},
    {"id": 2, "coordenadas": (25.0, 185.0)},
    {"id": 3, "coordenadas": (345.0,750.0)},
    {"id": 4, "coordenadas": (945.0,685.0)},
    {"id": 5, "coordenadas": (845.0, 655.0)},
    {"id": 6, "coordenadas": (880.0, 660.0)},
    {"id": 7, "coordenadas": (25.0, 230.0)},
    {"id": 8, "coordenadas": (525.0, 1000.0)},
    {"id": 9, "coordenadas": (580.0, 1175.0)},
    {"id": 10, "coordenadas": (650.0, 1130.0)},
    {"id": 11, "coordenadas": (1605.0, 620.0 )},
    {"id": 12, "coordenadas": (1220.0, 580.0)},
    {"id": 13, "coordenadas": (1465.0, 200.0)},
    {"id": 14, "coordenadas": (1530.0, 5.0)},
    {"id": 15, "coordenadas": (845.0, 680.0)},
    {"id": 16, "coordenadas": (725.0, 370.0)},
    {"id": 17, "coordenadas": (145.0, 665.0)},
    {"id": 18, "coordenadas": (415.0, 635.0)},
    {"id": 19, "coordenadas": (510.0, 875.0 )},
    {"id": 20, "coordenadas": (560.0, 365.0)},
    {"id": 21, "coordenadas": (300.0, 465.0)},
    {"id": 22, "coordenadas": (520.0, 585.0)},
    {"id": 23, "coordenadas": (480.0, 415.0)},
    {"id": 24, "coordenadas": (835.0, 625.0)},
    {"id": 25, "coordenadas": (975.0, 580.0)},
    {"id": 26, "coordenadas": (1215.0, 245.0)},
    {"id": 27, "coordenadas": (1320.0, 315.0)},
    {"id": 28, "coordenadas": (1250.0, 400.0)},
    {"id": 29, "coordenadas": (660.0, 180.0)},
    {"id": 30, "coordenadas": (410.0, 250.0)},
    {"id": 31, "coordenadas": (420.0, 555.0)},
    {"id": 32, "coordenadas": (575.0 ,665.0)},
    {"id": 33, "coordenadas": (1150.0 ,1160.0)},
    {"id": 34, "coordenadas": (700.0 ,580.0)},
    {"id": 35, "coordenadas": (685.0, 595.0)},
    {"id": 36, "coordenadas": (685.0, 610.0)},
    {"id": 37, "coordenadas": (770.0, 610.0)},
    {"id": 38, "coordenadas": (795.0, 645.0)},
    {"id": 39, "coordenadas": (720.0 ,635.0)},
    {"id": 40, "coordenadas": (760.0, 650.0)},
    {"id": 41, "coordenadas": (475.0, 960.0)},
    {"id": 42, "coordenadas": (95.0, 260.0)},
    {"id": 43, "coordenadas": (875.0 ,920.0)},
    {"id": 44, "coordenadas": (700.0, 500.0)},
    {"id": 45, "coordenadas": (555.0 ,815.0)},
    {"id": 46, "coordenadas": (830.0, 485.0)},
    {"id": 47, "coordenadas": (1170.0, 65.0)},
    {"id": 48, "coordenadas": (830.0, 610.0)},
    {"id": 49, "coordenadas": (605.0, 625.0)},
    {"id": 50, "coordenadas": (595.0 ,360.0)},
    {"id": 51, "coordenadas": (1340.0, 725.0)},
    {"id": 52, "coordenadas": (1740.0, 245.0)},
]



# Função para calcular a latência considerando tempo de percurso
def calcularLatencia(ponto1, ponto2):
    
    distancia = np.linalg.norm(np.array(ponto1["coordenadas"]) - np.array(ponto2["coordenadas"]))
    tempo = distancia / velocidade_media
    return tempo

# Função para calcular a latência total de uma rota
def calcularLatenciaTotal(rota, pontos_entrega):
    latenciaTotal = 0
    for i in range(len(rota) - 1):
        latenciaTotal += calcularLatencia(pontos_entrega[rota[i]], pontos_entrega[rota[i + 1]])
    return latenciaTotal

# Função para gerar uma solução inicial aleatória
def gerarSolucao1(n):
    solucaoInicial = list(range(n))
    random.shuffle(solucaoInicial)
    return solucaoInicial

# Função para realizar uma perturbação na solução trocando dois pontos aleatórios
def perturbacao(solucao):
    solucao_perturbada = solucao.copy()
    i, j = random.sample(range(len(solucao)), 2)
    solucao_perturbada[i], solucao_perturbada[j] = solucao_perturbada[j], solucao_perturbada[i]
    return solucao_perturbada

# Função para realizar a busca local usando Variable Neighborhood Descent (VND)
def vnd(solucao, pontos_entrega):
    melhor_solucao_local = solucao.copy()
    melhor_latencia_local = calcularLatenciaTotal(solucao, pontos_entrega)

    for i in range(len(solucao) - 1):
        for j in range(i + 1, len(solucao)):
            solucao_vizinha = solucao[:i] + solucao[i:j][::-1] + solucao[j:]
            latencia_vizinha = calcularLatenciaTotal(solucao_vizinha, pontos_entrega)

            if latencia_vizinha < melhor_latencia_local:
                melhor_solucao_local = solucao_vizinha
                melhor_latencia_local = latencia_vizinha

    return melhor_solucao_local

# Função principal para VNS-VND
def vns_vnd(n_iteracoes, k_max, pontosEntrega):
    melhor_solucao_global = gerarSolucao1(len(pontosEntrega))
    # Loop principal do algoritmo VNS
    for i in range(n_iteracoes):
        k = 1
        while k <= k_max:
            solucao_perturbada = perturbacao(melhor_solucao_global)
            
            solucao_vnd = vnd(solucao_perturbada, pontosEntrega)

            if calcularLatenciaTotal(solucao_vnd, pontosEntrega) < calcularLatenciaTotal(melhor_solucao_global, pontosEntrega):
                melhor_solucao_global = solucao_vnd
                k = 1
            else:
                k += 1

    return melhor_solucao_global

# Registra o tempo de início
start_time = time.time()

# Seu código principal aqui
n_iteracoes = 52
k_max = 10
velocidade_media = 30

melhor_rota = vns_vnd(n_iteracoes, k_max, pontosEntrega)

# Registra o tempo de término
end_time = time.time()

# Calcula o tempo total decorrido
total_elapsed_time = end_time - start_time

# Imprime o tempo total decorrido em segundos
print("Tempo total de execução:", total_elapsed_time, "segundos")

# Restante do seu código, como a impressão da melhor rota e latência
print("Melhor Rota:", [pontosEntrega[i]["id"] for i in melhor_rota])
print("Latência da Melhor Rota:", calcularLatenciaTotal(melhor_rota, pontosEntrega))



