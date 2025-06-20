import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do item (a)
taxa_crescimento = 0.5
capacidade_meio = 10
pop_inicial = 1
passo = 0.05
tempo_inicial = 0
tempo_final = 4

# Função da equação logística
def derivada_logistica(tempo, populacao):
  return taxa_crescimento * populacao * (1 - populacao / capacidade_meio)

# Solução analítica
def solucao_analitica(tempo):
  return (capacidade_meio * pop_inicial * np.exp(taxa_crescimento * tempo)) / (capacidade_meio + pop_inicial * (np.exp(taxa_crescimento * tempo) - 1))

# Método de Euler
num_passos = int((tempo_final - tempo_inicial) / passo) + 1
tempos = np.linspace(tempo_inicial, tempo_final, num_passos)
populacoes_euler = np.zeros(num_passos)
populacoes_euler[0] = pop_inicial

for i in range(1, num_passos):
  populacoes_euler[i] = populacoes_euler[i - 1] + passo * derivada_logistica(tempos[i - 1], populacoes_euler[i - 1])

# Solução analítica para comparação
populacoes_analitica = solucao_analitica(tempos)

# Plotando os resultados
plt.figure(figsize=(10, 6))
plt.plot(tempos, populacoes_euler, 'bo-', label='Euler', markersize=3)
plt.plot(tempos, populacoes_analitica, 'r-', label='Solução Analítica')
plt.title('Equação Logística')
plt.xlabel('Tempo')
plt.ylabel('População y(t)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()