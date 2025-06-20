import numpy as np
import matplotlib.pyplot as plt

# Parâmetros iniciais
taxa_padrao = 0.5
capacidade_padrao = 10
pop_inicial = 1
tempo_inicial = 0
tempo_final = 4

# Função da equação logística
def derivada_logistica(tempo, populacao, taxa, capacidade):
  return taxa * populacao * (1 - populacao / capacidade)

# Solução analítica
def solucao_analitica(tempo, taxa, capacidade):
  return (capacidade * pop_inicial * np.exp(taxa * tempo)) / (capacidade + pop_inicial * (np.exp(taxa * tempo) - 1))

# Método de Euler com parâmetros variáveis
def metodo_euler(passo, tempo_final, taxa, capacidade):
  num_passos = int((tempo_final - tempo_inicial) / passo) + 1
  tempos = np.linspace(tempo_inicial, tempo_final, num_passos)
  populacoes = np.zeros(num_passos)
  populacoes[0] = pop_inicial
  for i in range(1, num_passos):
    populacoes[i] = populacoes[i - 1] + passo * derivada_logistica(tempos[i - 1], populacoes[i - 1], taxa, capacidade)
  return tempos, populacoes

# Testes com diferentes configurações
configuracoes = [
  (0.1, 0.3, 15),
  (0.01, 0.5, 10),
  (0.2, 0.7, 8)
]

for passo_teste, taxa_teste, capacidade_teste in configuracoes:
  tempos, populacoes_euler = metodo_euler(passo_teste, tempo_final, taxa_teste, capacidade_teste)
  populacoes_exatas = solucao_analitica(tempos, taxa_teste, capacidade_teste)
  plt.figure(figsize=(10, 6))
  plt.plot(tempos, populacoes_euler, 'bo-', label=f'Euler (h={passo_teste})', markersize=3)
  plt.plot(tempos, populacoes_exatas, 'r-', label='Solução Analítica')
  plt.title(f'h={passo_teste}, r={taxa_teste}, K={capacidade_teste}')
  plt.xlabel('Tempo')
  plt.ylabel('População y(t)')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()