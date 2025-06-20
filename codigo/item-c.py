import numpy as np
import matplotlib.pyplot as plt

# Parâmetros iniciais
taxa_crescimento = 0.5
capacidade_meio = 10
pop_inicial = 1
passo = 0.05
tempo_inicial = 0
tempo_final = 10

# Função da equação logística
def derivada_logistica(tempo, populacao):
  return taxa_crescimento * populacao * (1 - populacao / capacidade_meio)

# Solução analítica
def solucao_analitica(tempo):
  return (capacidade_meio * pop_inicial * np.exp(taxa_crescimento * tempo)) / (capacidade_meio + pop_inicial * (np.exp(taxa_crescimento * tempo) - 1))

# Método de Runge-Kutta de 4ª ordem (RK4)
def metodo_rk4(passo, tempo_final):
  num_passos = int((tempo_final - tempo_inicial) / passo) + 1
  tempos = np.linspace(tempo_inicial, tempo_final, num_passos)
  populacoes = np.zeros(num_passos)
  populacoes[0] = pop_inicial
  for i in range(1, num_passos):
    t = tempos[i - 1]
    y = populacoes[i - 1]
    k1 = derivada_logistica(t, y)
    k2 = derivada_logistica(t + passo / 2, y + passo * k1 / 2)
    k3 = derivada_logistica(t + passo / 2, y + passo * k2 / 2)
    k4 = derivada_logistica(t + passo, y + passo * k3)
    populacoes[i] = y + (passo / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
  return tempos, populacoes

# Execução do método RK4
tempos_rk4, populacoes_rk4 = metodo_rk4(passo, tempo_final)
populacoes_analitica = solucao_analitica(tempos_rk4)

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(tempos_rk4, populacoes_rk4, 'go-', label='RK4 (h=0.05)', markersize=3)
plt.plot(tempos_rk4, populacoes_analitica, 'r-', label='Solução Analítica')
plt.title('RK4 vs Solução Analítica [0, 10]')
plt.xlabel('Tempo')
plt.ylabel('População y(t)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()