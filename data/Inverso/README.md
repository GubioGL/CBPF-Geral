# Problema inverso de Burgers em 20 000 épocas: PINN clássica × HQPINN com o circuito ideal

Mesmo treino de [`Inverso_110`](../Inverso_110/README.md) e [`Inverso_500`](../Inverso_500/README.md) (pastas apagadas em 09/11/2026; estão no histórico do git, commit `201f0b3c6`): os mesmos
CONFIGs (a melhor configuração do otimizador de cada modelo), os mesmos ajustes em T = 1, ν = 0,01, **N = 10** pontos de
colocação novos por época, 3/7/10 neurônios ou qubits, seeds 0–10, ou seja, 33 runs por modelo e por T. Mudam só:

- **20 000 épocas** (as decaídas cosseno da lr e do β do varpro se estendem por elas);
- a camada quântica é o **circuito ideal**, sem ruído e sem QPU fake (`quantico_ideal/`).

```
python run/Inverser_runs_paralelo.py --benchmark   # estimativa do tempo de tudo
python run/Inverser_runs_paralelo.py               # as 132 runs, 12 processos, barra de progresso (retoma de onde parou)
```

Rodado em 09/11/2026 neste Linux: cerca de 1 h 32 min com 12 processos (`run.log`, `benchmark.json`).

## Resultado

| Erro de ν, 33 runs | mediana / média / máx | por tamanho (3 / 7 / 10), mediana | L2 de u(x,T), mediana |
|---|---|---|---|
| T = 0,1, clássico | **0,6** / 0,8 / 3,1 % | 0,8 / 0,5 / 0,4 % | **0,003** |
| T = 0,1, HQPINN ideal | 1,0 / 1,6 / 9,7 % | 1,9 / 1,0 / 1,0 % | 0,005 |
| T = 1, clássico | **2,9** / 3,4 / 14,0 % | 4,7 / 2,9 / 1,8 % | **0,007** |
| T = 1, HQPINN ideal | 3,1 / 8,4 / 51,4 % | 11,7 / 1,9 / 2,2 % | 0,012 |

- Com 20 000 épocas os dois modelos identificam ν também em T = 1 (com 500 épocas a mediana era 21 % no clássico e
  30 % no quântico no FakeAachen). O clássico não tem ruído em nenhum dos casos, então para ele a diferença é só o
  número de épocas.
- O clássico tem o menor erro de ν em 22 e 21 dos 33 pares (Wilcoxon pareado p = 0,024 e 0,023) e ajusta u melhor.
- A HQPINN com 3 qubits é a exceção em T = 1 (mediana 11,7 %, máx. 51 %); com 7 e 10 qubits fica no nível do clássico.
- O ν aprendido ainda oscila no fim: desvio padrão nas últimas 20 épocas de 0,4–0,8 % (T = 0,1) e 2–3 % (T = 1).
