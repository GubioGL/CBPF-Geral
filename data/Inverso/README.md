# Problema inverso de Burgers em 20 000 épocas: PINN clássica × HQPINN com o circuito ideal

Mesmo treino de [`Inverso_110`](../Inverso_110/README.md) e [`Inverso_500`](../Inverso_500/README.md) (pastas apagadas em 09/11/2026; estão no histórico do git, commit `a36240c96`): os mesmos
CONFIGs (a melhor configuração do otimizador de cada modelo), os mesmos ajustes em T = 1, ν = 0,01, **N = 10** pontos de
colocação novos por época, 3/7/10 neurônios ou qubits, seeds 0–10, ou seja, 33 runs por modelo e por T. Mudam só:

- **20 000 épocas** (as decaídas cosseno da lr e do β do varpro se estendem por elas);
- a camada quântica é o **circuito ideal**, sem ruído e sem QPU fake (`quantico_ideal/`).

```
python run/Inverser_runs_paralelo.py --benchmark   # estimativa do tempo de tudo
python run/Inverser_runs_paralelo.py               # as 132 runs, 12 processos, barra de progresso (retoma de onde parou)
```

Rodado em 09/11/2026 neste Linux: cerca de 1 h 32 min com 12 processos (`run.log`, `benchmark.json`).

**n = 5, 12, 15 (10/11/2026):** o preset `ideal` agora tem n = 3, 5, 7, 10, 12, 15; as 132 runs novas levaram ~1 h 35 min
(`run_5_12_15.log`). Com os seis tamanhos são 66 runs por modelo e por T.

## Resultado (66 runs por modelo e por T)

| Erro de ν | mediana / média / máx | L2 de u(x,T), mediana |
|---|---|---|
| T = 0,1, clássico | **0,8** / 1,0 / 4,6 % | **0,003** |
| T = 0,1, HQPINN ideal | 0,9 / 1,2 / 9,7 % | 0,004 |
| T = 1, clássico | **1,7** / 3,5 / 48,1 % | **0,007** |
| T = 1, HQPINN ideal | 2,5 / 6,6 / 51,4 % | 0,011 |

Mediana do erro de ν por tamanho (n = 3 / 5 / 7 / 10 / 12 / 15):

| | T = 0,1 | T = 1 |
|---|---|---|
| clássico | 0,8 / 1,4 / 0,5 / 0,4 / 1,1 / 0,7 % | 4,7 / 1,6 / 2,9 / 1,8 / 1,4 / 1,8 % |
| HQPINN ideal | 1,9 / 0,6 / 1,0 / 1,0 / 1,0 / 0,7 % | 11,7 / 4,6 / 1,9 / 2,2 / 1,2 / 1,7 % |

- Os dois modelos identificam ν nos dois T. Em T = 0,1 não há diferença significativa (Wilcoxon pareado p = 0,71);
  em T = 1 o clássico tem erro um pouco menor (ganha 38 dos 66 pares, p = 0,034) e ajusta u melhor.
- Em T = 1 a HQPINN melhora com o número de qubits e, a partir de n = 7, fica no nível do clássico.
- O ν aprendido ainda oscila no fim: desvio padrão nas últimas 20 épocas de ~0,5 % (T = 0,1) e ~2 % (T = 1).
