# Problema inverso de Burgers em 200 épocas: PINN clássica × HQPINN no FakeAachen

Mesmo treino da antiga `Inverso_500` (apagada em 09/11/2026; no histórico do git, commit `201f0b3c6`): mesmos CONFIGs,
ajustes em T = 1, ν = 0,01, N = 10 pontos por época, QPU fake **FakeAachen**, com **200 épocas**, só **n = 10**
(neurônios ou qubits) e **seeds 0–9**: 10 runs por modelo e por T, 40 no total. As cadeias e a calibração de 10 qubits
são as da antiga `Inverso_500` (`quantico/calibration_NQ10_fake_aachen.npz`, copiada de lá).

```
python run/Inverser_runs_paralelo.py --preset aachen200 --benchmark   # estimativa do tempo
python run/Inverser_runs_paralelo.py --preset aachen200               # as 40 runs, 12 processos, barra de progresso
```

Rodada pelo autor; cada run quântica levou ~16 min (T = 0,1) e ~32 min (T = 1), mediana, com 12 runs em paralelo.

## Resultado

| Erro de ν, 10 runs | mediana / média / máx | L2 de u(x,T), mediana |
|---|---|---|
| T = 0,1, clássico | 4,6 / 6,5 / 18,0 % | **0,009** |
| T = 0,1, quântico (fake) | **1,9** / 3,7 / 11,6 % | 0,011 |
| T = 1, clássico | **18,0** / 24,9 / 52,2 % | **0,050** |
| T = 1, quântico (fake) | 36,9 / 32,1 / 71,4 % | 0,063 |

- **T = 0,1:** os dois identificam ν; a HQPINN tem a menor mediana, mas a diferença não é significativa (Wilcoxon
  pareado p = 0,43; a HQPINN ganha 6 dos 10 pares).
- **T = 1:** nenhum dos dois identifica ν em 200 épocas; os dois subestimam ν (10/10 e 7/10). Com 20 000 épocas e o
  circuito ideal (`data/Inverso`) os dois chegam a ~3 %.
- Mesmos pesos da HQPINN com o circuito ideal: em T = 0,1 o erro piora em 9 de 10 runs (a rede se adapta ao ruído).
