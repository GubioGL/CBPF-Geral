# Problema inverso de Burgers em 200 épocas: PINN clássica × HQPINN no FakeAachen (preparado, ainda não rodado)

Mesmo treino da antiga `Inverso_500` (apagada em 09/11/2026; no histórico do git, commit `201f0b3c6`) (mesmos CONFIGs, ajustes em T = 1, ν = 0,01, N = 10 pontos por
época, QPU fake **FakeAachen**), com **200 épocas**, só **n = 10** (neurônios ou qubits) e **seeds 0–9**: 10 runs por
modelo e por T, T = 0,1 e T = 1, 40 runs no total. As cadeias e a calibração de 10 qubits são as de `Inverso_500`
(`quantico/calibration_NQ10_fake_aachen.npz`, copiada de lá antes de ela ser apagada).

```
python run/Inverser_runs_paralelo.py --preset aachen200 --benchmark   # estimativa do tempo
python run/Inverser_runs_paralelo.py --preset aachen200               # as 40 runs, 12 processos, barra de progresso
```

Benchmark de 09/11/2026 (12 processos): 2,4 s/época (T = 0,1) e 3,9 s/época (T = 1) por run quântica, ou seja, 8 e
13 min por run; o clássico leva segundos. Estimativa para tudo: cerca de 21 min (`benchmark.json`).
