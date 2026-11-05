# Problema inverso de Burgers em 500 épocas: PINN clássica × PINN quântica (FakeAachen)

Mesmo estudo de [`Inverso_110`](../Inverso_110/README.md), com **500 épocas** em vez de 110 e o QPU fake
**FakeAachen** (`ibm_aachen`) em vez do FakeBerlin. Tudo o mais é igual: os mesmos scripts e CONFIGs, ν = 0,01,
T = 0,1 e T = 1, **N = 10** pontos de colocação novos por época, 3/7/10 neurônios ou qubits, seeds 0–10,
ou seja, 33 runs por modelo e por caso. Em T = 1 o treino usa os ajustes (IC suave, 10 épocas de warmup com ν
congelado, lr com plateau), com os mesmos valores em épocas. As decaídas cosseno (lr da rede e β do varpro)
se estendem pelas 500 épocas.

```
python run/Inverser_classical_pinn_E110.py --tfin 0.1 --nu 0.01 --epochs 500 --out-dir data/Inverso_500
python run/Inverser_classical_pinn_E110.py --tfin 1.0 --nu 0.01 --epochs 500 --out-dir data/Inverso_500
python run/Inverser_quantum_hardware_pinn_E110.py --tfin 0.1 --nu 0.01 --epochs 500 --backend ibm_aachen --out-dir data/Inverso_500 --workers 12
python run/Inverser_quantum_hardware_pinn_E110.py --tfin 1.0 --nu 0.01 --epochs 500 --backend ibm_aachen --out-dir data/Inverso_500 --workers 12
```

**Cadeias no ibm_aachen:** não havia cadeias registradas para o Aachen. Elas foram calculadas a partir da
tabela de erros do backend: todos os caminhos de n qubits acoplados, com nota = erros de readout + sx dos
qubits + erros de CZ das arestas, e depois cadeias disjuntas escolhidas da melhor para a pior. Foram usadas
41, 14 e 10 cadeias (3, 7 e 10 qubits). O rms do ajuste da calibração é 0,027 com 3 qubits (0,032 no
Berlin). As calibrações estão em `quantico/calibration_NQ*_fake_aachen.npz`.

## Resultado

| Erro de ν, 33 runs | mediana / média / p90 / máx | por tamanho (3 / 7 / 10), mediana | L2 de u(x,T), mediana |
|---|---|---|---|
| T = 0,1, clássico | **2,3** / 3,1 / 7 / 12 % | 2,6 / 2,3 / 1,4 % | **0,005** |
| T = 0,1, quântico (fake) | 2,8 / 3,2 / 7 / 13 % | 3,2 / 3,0 / 2,6 % | 0,013 |
| T = 1, clássico | **20,9** / 26,0 / 51 / 63 % | 16,9 / 33,7 / 14,6 % | **0,044** |
| T = 1, quântico (fake) | 30,5 / 30,7 / 62 / 75 % | 41,2 / 27,8 / 12,8 % | 0,068 |

- **T = 0,1:** com 500 épocas, os dois modelos melhoram muito em relação às 110 épocas: a mediana do erro de ν
  cai de 7,3 para 2,3 % no clássico e de 6,0 para 2,8 % no quântico. No erro de ν há empate (Wilcoxon pareado
  p = 0,76; o quântico ganha 16 dos 33 pares). O clássico ajusta u melhor.
- **T = 1:** as épocas extras quase não mudam ν (clássico 21,5 → 20,9 %; quântico 30,2 → 30,5 %), mas
  melhoram u(x,T) (L2 0,073 → 0,044 e 0,107 → 0,068). Várias runs terminam com ν bem abaixo do verdadeiro
  (≈ 0,003–0,006), nos dois modelos. A vantagem do clássico não é significativa aqui (p = 0,37; o quântico
  ganha 15 dos 33 pares).

Os valores de 110 épocas, para comparação, foram obtidos no FakeBerlin. A diferença entre os dois QPUs fake
não foi separada da diferença no número de épocas.
