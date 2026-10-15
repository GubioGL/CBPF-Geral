# As 3 melhores configurações no hardware (FakeBerlin)

Caso: ν = 0,1, T = 0,1, N = 10 pontos de colocação por época, **110 épocas com o QPU no laço**
(1 job por época), 3/7/10 qubits, 1024 shots.

As configurações são **as mesmas** do caso exato, sem nenhum ajuste para o hardware. A tabela
completa dos hiperparâmetros está em
[`quantum_inverse_exact_E110/README_melhores_configuracoes.md`](../quantum_inverse_exact_E110/README_melhores_configuracoes.md),
e o código em `CONFIGS` de `Inverser_quantum_exact_pinn_E110.py`, importado pelo script de hardware.

| | Rede (AdamW) | Pesos da loss | ν (em log ν) |
|---|---|---|---|
| 1. `gradnorm_varpro` (padrão) | lr 2,55·10⁻² → 6,9·10⁻⁴, cosseno | grad-norm a cada época | varpro, β 0,46 → 0,099 |
| 2. `fixed_varpro` | lr 1,85·10⁻² → 1,8·10⁻⁴, cosseno | fixos, w_d = 15,4 | varpro, β 0,166 → 0,013 |
| 3. `fixed_adam` | lr 2,17·10⁻² → 4,7·10⁻³, exponencial | fixos, w_d = 20,8 | Adam, lr 0,213 → 9,9·10⁻³ |

## Resultados no FakeBerlin (seeds 0, 1, 2): ν aprendido (erro relativo)

**1. `gradnorm_varpro`**: mediana 1,49%, média 1,76%, máx 4,6%. Erro L2 de u(x,T) no QPU: 1,1·10⁻².
É a configuração da figura [`nu_summary_10_E110.png`](T_0.1/fake_berlin/gradnorm_varpro/nu_summary_10_E110.png).

| Qubits | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 3 | 0,1046 (4,6%) | 0,0985 (1,5%) | 0,0981 (1,9%) |
| 7 | 0,1003 (0,3%) | 0,1004 (0,4%) | 0,0979 (2,1%) |
| 10 | 0,1031 (3,1%) | 0,1009 (0,9%) | 0,0988 (1,2%) |

**2. `fixed_varpro`**: mediana 1,21%, média 2,53%, máx 8,3%. Erro L2 de u(x,T) no QPU: 1,5·10⁻².

| Qubits | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 3 | 0,1006 (0,6%) | 0,0917 (8,3%) | 0,0936 (6,4%) |
| 7 | 0,0987 (1,3%) | 0,0964 (3,6%) | 0,1012 (1,2%) |
| 10 | **0,1000 (0,02%)** | 0,0993 (0,7%) | 0,0995 (0,5%) |

**3. `fixed_adam`**: mediana 1,44%, média 1,33%, máx 2,8%. Erro L2 de u(x,T) no QPU: 2,2·10⁻².

| Qubits | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 3 | 0,1027 (2,7%) | 0,0978 (2,2%) | 0,1016 (1,6%) |
| 7 | 0,1014 (1,4%) | 0,0985 (1,5%) | 0,1011 (1,1%) |
| 10 | 0,0996 (0,4%) | 0,1008 (0,8%) | 0,1002 (0,2%) |

As melhores runs individuais:
- `fixed_varpro`, 10 qubits, seed 0: ν = 0,09998 (0,02%);
- `fixed_adam`, 10 qubits, seed 2: ν = 0,10017 (0,17%);
- `gradnorm_varpro`, 7 qubits, seed 0: ν = 0,10034 (0,34%).

## Estatística: 60 runs por configuração (proxy do fake, seeds 200–219)

| | mediana | média | máx | mediana com 3 / 7 / 10 qubits | L2 de u(x,T) no QPU |
|---|---|---|---|---|---|
| `gradnorm_varpro` | 1,29% | 1,59% | **4,7%** | 2,13% / 1,03% / 0,86% | 1,1·10⁻² |
| `fixed_varpro` | 1,34% | 1,67% | 11,0% | 1,56% / 1,02% / 1,81% | 1,1·10⁻² |
| `fixed_adam` | 1,26% | 1,60% | 7,0% | 1,87% / 1,16% / 1,26% | 2,3·10⁻² |

As três empatam na mediana. O que separa as configurações são os casos ruins e a qualidade de u.

## Qual usar no hardware
- **`gradnorm_varpro`** é o padrão. Tem o menor erro máximo nas 60 runs (4,7%) e o melhor u(x,T) no QPU,
  empatado com o `fixed_varpro`. Além disso, não depende de um w_d ajustado à mão: o grad-norm se adapta
  sozinho, o que é útil se a escala das perdas mudar com o ruído de outro chip.
- **`fixed_adam`** é o mais estável nas 9 runs do fake (máx 2,8%). Mas o u(x,T) dele é ~2× pior, e nas
  60 runs do proxy chega a 7% de erro.
- **`fixed_varpro`** tem as piores caudas com 3 qubits: 8,3% no fake e 11% no proxy.

Com 3 qubits, o erro é maior em todas (mediana ~1,6–2,1% no proxy). Com 7 e 10 qubits, a mediana fica
em ~0,9–1,8%.
