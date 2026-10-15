# As 3 melhores configurações

Caso: ν = 0,1, T = 0,1, N = 10 pontos de colocação por época, **110 épocas**, 3/7/10 qubits.
O porquê de cada escolha está no [`README.md`](README.md).
As três configurações estão em `CONFIGS` de [`Inverser_quantum_exact_pinn_E110.py`](../../../Inverser_quantum_exact_pinn_E110.py),
e os resultados de cada uma em `T_0.1/<nome>/`.

O ranking segue a validação (60 runs: 3/7/10 qubits × seeds 200–219). As três estão **estatisticamente
empatadas**: no teste pareado entre as versões não arredondadas, p = 0,32–0,58.

## Hiperparâmetros

| | 1. `gradnorm_varpro` (padrão) | 2. `fixed_varpro` | 3. `fixed_adam` |
|---|---|---|---|
| **Rede: AdamW** | | | |
| lr inicial → final | 2,55·10⁻² → 6,9·10⁻⁴ | 1,85·10⁻² → 1,8·10⁻⁴ | 2,17·10⁻² → 4,7·10⁻³ |
| decaimento do lr | cosseno | cosseno | exponencial |
| betas | (0,87; 0,999) | (0,75; 0,99) | (0,78; 0,99) |
| weight decay | 2,6·10⁻⁵ | 1,9·10⁻⁴ | 1,0·10⁻⁴ |
| clip da norma do gradiente | 10 | 0,5 | 10 |
| **Loss** | | | |
| termos | MAE(resíduo), MAE(dados) | MAE(resíduo), MAE(dados) | MAE(resíduo), MAE(dados) |
| pesos | **grad-norm** a cada época (w_d/w_r ≈ 9–11) | fixos, w_r = 1, **w_d = 15,4** | fixos, w_r = 1, **w_d = 20,8** |
| rampa do resíduo (neutra) | 10 épocas | 20 épocas | 40 épocas |
| **ν: otimizador separado, em log ν** | | | |
| otimizador | **varpro** | **varpro** | **Adam** |
| lr (Adam) ou relaxação β (varpro), inicial → final | 0,46 → 0,099 | 0,166 → 0,013 | 0,213 → 9,9·10⁻³ |
| decaimento | exponencial | cosseno | cosseno |
| betas | — | — | (0,86; 0,999) |

Valores comuns às três configurações: chute inicial ν₀ = `np.random.random()` após `set_seed(seed)`
(o mesmo dos scripts anteriores); 10 pontos aleatórios novos por época; estado final ao fim das 110
épocas, sem snapshot nem L-BFGS.

## Resultados: erro relativo de ν

### Validação (60 runs)

| | mediana | média | máx | mediana com 3 / 7 / 10 qubits | L2 relativo de u(x,t) |
|---|---|---|---|---|---|
| `gradnorm_varpro` | 0,98% | 1,21% | 5,4% | 1,41% / 0,89% / 0,72% | 1,3·10⁻² |
| `fixed_varpro` | 0,98% | 1,30% | 7,5% | 1,21% / 0,97% / 0,74% | 1,4·10⁻² |
| `fixed_adam` | 1,04% | 1,37% | 6,2% | 1,42% / 0,86% / 1,33% | 1,9·10⁻² |
| protocolo original | 39% | 76% | 298% | | 8,4·10⁻² |

### Teste (seeds 0, 1, 2): ν aprendido (erro relativo)

**1. `gradnorm_varpro`**: mediana 1,47%, média 1,75%, máx 3,1%. É a configuração da figura
[`nu_summary_10_E110.png`](T_0.1/gradnorm_varpro/nu_summary_10_E110.png).

| Qubits | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 3 | 0,1015 (1,5%) | 0,0971 (2,9%) | 0,0969 (3,1%) |
| 7 | 0,1030 (3,0%) | 0,0990 (1,0%) | 0,0995 (0,5%) |
| 10 | 0,1014 (1,4%) | 0,0997 (0,3%) | 0,0980 (2,0%) |

**2. `fixed_varpro`**: mediana 0,78%, média 1,47%, máx 5,6%.

| Qubits | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 3 | 0,1011 (1,1%) | 0,0973 (2,7%) | 0,0944 (5,6%) |
| 7 | 0,1003 (0,3%) | 0,0978 (2,2%) | 0,0999 (0,1%) |
| 10 | **0,1000 (0,03%)** | 0,0992 (0,8%) | 0,1004 (0,4%) |

**3. `fixed_adam`**: mediana 0,82%, média 1,67%, máx 5,1%.

| Qubits | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 3 | 0,1051 (5,1%) | 0,0981 (1,9%) | 0,1001 (0,1%) |
| 7 | 0,1046 (4,6%) | 0,1006 (0,6%) | 0,0992 (0,8%) |
| 10 | 0,0992 (0,8%) | 0,0998 (0,2%) | 0,0990 (1,0%) |

As 3 melhores runs individuais:
- `fixed_varpro`, 10 qubits, seed 0: ν = 0,09997 (0,03%);
- `fixed_varpro`, 7 qubits, seed 2: ν = 0,09994 (0,06%);
- `fixed_adam`, 3 qubits, seed 2: ν = 0,10010 (0,10%).

Com só 9 runs, a ordem das configurações no teste é ruído. A comparação que vale é a validação,
com 60 runs.

## Qual usar
- **`gradnorm_varpro`** é o padrão. Teve o melhor resultado na validação e o menor erro máximo no teste
  (3,1%), e é o único que **não precisa de w_d ajustado à mão**: o grad-norm encontra a razão sozinho e
  também funciona com MSE. É a melhor escolha para mudar o caso (outro ν, T ou N).
- **`fixed_varpro`** é o mais simples: não calcula gradientes extras. Mas o w_d = 15,4 foi ajustado para
  este caso com MAE (com MSE, o erro sobe para 33%).
- **`fixed_adam`** é para quem prefere um otimizador de gradiente clássico em ν. O lr de ν precisa ser
  alto (~0,2): com lr 0,05 → 0,005, algumas runs não convergem.

As três usam **uma** avaliação da rede por época (50 pontos do estêncil + 100 dados). O varpro reaproveita
A e B dessa mesma avaliação, e o grad-norm só acrescenta dois backward. Por isso o custo por época na QPU
continua o mesmo.
