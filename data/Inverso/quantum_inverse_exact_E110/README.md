# PINN híbrida inversa (camada quântica exata) em 110 épocas

Estimativa da viscosidade ν da equação de Burgers com a PINN híbrida
`pré-rede clássica → camada CNOT-cosseno exata → pós-rede clássica`, treinada com apenas
**110 épocas**. O código está em [`Inverser_quantum_exact_pinn_E110.py`](../../../Inverser_quantum_exact_pinn_E110.py).
As configurações estão detalhadas em [`README_melhores_configuracoes.md`](README_melhores_configuracoes.md).
A verificação das mesmas configurações com a camada quântica medida no QPU (FakeBerlin) está em
[`quantum_inverse_hardware_E110`](../quantum_inverse_hardware_E110/README.md).

## Problema

| | |
|---|---|
| Equação | u_t + u u_x = ν u_xx em [0, 1) periódico, u(x,0) = sin(2πx), imposta exatamente na rede |
| ν verdadeiro | 0,1 |
| Tempo final | T = 0,1 |
| Dados | 100 pontos da solução exata (Cole-Hopf) em t = T |
| Colocação | N = 10 pontos aleatórios, sorteados de novo a cada época |
| Resíduo | diferenças finitas sobre a rede (hx = 1/512, ht = 10⁻³), resíduo = A − ν·B com A = u_t + u u_x e B = u_xx |
| Modelo | pré-rede 10 neurônios → 3, 7 ou 10 qubits (brickwork CNOT, Ry) → pós-rede 10 neurônios |
| Orçamento | 110 épocas = 110 passos, com **uma** avaliação da rede por época (sem L-BFGS) |

O modelo, os dados e o resíduo são os mesmos de `Inverser_quantum_exact_pinn_paralelo.py`. Só o treino mudou.

Piso do método: aplicado à solução exata, o resíduo por diferenças finitas recupera ν com erro de
0,003%. Portanto, todo o erro abaixo vem do treino.

## Resultado

Erro relativo de ν:

| | Seeds de teste 0, 1, 2 (9 runs) | Seeds de validação 200–219 (60 runs) |
|---|---|---|
| Protocolo original cortado em 110 épocas | mediana 14,3%, média 12,4%, máx 24,7% | mediana 39%, média 76%, máx 298% |
| **Novo treino** (`gradnorm_varpro`) | **mediana 1,5%, média 1,8%, máx 3,1%** | **mediana 1,0%, média 1,2%, máx 5,4%** |

O erro relativo L2 de u(x,t) em todo o domínio caiu de 4,5·10⁻² para 1,5·10⁻².

![ν × época](T_0.1/gradnorm_varpro/nu_summary_10_E110.png)

## Diagnóstico: por que o protocolo original falha em 110 épocas

Na primeira época, a MAE do resíduo (~12–20) é **50 a 100 vezes** maior que a MAE dos dados
(~0,15–0,27). Com pesos 1:1, a rede passa a satisfazer a EDP com o ν errado do chute inicial
(ν₀ ~ U(0, 1)) e quase não ajusta os dados. A MAE dos dados fica em ~0,1, cerca de 15% da amplitude de
u(x,T). Como ν só recebe gradiente do resíduo, e a rede já se adaptou ao ν errado, ν estaciona longe de 0,1.

## O que ajudou, por ordem de impacto

Cada linha da tabela troca **um** ingrediente das 3 melhores configurações e mede o efeito em 60 runs de
validação, pareadas pelas mesmas seeds (teste de Wilcoxon). A saída completa está em
[`T_0.1/search/ablation_ingredientes.txt`](T_0.1/search/ablation_ingredientes.txt).

| # | Ingrediente | Sem ele (mediana do erro de ν) | Com ele |
|---|---|---|---|
| 1 | **Peso dos dados ≫ peso do resíduo** (w_d/w_r ≈ 10–20) | pesos 1:1: **37–77%** | ~1% |
| 2 | **Otimizador separado para ν, com passos fortes** | Adam em ν com lr 0,05→0,005: mediana 2,5–3,3%, mas **média ~30% e máx >100%** (runs em que ν não chega) | ~1%, máx 5–8% |
| 3 | **MAE** no lugar de MSE (quando os pesos são fixos) | MSE: **24–33%** | ~1% |
| 4 | lr da rede ~0,02 com decaimento (cosseno/exponencial) | 1e-2 constante: 1,4–1,9% (p = 0,001 e 0,01 em 2 das 3 configurações) | ~1% |

### 1. Peso dos dados ≫ peso do resíduo
É o fator mais importante. O peso dos dados precisa ser 10–20 vezes o do resíduo para compensar
a diferença de escala da primeira época. Há duas formas de chegar lá, com o mesmo resultado:
- **peso fixo**: w_d = 15–20, ajustado na busca;
- **grad-norm**: em toda época, w_k = (‖g_r‖ + ‖g_d‖) / ‖g_k‖, em que g_k é o gradiente da loss k
  em relação aos pesos da rede (Wang, Teng & Perdikaris, SIAM J. Sci. Comput. 2021). Sozinho, ele chega a
  w_d/w_r ≈ 9–14, sem precisar de ajuste.

### 2. Dois otimizadores: um para a rede e outro só para ν
A rede usa AdamW. ν é otimizado em log (ν = e^s), então os passos são relativos e ν é
sempre positivo. O que importa é que ν acompanhe a rede **rapidamente**. Há duas opções equivalentes:
- **varpro**: com a rede fixa, o ν que minimiza mean|A − ν·B| no lote é a **mediana de A/B
  ponderada por |B|**. A cada época, log ν é puxado na direção desse ν*:
  `log ν ← log ν + β·(log ν* − log ν)`, com β decaindo (por exemplo, 0,46 → 0,1). Isso não exige nenhuma avaliação
  extra da rede, porque A e B já saíram do forward da época.
- **Adam** em log ν com lr alto (0,21 → 0,01, cosseno), alimentado só pelo gradiente da loss do resíduo
  (ν não aparece na loss de dados, e os pesos não reescalam esse gradiente).

Com lr baixo em ν, a rede se adapta ao ν errado antes que ele chegue e o treino estaciona. É daí que
vêm os erros acima de 100%.

### 3. MAE
Com pesos fixos, trocar MAE por MSE muda a escala relativa dos dois termos, e o w_d ajustado deixa de
servir: o erro sobe para 24–33%. Com **grad-norm, MSE e MAE empatam** (1,06% contra 0,98%, p = 0,54),
porque o grad-norm reequilibra a escala sozinho. Essa é mais uma vantagem dele: o resultado depende menos
de escolhas manuais.

### 4. lr da rede
AdamW começa com lr ≈ 0,02 e decai até ~10⁻⁴–5·10⁻³ ao longo das 110 épocas.

## Balanceamento da loss: grad-norm vs NTK

Testei a **mesma** configuração mudando só o esquema de pesos, em 60 runs pareadas
([`T_0.1/search/ablation_pesos.txt`](T_0.1/search/ablation_pesos.txt)):

| Esquema | w_d/w_r ao final | Erro de ν (mediana) |
|---|---|---|
| Fixo, ajustado | 15–21 | 0,9–1,1% |
| **Grad-norm** | ≈ 10–14 | 0,7–1,3% nas configurações ajustadas para ele; empata com o fixo ajustado (p = 0,32–0,49) |
| NTK (tr K, média por ponto; Wang, Yu & Perdikaris, JCP 2022) | ≈ 165–185 | **2,7–3,6%**, pior em todas as 4 configurações testadas (p < 0,05) |

**Conclusão:** o grad-norm funciona e é recomendado, porque encontra sozinho a razão certa. O NTK
põe peso demais nos dados (~180×) e piora a estimativa de ν. Por isso ele foi retirado do código.

## O que foi neutro ou não ajudou
- **Rampa do resíduo** (peso do resíduo subindo de 0 a 1 nas primeiras 10–40 épocas): estava em todas
  as vencedoras da busca, mas removê-la não muda o resultado (p = 0,12–0,94). Ficou no código só para
  reproduzir exatamente os resultados salvos. `r_ramp = 1` a desliga.
- **NTK**: ver acima.
- **Congelar ν nas primeiras épocas** (`nu_warmup`) e **MSE**: a busca não os escolheu e eles foram
  retirados do código.

## Como as configurações foram escolhidas
Para que os números de teste não fossem otimistas, a escolha foi feita em seeds separadas:
1. **Busca**: 768 configurações (480 aleatórias + 2 rodadas de refinamento em torno das melhores de cada
   esquema) × 18 runs (3, 7, 10 qubits × seeds 100–105). Arquivo: `search/search_trials.json`.
2. **Validação**: as 5 melhores de cada par (esquema de pesos × otimizador de ν) rodadas de novo em
   60 runs (seeds 200–219), e a escolha foi feita aqui. Arquivo: `search/validation.json`.
3. **Teste**: seeds 0, 1, 2, nunca usadas para escolher nada.

Arredondar os hiperparâmetros mudou a mediana de validação de 0,71% para 0,98%. Portanto, diferenças
menores que ~0,3 ponto percentual entre configurações são ruído. Com 3 qubits o erro é um pouco maior
(mediana 1,2–1,4%, contra 0,7–1,3% com 7 e 10 qubits).

## Estrutura da pasta
```
quantum_inverse_exact_E110/
├── README.md                          este arquivo
├── README_melhores_configuracoes.md   as 3 melhores configurações e seus resultados
└── T_0.1/
    ├── gradnorm_varpro/   configuração padrão: os 9 runs da figura nu_summary_10_E110.png
    ├── fixed_varpro/
    ├── fixed_adam/
    │     cada uma: model_10_NQ<q>_<seed>_E110.pt (+ _config.json), losshist_*.npz (loss, pesos, ν
    │     por época), loss_*.png, nu_history_*.png, solution_*.png, nu_summary_10_E110.png, summary.json
    └── search/
        ├── search_trials.json         busca (768 configurações)
        ├── validation.json            validação das 5 melhores de cada esquema
        ├── top3_rounded.json          as 3 configurações arredondadas: validação + teste
        ├── test_all_schemes.json      todos os esquemas + protocolo original nas seeds de teste
        │                              (DEFAULT_CFG = gradnorm_varpro, ADAM_NU_CFG = fixed_adam)
        ├── ablation_pesos.txt         troca só do esquema de pesos
        ├── ablation_ingredientes.txt  troca de um ingrediente por vez
        └── search_code_snapshot.py    cópia congelada do código que fez a busca e a validação
```

## Como reproduzir
No ambiente conda `qiskit` (o `python` padrão do sistema não tem torch), a partir da raiz do repositório:
```
python Inverser_quantum_exact_pinn_E110.py                          # as 3 configurações (~1 min)
python Inverser_quantum_exact_pinn_E110.py --config gradnorm_varpro # só a padrão
```
Os resultados são determinísticos: o script atual reproduz bit a bit os pesos e as histórias salvos aqui.
