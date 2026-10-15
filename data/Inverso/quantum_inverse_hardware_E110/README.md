# PINN híbrida inversa no hardware quântico (IBM) em 110 épocas

Mesmo problema e mesmo treino de [`quantum_inverse_exact_E110`](../quantum_inverse_exact_E110/README.md),
mas com a camada quântica **medida no QPU** a cada época, com ruído de shot e ruído do dispositivo,
em vez da fórmula exata ⟨Z_i⟩ = ∏ cos(φ_j)^M_ij. O código está em
[`Inverser_quantum_hardware_pinn_E110.py`](../../../Inverser_quantum_hardware_pinn_E110.py).
As configurações estão detalhadas em [`README_melhores_configuracoes.md`](README_melhores_configuracoes.md).

**Caso:** ν = 0,1, T = 0,1, N = 10 pontos de colocação por época, 110 épocas, 3/7/10 qubits, seeds 0, 1, 2.

**Onde rodou:** no **FakeBerlin**, o modelo de ruído do `ibm_berlin` (IBM Nighthawk r1, 120 qubits)
simulado localmente com Aer. Nenhum job foi enviado à IBM. O caminho para o QPU real (`--real`) está
no script, mas **não foi executado**.

## Resultado

Erro relativo de ν nas seeds 0, 1, 2 (9 runs por configuração):

| Configuração | FakeBerlin | Mesmo treino, camada exata | L2 relativo de u(x,T) no QPU |
|---|---|---|---|
| `gradnorm_varpro` | mediana 1,49%, média 1,76%, máx 4,6% | 1,47%, 1,75%, máx 3,1% | 1,1·10⁻² |
| `fixed_varpro` | mediana 1,21%, média 2,53%, máx 8,3% | 0,78%, 1,47%, máx 5,6% | 1,5·10⁻² |
| `fixed_adam` | mediana 1,44%, média 1,33%, máx 2,8% | 0,82%, 1,67%, máx 5,1% | 2,2·10⁻² |

Nas mesmas seeds, o ν aprendido no hardware difere do ν da camada exata em ~0,001 (mediana). Ou seja,
**as 3 configurações encontradas no caso exato funcionam no hardware sem nenhum ajuste.**

![ν × época no FakeBerlin](T_0.1/fake_berlin/gradnorm_varpro/nu_summary_10_E110.png)

Com 9 runs, a ordem das configurações é ruído. Para ter estatística, rodei também 60 runs por
configuração (3/7/10 qubits × seeds 200–219) num **proxy do fake**, descrito mais abaixo:

| Configuração | mediana | média | máx | mediana com 3 / 7 / 10 qubits |
|---|---|---|---|---|
| `gradnorm_varpro` | 1,29% | 1,59% | 4,7% | 2,13% / 1,03% / 0,86% |
| `fixed_varpro` | 1,34% | 1,67% | 11,0% | 1,56% / 1,02% / 1,81% |
| `fixed_adam` | 1,26% | 1,60% | 7,0% | 1,87% / 1,16% / 1,26% |
| *camada exata, mesmas seeds* | ~1,0% | 1,2–1,4% | 5–8% | |

As três empatam. O ruído do hardware custa cerca de 0,3 ponto percentual na mediana do erro de ν.

## O que foi feito

### 1. O mesmo treino do caso exato
O script de hardware chama a **mesma** função `train` de `Inverser_quantum_exact_pinn_E110.py`, com as
mesmas `CONFIGS`:
- dois otimizadores: AdamW na rede e varpro ou Adam em log ν;
- peso dos dados ≫ peso do resíduo (w_d fixo ou grad-norm);
- MAE.

O porquê de cada escolha está no [README do caso exato](../quantum_inverse_exact_E110/README.md).
A única diferença é o gancho `measure`, que mede o QPU a cada época. Sem o gancho, o script exato dá
resultados idênticos bit a bit aos de antes.

### 2. O QPU no laço, desde a primeira época
A maquinaria de hardware vem de `Inverser_quantum_hardware_pinn.py`, que não foi alterado.

1. **Calibração**: 1 job por número de qubits, compartilhado por todas as runs. Ângulos aleatórios
   rodam em todas as cadeias de n qubits do chip. As cadeias com nota de erro > 0,05·n são descartadas,
   e também as que destoam do polinômio médio do chip. O resultado é um mínimo-quadrados para a resposta
   medida, que é exatamente um polinômio trigonométrico de grau 1 em cada ângulo do cone causal.
   No FakeBerlin: 33 cadeias de 3 qubits, 14 de 7 e 9 de 10. O RMS do ajuste fica em 0,031–0,035,
   igual ao ruído de shot (1/32).
2. **Treino**: 110 jobs, um por época. Em cada época, o QPU mede ⟨Z⟩ nos ângulos atuais dos 10 pontos de
   colocação e dos 100 pontos de dados (110 pontos distribuídos entre as cadeias). O forward usa os
   valores do QPU, e o gradiente passa pelo polinômio (straight-through):
   `q(φ) = poly(φ) + stopgrad(q_QPU(φ_c) − poly(φ_c))`. A mesma correção vale para os 5 pontos do
   estêncil de cada ponto de colocação. A cada 10 épocas, o polinômio é reajustado com todas as medidas.
3. **Validação**: 1 job. u(x,T) em 100 pontos, calculado com a rede treinada rodando no QPU
   (uma réplica por cadeia), comparado com a solução exata.

Cada run usa **111 jobs**: 110 de treino e 1 de validação.

### 3. Por que o treino do caso exato funciona no hardware sem mudanças
- **Nenhum job extra por época.** O varpro calcula ν* com A e B do próprio forward, e o grad-norm usa os
  gradientes do polinômio.
- **O ruído de shot não é amplificado nas derivadas.** Os valores de u (em u·u_x e nos dados) vêm do
  QPU, mas as diferenças finitas vêm do polinômio. Medir os vizinhos do estêncil separadamente
  multiplicaria o ruído por hx⁻² ≈ 2,6·10⁵ em u_xx.
- **O varpro suaviza o ruído de ν.** O ν* de cada lote é ruidoso, porque A carrega o ruído de u. Mas a
  relaxação de log ν funciona como uma média móvel: com β final de 0,1 (`gradnorm_varpro`) ou 0,013
  (`fixed_varpro`), a média cobre ~10 ou ~80 épocas.

### O que muda em relação a `Inverser_quantum_hardware_pinn.py`
| | Original | E110 |
|---|---|---|
| ν verdadeiro | 0,01 | 0,1 |
| Fase no simulador | 500 épocas AdamW (lr 0,1) com o polinômio | nenhuma: o QPU entra desde a 1ª época |
| Fase com QPU no laço | 1000 épocas, AdamW único (lr 10⁻³) na rede e em ν | 110 épocas, dois otimizadores (configurações acima) |
| Calibração | 1 job por run | 1 job por número de qubits, reaproveitado |
| Reajuste do polinômio | a cada 50 épocas | a cada 10 épocas |

## A rede aprende o dispositivo
Com os mesmos pesos, u(x,T) calculado com o **circuito ideal** tem erro L2 de 2,7·10⁻² (mediana de
`gradnorm_varpro`), contra 1,1·10⁻² **no QPU**. A rede compensa a resposta ruidosa do chip. Por isso,
uma rede treinada num chip deve ser usada nele mesmo, ou recalibrada.

## O proxy do fake
Para ter 60 runs por configuração sem gastar ~12 min de Aer em cada uma, o polinômio calibrado no
FakeBerlin faz o papel do chip, e cada ⟨Z⟩ recebe ruído binomial de 1024 shots. É a mesma ideia de
`search_optimizer_ibm_faker.py`. O proxy é fiel ao fake porque o RMS do ajuste (0,031–0,035) é igual ao
ruído de shot: o polinômio reproduz o fake dentro do ruído. Ele roda a mesma `run_one` do script; só
`make_qpu` é trocada. O código está em `T_0.1/proxy_validation/hw_proxy.py`.

## Limitações
- O fake **não** reproduz crosstalk, deriva no tempo nem a mitigação do runtime: o fake roda os circuitos
  crus. Num QPU real, o erro deve ser algo maior.
- O caminho `--real` (`PackedCircuit` + `Session`, com a mitigação `MITIGATION` do script original)
  **não foi executado**.
- Tempo no fake, com 4 runs em paralelo: ~9–11 min por run (3 qubits), ~12 min (7) e ~21–24 min (10).

## Estrutura da pasta
```
quantum_inverse_hardware_E110/
├── README.md                          este arquivo
├── README_melhores_configuracoes.md   as 3 configurações no hardware
└── T_0.1/
    ├── fake_berlin/
    │   ├── calibration_NQ{3,7,10}_fake_berlin.npz   cadeias + medidas da calibração
    │   ├── gradnorm_varpro/   configuração padrão: os 9 runs da figura nu_summary_10_E110.png
    │   ├── fixed_varpro/
    │   └── fixed_adam/
    │         cada uma: model_*.pt (pesos, polinômio final, cadeias), run_*.json (resultado, ids dos jobs),
    │         losshist_*.npz, validation_*.npz (u no QPU / modelo medido / circuito ideal / exato),
    │         loss_*.png, nu_history_*.png, solution_*.png, nu_summary_10_E110.png, summary.json
    └── proxy_validation/
        ├── proxy_resumo.txt           as 60 runs por configuração (seeds 200–219)
        ├── proxy_results.json
        └── hw_proxy.py                o proxy (rodar da raiz do repositório)
```

## Como reproduzir
No ambiente conda `qiskit`, a partir da raiz do repositório:
```
python Inverser_quantum_hardware_pinn_E110.py                     # fake: 3 configs × 3/7/10 qubits × seeds 0/1/2 (~2 h)
python Inverser_quantum_hardware_pinn_E110.py --config gradnorm_varpro --qubits 10 --seeds 0
python Inverser_quantum_hardware_pinn_E110.py --real --config gradnorm_varpro --qubits 10 --seeds 0   # QPU real (não testado)
```
As calibrações já salvas são reaproveitadas. Apague os `.npz` de calibração para recalibrar.
