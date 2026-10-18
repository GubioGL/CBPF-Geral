# PINN clássica × PINN híbrida quântica para a equação de Burgers viscosa

Relatório de apoio à escrita do artigo: o que cada código faz, como os experimentos
foram montados e quais resultados estão em `data/`. Todos os números citados foram
lidos dos logs (`run_*.json`, `summary*.json`, `*_summary.csv`) ou recalculados a
partir dos checkpoints salvos (detalhes na seção 9).

Códigos cobertos:

| Script | Papel |
|---|---|
| `run_classical.py` | varredura da PINN clássica (largura × nº de pontos × seeds) |
| `run_quantum_ibm_faker.py` | PINN híbrida treinada no QPU simulado com ruído (FakeBerlin) e avaliada no QPU real |
| `run_quantum_hardware_real.py` | PINN híbrida treinada com o QPU real (`ibm_berlin`) no loop |
| `run_classical_vs_qpu.py` | PINN clássica treinada com o **mesmo protocolo** do run no QPU real |
| `plote_erro_epocas_classico.py` | figura 3D: erro × épocas × largura (caso clássico) |
| `plote_inverse_classico_vs_collocation.py` | figura do problema inverso: erro em ν × nº de pontos |

Os scripts quânticos importam a maquinaria de `run_quantum_hardware.py`
(modelo, perda, passo de otimização, circuito no fake) e de
`Inverser_quantum_hardware_pinn.py` (cadeias de qubits, polinômio de calibração).

---

## 1. Resumo dos resultados

1. **QPU real.** A PINN híbrida de 10 qubits treinada com o `ibm_berlin` no loop
   (110 épocas, 1 job por época, sem mitigação de erro) reduziu o MSE de
   u(x, 0,1) contra a FDM de 3,90×10⁻² para **1,45×10⁻³** (erro L2 relativo de
   29,4 % para **5,7 %**).
2. **Simulação com ruído × hardware.** A mesma rede treinada no FakeBerlin chegou
   a MSE 7,7×10⁻⁴ (L2 4,1 %). Executada no chip real (só predição), deu MSE
   **1,06×10⁻³** (L2 4,8 %). Assim, o modelo de ruído do simulador transfere para o
   hardware com perda pequena.
3. **Mesmo protocolo, rede clássica.** Com o mesmo orçamento (110 passos AdamW,
   10 pontos de colocação por época, mesmas seeds e mesmo cronograma de LR), a PINN
   clássica de mesma arquitetura externa teve MSE médio entre 1,0×10⁻² e 4,5×10⁻²
   (larguras 1–10, 5 seeds). Isso é **7 a 30 vezes pior** que o QPU real. O melhor
   run clássico individual (MSE 9,5×10⁻⁴) só empata com o QPU.
4. **Orçamento longo (20 000 AdamW + L-BFGS), protocolo idêntico para os dois
   modelos** (`CBPF_qubit_sweep_quantum_classical_fixed_validation_results`, 50 seeds,
   n = 2–15, mesma validação fixa e mesmo critério de parada).
   - Em T = 0,1 com 100 pontos, os dois chegam ao piso da FDM (MSE ≈ 7×10⁻⁶).
   - **Nenhum dos dois falha em T = 0,1**: MSE máximo de 2,2×10⁻³ na clássica e
     1,2×10⁻³ na quântica, em 1400 runs cada.
   - Com 10 pontos, a mediana da HQPINN exata é de 1,2 a 4,5 vezes menor, conforme a
     largura.
   - A diferença maior está na convergência. Com 10 qubits, a HQPINN atinge o critério
     físico em 78 % dos runs com 10 pontos (clássica: 56 %) e em 100 % com 100 pontos
     (clássica: 64 %). Ela também para em cerca de metade dos passos, com 271 parâmetros
     contra 491.
   - Em T = 1, 100 pontos e n = 10, a HQPINN tem MSE mediano de 4,8×10⁻⁵ (L2 3,3 %),
     contra 3,0×10⁻⁴ (8,2 %) da clássica.

   A "bimodalidade" clássica com 10 pontos (24 % a 58 % de runs presos em
   MSE ≈ 3,6×10⁻²) aparecia apenas na varredura antiga `data/Forward/classical`. Essa
   varredura usava outro monitor de parada e de melhor snapshot: a grade fixa
   `uniform_exact` de N pontos. Ela não é uma propriedade da arquitetura clássica e não
   deve ser comparada com os runs quânticos (seção 6.1).
5. **Problema inverso (estimar ν).** Em T = 0,1, a PINN clássica recupera ν com
   erro relativo mediano de ≈ 2 % (100 pontos), e a quântica simulada com ≈ 7 %.
   Em T = 1, ambas falham (erro > 60 %), embora a quântica erre menos.

---

## 2. O problema físico

Equação de Burgers viscosa 1D, problema direto (ν conhecido):

$$u_t + u\,u_x = \nu\,u_{xx},\qquad x\in[0,1)\ \text{periódico},\quad t\in[0,T]$$

- ν = 0,01; L = 1; T ∈ {0,1; 1}.
- Condição inicial: u(x,0) = sin(2πx).
- Em T = 0,1 a solução já está deformada (início da formação do choque em x = 0,5).
  Em T = 1 a onda está bem mais amortecida (‖u‖_rms ≈ 0,21) e o problema é mais difícil
  para as redes.

### Referência numérica (FDM)

`main.functions.solve_burgers_fd`: diferenças finitas explícitas, upwind na
advecção, central na difusão, Nx = 512, Δt = 10⁻⁴, periódico. A referência é usada
**só para avaliação**, nunca no treino (o problema direto não usa dados).

**Métricas** (em todos os scripts): u(x, T) avaliado em uma grade uniforme de x e
comparado com a FDM interpolada:

- MSE = média de (u − u_FDM)²;
- MAE;
- erro L2 relativo = ‖u − u_FDM‖ / ‖u_FDM‖.

Grade de avaliação: 200 pontos (scripts quânticos e clássico × QPU, e os helpers de
plot do clássico).

**Linha de base trivial** (útil para interpretar falhas): se a rede não aprende
nada, u = sin(2πx) e o MSE é **3,93×10⁻²** em T = 0,1 e **2,82×10⁻¹** em T = 1.

---

## 3. Modelos

Os dois modelos compartilham a mesma estrutura externa:

```
(x,t) → [sin 2πx, cos 2πx, t/T] → pré-rede → BLOCO INTERNO (largura n) → pós-rede → N(x,t)
u(x,t) = sin(2πx) + (t/T) · N(x,t)
```

- **Periodicidade** em x é exata, porque x só entra por sin e cos.
- **Condição inicial** é imposta por construção (*hard constraint*). Por isso não há
  termos de contorno nem de condição inicial na perda.
- Pré-rede: Linear(3→10) + tanh + Linear(10→n). Pós-rede: Linear(n→10) + tanh +
  Linear(10→1).

### PINN clássica — `main.models.ClassicalPINN2`

O bloco interno é um MLP: Linear(n→10) + tanh + Linear(10→n), onde n é o
`INTRA_NEURO`. A versão com ativação seno existe (`seno=True`), mas os dados em
`data/Forward/classical` usam tanh. Nº de parâmetros treináveis: **42n + 71**
(n = 10 → 491).

### PINN híbrida quântica — `QuantumExactPINN` / `QuantumHardwarePINN`

O bloco interno é um **circuito de n qubits sem parâmetros treináveis**:

1. codificação: RY(φ_k)|0⟩ em cada qubit, com os ângulos φ = saída da pré-rede;
2. uma camada fixa de CNOTs *brickwork*: (0,1),(2,3),…,(8,9) seguida de
   (1,2),(3,4),…,(7,8). São 9 CNOTs para 10 qubits;
3. saída: ⟨Z_i⟩ de cada qubit, que entra na pós-rede.

Para essa família de circuitos o valor ideal tem forma fechada
(`cnot_cosine_expectations`):

$$\langle Z_i\rangle = \prod_j \cos(\varphi_j)^{M_{ij}},$$

onde M é a matriz binária (GF(2)) da rede de CNOTs. A versão "exata" usa essa
fórmula (diferenciável, sem shots). As versões em hardware substituem a fórmula pelo
valor medido. Com n = 10 a rede tem **271 parâmetros**, todos clássicos (pré e pós).

Com a camada brickwork, cada ⟨Z_i⟩ depende de no máximo 4 ângulos (o cone causal).
Isso torna viável o modelo polinomial do hardware descrito na seção 4.2.

---

## 4. Perda e treinamento

### 4.1 Perda física (comum a todos)

A perda é o resíduo da EDP, calculado por **diferenças finitas sobre a própria rede**
(não por autograd). Em cada ponto de colocação (x, t), a rede é avaliada em
5 pontos: centro, x ± h_x e t ± h_t, com h_x = 1/512 e h_t = 10⁻³.

$$\mathcal{L}=\frac1N\sum\big(u_t+u\,u_x-\nu\,u_{xx}\big)^2,\quad
u_t=\tfrac{u(t+h_t)-u(t-h_t)}{2h_t},\ u_x=\tfrac{u(x+h_x)-u(x-h_x)}{2h_x},\ u_{xx}=\tfrac{u(x+h_x)-2u+u(x-h_x)}{h_x^2}$$

Os pontos de colocação são sorteados uniformemente em x ∈ [0,1) e
t ∈ [h_t, T − h_t], e **renovados a cada época**.

### 4.2 Treino com o QPU no loop (`run_quantum_hardware*.py`, `run_quantum_ibm_faker.py`)

O protocolo tem três fases:

1. **Calibração (1 job).** Ângulos aleatórios são enviados a todas as cadeias de
   10 qubits do chip: 128 conjuntos de parâmetros × 8 cadeias = 1024 amostras. Para
   cada saída i, o ⟨Z_i⟩ medido é ajustado por mínimos quadrados a um polinômio
   trigonométrico de grau 1 em cada ângulo do cone causal (3⁴ = 81 coeficientes).
   Esse polinômio (`HardwareSurrogateLayer`) é um modelo clássico do chip, com o
   ruído médio do dispositivo embutido. Esta fase não treina a rede.
2. **Treino (1 job por época, 110 épocas).** Os 10 pontos de colocação da época são
   medidos no QPU, distribuídos pelas 8 cadeias em paralelo (2 conjuntos de
   parâmetros por job). O passo usa um estimador *straight-through*:
   $$q(\varphi)=\mathrm{poly}(\varphi)+\mathrm{stopgrad}\big(q_{\text{QPU}}(\varphi_c)-\mathrm{poly}(\varphi_c)\big)$$
   - o *forward* usa o valor medido no QPU;
   - o gradiente passa pelo polinômio;
   - a mesma correção é aplicada aos 5 pontos do estêncil.

   Se os vizinhos do estêncil fossem medidos no QPU, o ruído de shot em u_xx seria
   amplificado por h_x⁻² ≈ 2,6×10⁵. O polinômio é reajustado a cada 10 épocas com
   todas as medidas acumuladas.
3. **Validação (1 job).** u(x, T) é calculado nos 200 pontos com a rede treinada
   rodando no QPU, com 8 réplicas por ponto (uma em cada cadeia), e comparado com a
   FDM.

Configuração comum aos runs salvos:

| Parâmetro | Valor |
|---|---|
| Qubits / cadeias | 10 qubits × 8 cadeias em paralelo (80 qubits físicos) |
| Shots | 1024 por circuito (precisão ≈ 1/32 por ⟨Z⟩) |
| Mitigação de erro | nenhuma (contagens brutas) |
| Otimizador | AdamW, LR 0,1 → 0,01 (−0,01 a cada 10 épocas), weight decay 10⁻⁶, clip 1,0 |
| Seed | 0 (a melhor de 5 seeds no lote rodado no fake) |
| Circuito transpilado | profundidade 12, 72 CZ (9 por cadeia), sem SWAPs |

### 4.3 Treino clássico longo (`run_classical.py` → `main.functions.train`)

1. AdamW (LR 10⁻³, weight decay 10⁻⁶, clip 1) com `ReduceLROnPlateau`, até
   20 000 épocas, com novos pontos aleatórios a cada época.
2. Refinamento L-BFGS (LR 0,5), até 1000 iterações, com parada por paciência
   (50 iterações sem melhora).
3. **Critério de parada físico.** O treino para quando a perda, monitorada em uma
   grade uniforme fixa de N pontos, cai abaixo de `target_loss`. Esse alvo é o
   resíduo da própria solução FDM nos mesmos pontos e com o mesmo estêncil
   (`reference_pde_residual`). Abaixo desse valor a rede estaria "ajustando o erro
   da referência". O teto é 20 000 + 1000 passos.

---

## 5. O que cada script faz

### `run_classical.py`

Varre, em paralelo (`ProcessPoolExecutor`), a PINN clássica nas combinações de:

- N_col ∈ {10, …, 100};
- INTRA_NEURO ∈ {1, …, 10, 15};
- T ∈ {0,1; 1};
- seeds.

Bloco A: N_col completo para larguras {3, 5, 7, 10, 15}. Bloco B: N_col ∈ {10, 100}
para as demais larguras. Pula jobs cujo `.pt` já existe, o que permite retomar a
varredura.

Saídas por run:
- `Cmodel_*.pt` + `_config.json`: pesos, hiperparâmetros, `TARGET_LOSS`,
  `TARGET_REACHED` e `STOPPED_AT_STEP`;
- `losshist_*.npz`: histórico completo da perda;
- figuras de loss e solução.

### `run_quantum_hardware_real.py`

Treino completo no QPU real. Fluxo:

1. lê o lote fake de referência e escolhe a melhor seed;
2. escolhe as cadeias de 10 qubits conectados com a calibração **atual** do chip;
3. descarta as cadeias com nota de erro > 0,5 (eram 12 candidatas, ficaram 8);
4. transpila tudo em **um circuito empacotado**;
5. roda calibração, treino e validação dentro de **uma Session** IBM;
6. salva um checkpoint a cada época, o que permite `--resume`.

Opções: `--plan` (prepara e mostra o plano, sem enviar jobs) e `--fake`
(ensaio local completo).

Saídas em `data/quantum_forward_hardware_real/T_0.1/`, todas com a tag
`10_NQ10_0_ibm_berlin_none_HW110`:

| Arquivo | Conteúdo |
|---|---|
| `run_*.json` | log: job ids, métricas, tempos, versões |
| `plan_*.json` | plano da execução |
| `backend_*.json` | T1, T2 e erros dos qubits e acopladores usados |
| `isa_*.qpy` | circuito enviado ao chip |
| `calibration_*.npz` | dados do job de calibração |
| `history_*.npz` | por época: LR, perdas, pontos, ângulos, ⟨Z⟩ medidos, job ids |
| `validation_*.npz` | u_qpu, u_surrogate, u_ideal, u_reference |
| `validation_*.png`, `before_after_*.png`, `loss_*.png` | figuras |
| `output.txt` | saída do terminal |
| `resumo_apresentacao.txt` | resumo já redigido |

### `run_quantum_ibm_faker.py`

Fluxo:

1. treina a mesma rede (mesma seed, mesma sequência de pontos e **as mesmas cadeias
   físicas** do run real, lidas do log) no **FakeBerlin**, simulado no Aer com matriz
   densidade. O modelo de ruído inclui erros de porta, T1/T2 e leitura, com
   1024 shots;
2. valida no fake;
3. (fase 4) envia **um único job de predição** ao `ibm_berlin` real com os pesos
   treinados no fake.

Isso permite separar o efeito de **treinar** no hardware do efeito de **avaliar** no
hardware. Saídas em `data/quantum_ibm_faker/T_0.1/`, com a tag
`10_NQ10_0_fake_berlin_HW110`. O arquivo `prediction_real_*.npz` contém u_real,
u_fake, u_surrogate, u_ideal e q_real.

> **Atenção:** o código atual difere do run salvo (ver seção 8).

### `run_classical_vs_qpu.py`

Controle clássico com **protocolo idêntico ao run no QPU real**. Tudo é lido do
`run_*.json` real:

- 110 épocas e 10 pontos por época;
- mesma sequência de pontos (gerador com a mesma seed);
- mesmo cronograma de LR, weight decay e clip;
- mesma pré e pós-rede.

Só o bloco interno é trocado por um MLP tanh de largura n = 1…10, que faz o papel do
nº de qubits. Seeds 0–4. Gera o `summary_10_NEURO110_tanh_seeds0-4_E110.json`, com
estatísticas por largura e a linha do QPU real.

### `plote_erro_epocas_classico.py`

Para T ∈ {0,1; 1} e N_col ∈ {10, 100}, gera a dispersão 3D de cada run clássico:

- eixo x: INTRA_NEURO; eixo y: passos até parar; eixo z: log₁₀ MSE;
- verde: atingiu o critério de parada; vermelho: esgotou o orçamento;
- um plano marca o teto de épocas.

Saída: `ErroEpocas_3D_T_<T>_NCOL<N>.png`. Usa
`compute_classical_error_and_stopping_per_run`.

### `plote_inverse_classico_vs_collocation.py`

Problema inverso (ν desconhecido, aprendido junto com u). Plota o erro quadrático de
ν, (ν_aprendido − 0,01)², em função de N_col, com média e barra de variância sobre as
seeds:

- curvas clássicas coloridas para larguras 3, 7 e 15;
- curvas quânticas em preto tracejado para 3, 7 e 15 qubits.

Saída: `NuError_vs_N_COLLOCATION_T_<T>_NEURO3-7-15.png`.

---

## 6. Resultados

### 6.1 PINN clássica × HQPINN exata — varredura longa com protocolo comum

Fonte: `data/Forward/CBPF_qubit_sweep_quantum_classical_fixed_validation_results/qubit_sweep_runs.csv`
(protocolo `fixed_validation_v2`, 5600 runs). Os dois modelos são treinados com o mesmo
protocolo:

- otimização: 20 000 AdamW (LR 10⁻³ com `ReduceLROnPlateau`) e depois até 1000 L-BFGS;
- pontos de colocação: N_col novos pontos aleatórios a cada passo;
- monitor: residual em um **conjunto de validação fixo** de N_col pontos aleatórios
  (seed 10⁶ + seed); a parada ocorre quando o residual atinge o da FDM nesse conjunto, e
  o modelo retornado é o melhor snapshot;
- amostragem: n = 2–15, N_col ∈ {10, 100}, T ∈ {0,1; 1}, 50 seeds.

O bloco do meio clássico é n→n→n (2n² + 23n + 61 parâmetros; 491 para n = 10). A HQPINN
tem 21n + 61 parâmetros (271 para n = 10).

**MSE de u(x,T) contra a FDM: mediana (média)**, 200 pontos. Ao lado, a fração de runs que
atingiu o critério de parada físico.

**T = 0,1**

| N_col | n | clássica | alvo | HQPINN exata | alvo |
|---|---|---|---|---|---|
| 10 | 3 | 5,8e-5 (1,8e-4) | 4 % | 2,1e-5 (4,7e-5) | 22 % |
| 10 | 7 | 3,5e-5 (8,2e-5) | 34 % | 1,5e-5 (1,8e-5) | 78 % |
| 10 | 10 | 1,9e-5 (3,0e-5) | 56 % | 1,3e-5 (1,5e-5) | 78 % |
| 10 | 15 | 1,6e-5 (3,0e-5) | 80 % | 1,1e-5 (1,3e-5) | 88 % |
| 100 | 3 | 8,9e-6 (1,1e-5) | 2 % | 7,1e-6 (7,1e-6) | 50 % |
| 100 | 7 | 6,9e-6 (7,6e-6) | 34 % | 7,2e-6 (7,2e-6) | 100 % |
| 100 | 10 | 6,7e-6 (7,2e-6) | 64 % | 6,9e-6 (6,8e-6) | 100 % |
| 100 | 15 | 6,5e-6 (7,0e-6) | 90 % | 6,9e-6 (7,0e-6) | 100 % |

- **Nenhuma falha em T = 0,1**, em nenhum dos dois modelos: em 1400 runs de cada, o MSE
  máximo é de 2,2×10⁻³ na clássica e de 1,2×10⁻³ na quântica.
- Com N_col = 100 os dois saturam em ≈ 7×10⁻⁶ (L2 ≈ 0,4 %), que é o limite da referência
  FDM.
- Com N_col = 10, a mediana da HQPINN é menor em todas as larguras, por um fator de 1,2 a
  4,5. A média fica próxima da mediana nos dois modelos.
- **Convergência.** Com N_col = 100 e n ≥ 5, a HQPINN atinge o critério físico em 98 % dos
  runs, contra 55 % da clássica. Com n = 10, o run mediano para em 7,1×10³ passos, contra
  1,6×10⁴ da clássica.

**T = 1**

| N_col | n | clássica | alvo | HQPINN exata | alvo |
|---|---|---|---|---|---|
| 10 | 3 | 6,0e-3 (1,9e-2) | 0 % | 3,2e-3 (8,5e-3) | 0 % |
| 10 | 7 | 3,8e-3 (7,8e-3) | 2 % | 1,7e-3 (2,6e-3) | 8 % |
| 10 | 10 | 2,7e-3 (4,9e-3) | 6 % | 1,3e-3 (2,2e-3) | 18 % |
| 10 | 15 | 2,8e-3 (5,4e-3) | 8 % | 6,0e-4 (1,4e-3) | 20 % |
| 100 | 3 | 1,9e-3 (3,0e-3) | 0 % | 8,1e-4 (1,4e-3) | 0 % |
| 100 | 7 | 3,7e-4 (9,2e-4) | 16 % | 8,9e-5 (1,6e-4) | 66 % |
| 100 | 10 | 3,0e-4 (9,7e-4) | 34 % | 4,8e-5 (1,1e-4) | 92 % |
| 100 | 15 | 6,7e-4 (1,3e-3) | 40 % | 5,6e-5 (9,7e-5) | 94 % |

- A mediana da HQPINN é menor em todas as larguras e nos dois N_col. Com N_col = 100 o
  fator vai de 1,25 a 14. Com n = 10 o L2 relativo é de 3,3 % na HQPINN e de 8,2 % na
  clássica.
- Com N_col = 10, nenhum dos dois modelos é bom, mas nenhum fica parado na condição
  inicial. Com n = 10, o L2 relativo mediano é de 17 % na HQPINN e de 25 % na clássica.
- A clássica melhora de forma não monotônica com n. A melhor mediana dela (1,1×10⁻⁴ com
  n = 9 e 430 parâmetros) fica acima da HQPINN para todo n ≥ 6 (a partir de 187
  parâmetros).

#### 6.1.1 Varredura clássica antiga (`data/Forward/classical`) — protocolo diferente

Esta varredura foi feita com uma versão anterior do protocolo. O monitor de parada e de
melhor snapshot era a grade fixa `uniform_exact` de N_col pontos, e não um conjunto de
validação aleatório. Por isso ela **não é comparável** com os runs quânticos. As
"falhas" com N_col = 10 (runs presos em MSE ≈ 3,6×10⁻², perto da linha de base) aparecem
só aqui: no protocolo comum (tabela acima) não há nenhuma. O mesmo vale para T = 1 com
N_col = 10. Aqui o L2 passa de 100 %; no protocolo comum, a mediana com n = 10 é de 25 %.
A tabela abaixo continua útil para a dependência em N_col dentro deste protocolo.

**MSE de u(x,T) contra a FDM: mediana (média)**, 200 pontos de avaliação.

**T = 0,1**

| N_col | n = 3 | n = 7 | n = 10 | n = 15 |
|---|---|---|---|---|
| 10 | 2,7e-4 (1,7e-2) | 1,3e-4 (1,3e-2) | 8,0e-5 (1,3e-2) | 3,7e-5 (8,9e-3) |
| 20 | 1,3e-5 (3,0e-5) | 9,2e-6 (1,8e-5) | 8,4e-6 (1,7e-5) | 9,8e-6 (2,1e-5) |
| 30 | 9,3e-6 (1,9e-5) | 8,0e-6 (9,5e-6) | 7,6e-6 (7,8e-6) | 7,6e-6 (8,0e-6) |
| 50 | 8,4e-6 (3,2e-5) | 7,1e-6 (1,5e-5) | 6,5e-6 (9,4e-6) | 7,1e-6 (1,3e-5) |
| 100 | 6,9e-6 (1,1e-5) | 7,3e-6 (7,7e-6) | 6,3e-6 (7,0e-6) | 6,9e-6 (7,6e-6) |

- Com N_col ≥ 20 o erro **satura em ≈ 7×10⁻⁶** (L2 relativo ≈ 0,4 %),
  qualquer que seja a largura (n ≥ 3).
- Com **N_col = 10** e este protocolo, a distribuição é bimodal. A fração de runs com
  MSE > 10⁻² é a seguinte:

  | n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 15 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | MSE > 10⁻² | 58 % | 54 % | 46 % | 38 % | 54 % | 34 % | 36 % | 30 % | 30 % | 36 % | 24 % |

  No protocolo comum (6.1) a fração é **0 %** para todo n. Isso indica que a bimodalidade
  vem do monitor da grade `uniform_exact`, que escolhe o melhor snapshot em 10 pontos fixos
  pouco informativos, e não da arquitetura clássica.
- Fração de runs que atingiram o critério de parada físico com N_col = 100:
  2 % (n = 3), 40 % (n = 7), 50 % (n = 10) e 68 % (n = 15). Os demais esgotaram o
  orçamento. A largura acelera a convergência, mas não muda o erro final.

**T = 1**

| N_col | n = 3 | n = 7 | n = 10 | n = 15 |
|---|---|---|---|---|
| 10 | 1,3e-1 (1,4e-1) | 1,4e-1 (1,4e-1) | 1,4e-1 (1,4e-1) | 1,4e-1 (1,4e-1) |
| 20 | 2,0e-3 (3,7e-3) | 1,4e-3 (2,0e-3) | 9,6e-4 (1,4e-3) | 1,2e-3 (1,3e-3) |
| 30 | 1,6e-3 (2,9e-3) | 1,1e-3 (1,2e-3) | 4,3e-4 (9,5e-4) | 7,8e-4 (2,0e-3) |
| 50 | 9,8e-4 (2,5e-3) | 5,2e-4 (1,2e-3) | 2,7e-4 (6,7e-4) | 7,1e-4 (8,1e-4) |
| 100 | 5,7e-4 (1,7e-3) | 2,4e-4 (4,9e-4) | 2,2e-4 (4,3e-4) | 4,2e-4 (5,1e-4) |

- Com N_col = 10 todos os runs deste protocolo falham (L2 relativo > 100 %). No protocolo
  comum, a mediana com n = 10 é de 2,7×10⁻³ (L2 25 %).
- Nenhum run em T = 1 atingiu o critério de parada neste protocolo: 0 % em todas as
  combinações. No protocolo comum, com N_col = 100 e n = 10, foram 34 %.
- O erro cai com N_col e fica em ~2×10⁻⁴ (mediana, n ≥ 7, N_col ≥ 60).

### 6.2 HQPINN exata × número de pontos de colocação (simulação sem ruído)

Fonte: `data/Forward/CBPF_quantum_exact_results_*/quantum_exact_summary.csv` (pastas
`Brickwork3`, `_Brickwork5`, `fixed_BrickWork7/10/15`; 50 seeds cada). A rede é brickwork
com 1 camada e o orçamento é o mesmo da seção 6.1 (20 000 AdamW + L-BFGS).

As pastas usam o mesmo protocolo `fixed_validation_v2` da varredura comum. Para 10 qubits,
os passos de parada coincidem run a run com os da seção 6.1, e o MSE difere em menos de
1 %.

Valores: **MSE médio**.

| N_col | T=0,1 · 3q | 7q | 10q | 15q | T=1 · 3q | 7q | 10q | 15q |
|---|---|---|---|---|---|---|---|---|
| 10 | 4,7e-5 | 1,8e-5 | 1,5e-5 | 1,3e-5 | 8,5e-3 | 2,6e-3 | 2,2e-3 | 1,4e-3 |
| 20 | 1,3e-5 | 1,0e-5 | 1,0e-5 | 1,0e-5 | 3,8e-3 | 2,2e-3 | 1,3e-3 | 5,4e-4 |
| 30 | 9,4e-6 | 8,1e-6 | 8,4e-6 | 8,4e-6 | 3,8e-3 | 7,2e-4 | 3,7e-4 | 4,2e-4 |
| 50 | 8,9e-6 | 7,8e-6 | 8,1e-6 | 8,0e-6 | 1,8e-3 | 3,2e-4 | 1,8e-4 | 1,5e-4 |
| 100 | 7,1e-6 | 7,2e-6 | 6,7e-6 | 7,0e-6 | 1,4e-3 | 1,5e-4 | 1,1e-4 | 9,7e-5 |

Leitura para o artigo:

- **Poucos pontos.** Com N_col = 10, T = 0,1 e 10 qubits (271 parâmetros), a HQPINN teve
  MSE médio de 1,5×10⁻⁵ e máximo de 5,0×10⁻⁵ em 50 seeds. A clássica de largura 10
  (491 parâmetros), **no mesmo protocolo**, teve média de 3,0×10⁻⁵ e máximo de 2,3×10⁻⁴,
  também sem falhas. A vantagem existe, mas é moderada: cerca de 2× na média e uma cauda
  mais curta. Não é uma diferença entre "aprende" e "não aprende".
- **Convergência.** Com N_col = 100 e T = 1, a HQPINN de 10 qubits atinge o critério de
  parada em 92 % dos runs, com MSE médio de 1,1×10⁻⁴. A clássica de mesma largura, no
  mesmo protocolo, atinge em 34 %, com MSE médio de 9,7×10⁻⁴ (mediana 3,0×10⁻⁴).
- **Saturação.** Com N_col ≥ 30 e T = 0,1, a HQPINN chega ao patamar de ≈ 7×10⁻⁶ para
  qualquer número de qubits. A clássica chega ao mesmo patamar com N_col = 100. É o limite
  imposto pela própria referência FDM e pelo estêncil.
- Em T = 1 o erro da HQPINN cai até N_col ≈ 100 e satura em ≈ 5×10⁻⁵ para n ≥ 7. Com
  3 qubits o patamar é cerca de 10 vezes maior, o que indica que o limite em horizonte
  longo é a largura, e não o número de pontos.

### 6.3 Simulação com ruído (FakeBerlin) — `data/quantum_ibm_faker/T_0.1/`

Seed 0, 10 qubits, N_col = 10, T = 0,1, 110 épocas, mesmas 8 cadeias físicas do run
real. Fonte: `run_10_NQ10_0_fake_berlin_HW110.json`.

| u(x, 0,1) vs FDM | MSE | MAE | L2 rel. |
|---|---|---|---|
| Rede inicial (polinômio medido) | 3,90e-2 | 0,176 | 29,3 % |
| Treinada no fake, validada no fake | 7,73e-4 | 0,0238 | 4,13 % |
| Treinada no fake, polinômio (sem job) | 7,74e-4 | 0,0239 | 4,13 % |
| Treinada no fake, circuito ideal | 4,58e-4 | 0,0179 | 3,18 % |
| **Treinada no fake, predição no `ibm_berlin` real** | **1,06e-3** | 0,0281 | **4,83 %** |

- **Calibração no fake:** RMS do resíduo do ajuste = 0,028 (ruído de shot esperado:
  0,031). Contraste do termo ideal entre 0,93 e 0,96. RMS fake − fórmula ideal = 0,039.
- **Na predição real:** RMS de ⟨Z⟩ real − polinômio do fake = 0,027. RMS real −
  fórmula ideal = 0,048.
- **Tempo:** treino de 383 s (3,4 s/época). O job de predição no chip real levou
  838 s, incluindo fila.

### 6.4 QPU real (`ibm_berlin`) — `data/quantum_forward_hardware_real/T_0.1/`

Fonte: `run_10_NQ10_0_ibm_berlin_none_HW110.json` (execução de 08/10/2026,
instância PDE_Europe, Session única).

| u(x, 0,1) vs FDM | MSE | MAE | L2 rel. |
|---|---|---|---|
| Rede inicial (polinômio medido) | 3,90e-2 | 0,176 | 29,4 % |
| **Treinada no QPU, validada no QPU** | **1,45e-3** | 0,0315 | **5,67 %** |
| Treinada no QPU, polinômio medido | 1,35e-3 | 0,0303 | 5,47 % |
| Treinada no QPU, circuito ideal | 1,64e-3 | 0,0341 | 6,01 % |
| Fake, mesma seed (lote original) | 8,06e-4 | — | 4,22 % |
| Fake, seeds 0–4 (lote original) | 8,1e-4 a 2,9e-3 | — | — |

**Hardware:**

- 80 de 120 qubits usados (8 cadeias × 10).
- CZ nativa; profundidade ISA 12; 72 CZ.
- Mediana nos qubits usados: T1 = 322 µs, T2 = 255 µs, erro de leitura 1,7 %,
  erro da CZ 1,7×10⁻³. Fonte: `backend_*.json`, resumido em
  `resumo_apresentacao.txt`.

**Calibração:**

- RMS do ajuste = 0,039 (shot ≈ 0,031).
- Contraste entre 0,86 e 0,96.
- RMS hardware − fórmula ideal = 0,064. É o ruído do chip real, maior que o do fake
  (0,039).

**Custo:**

- 112 jobs: 1 de calibração, 110 de treino e 1 de validação;
- 548 conjuntos de parâmetros e 561 152 shots;
- tempo total de 54 min: calibração 11 min, treino 29 min (≈ 16 s/época).

**Pontos para a discussão:**

1. O treino no hardware reduziu o MSE em cerca de 27 vezes. O resultado fica dentro
   da faixa das 5 seeds do simulador ruidoso, mas acima da mesma seed no fake, o que
   é compatível com *drift* e *crosstalk*, que o Aer não modela.
2. Os pesos treinados no chip funcionam no circuito ideal (L2 de 6,0 %), então a rede
   não ficou presa ao ruído específico do dispositivo.
3. O RMS entre os ⟨Z⟩ medidos na validação e o polinômio foi de apenas 0,013. O
   modelo do dispositivo acompanhou o hardware durante todo o treino.
4. Treinar no fake e avaliar no real (MSE 1,06×10⁻³) foi ligeiramente melhor que
   treinar no real (1,45×10⁻³). Com uma seed só, as duas estão dentro da
   variabilidade entre seeds; não se deve concluir que uma estratégia é superior.

### 6.5 Mesmo protocolo: clássica × QPU real (`data/Forward/classical_vs_qpu/T_0.1/`)

Configuração do run real: 110 épocas AdamW, 10 pontos por época, T = 0,1, seeds 0–4.

| Bloco interno | Parâm. | MSE média ± desvio | MSE mediana | MSE mín. | L2 rel. mediano |
|---|---|---|---|---|---|
| MLP n = 1 | 113 | 1,0e-2 ± 5,6e-3 | 8,8e-3 | 4,3e-3 | 13,9 % |
| MLP n = 2 | 155 | 2,6e-2 ± 2,1e-2 | 2,2e-2 | 2,0e-3 | 22,1 % |
| MLP n = 3 | 197 | 1,1e-2 ± 1,1e-2 | 8,5e-3 | 9,5e-4 | 13,7 % |
| MLP n = 4 | 239 | 1,7e-2 ± 1,6e-2 | 5,4e-3 | 1,9e-3 | 10,9 % |
| MLP n = 5 | 281 | 2,4e-2 ± 1,7e-2 | 2,7e-2 | 4,4e-3 | 24,5 % |
| MLP n = 6 | 323 | 2,0e-2 ± 1,5e-2 | 1,5e-2 | 3,2e-3 | 18,3 % |
| MLP n = 7 | 365 | 2,5e-2 ± 1,4e-2 | 3,4e-2 | 2,8e-3 | 27,4 % |
| MLP n = 8 | 407 | 3,0e-2 ± 1,4e-2 | 3,8e-2 | 3,4e-3 | 28,8 % |
| MLP n = 9 | 449 | 3,9e-2 ± 5,4e-3 | 4,1e-2 | 2,8e-2 | 30,1 % |
| MLP n = 10 | 491 | 4,5e-2 ± 4,1e-3 | 4,4e-2 | 4,1e-2 | 31,1 % |
| **Circuito 10 qubits, QPU real** | **271** | **1,45e-3** (1 seed) | — | — | **5,7 %** |

- Com esse orçamento curto, as redes clássicas mais largas (n ≥ 9) terminam **pior
  que a linha de base** de não evoluir a condição inicial (3,9×10⁻²).
- A curva de perda clássica fica entre 10⁰ e 10¹, enquanto as quânticas (fake e real)
  descem até ~10⁻¹ (Figura 1b).
- Uma leitura possível: o bloco quântico sem parâmetros (produto de cossenos) é um
  *feature map* fixo e bem condicionado. Com LR alto (0,1) e poucos passos, ele
  estabiliza a otimização, enquanto o MLP interno precisa aprender sua própria
  representação.

### 6.6 Figura 1 do artigo (`FIGURA_1_0_E110.png/.pdf`, gerada por `plot_FIGURA_1.py`)

- **(a)** u(x, 0,1): FDM (preto), PINN clássica de 10 neurônios (azul tracejado),
  HPINN treinada no FakeBerlin com predição no `ibm_berlin` (pontos verdes) e HPINN
  treinada inteiramente no `ibm_berlin` (estrelas vermelhas). As duas curvas
  quânticas seguem a frente íngreme em x ≈ 0,5. A clássica fica defasada, com pico e
  vale deslocados.
- **(b)** Perda de treino por época: a clássica oscila entre 3 e 40. As quânticas
  caem de ~10² para ~0,1–0,3.

Todas as curvas usam seed 0 e 110 épocas.

### 6.7 Problema inverso — ν desconhecido (`data/Inverso/`)

Dados: `classical_inverse` (50 seeds) e `quantum_inverse` (10 seeds; `QuantumPINN`
simulada no Qiskit, rede `linear_cnot`, RY, escala de ângulo π). Configuração:

- ν verdadeiro = 0,01; ν inicial ≈ 0,549;
- 100 pontos de dados;
- 20 000 AdamW + 1000 L-BFGS.

Valores: média de (ν − 0,01)² / erro relativo mediano.

| T | N_col | clássica n=3 | clássica n=7 | clássica n=15 | quântica 3q | quântica 7q |
|---|---|---|---|---|---|---|
| 0,1 | 10 | 1,7e-4 / 20 % | 9,1e-6 / 22 % | 8,4e-6 / 19 % | 2,2e-5 / 27 % | 5,2e-6 / 11 % |
| 0,1 | 100 | 6,3e-8 / 1,8 % | 9,7e-8 / 1,9 % | 7,7e-8 / 1,5 % | 5,9e-7 / 7,2 % | 5,2e-7 / 7,0 % |
| 1 | 10 | 6,6e-4 / 241 % | 7,8e-4 / 227 % | 9,9e-4 / 279 % | 9,3e-4 / 271 % | 6,1e-4 / 241 % |
| 1 | 100 | 3,9e-4 / 107 % | 3,3e-4 / 112 % | 5,1e-4 / 126 % | 1,7e-4 / 114 % | 4,7e-5 / 64 % |

- Em T = 0,1 a clássica recupera ν melhor quando há pontos suficientes.
- Em T = 1 nenhuma recupera ν; a quântica de 7 qubits é a menos ruim.
- Não há dados quânticos de 15 qubits (ver seção 8).

Há também uma comparação curta, com 110 épocas, em `data/Inverso_110/` (clássico ×
FakeBerlin, 3 seeds). Ela não é coberta pelos scripts deste relatório.

---

## 7. Sugestão de estrutura para a seção de Resultados

1. **Comparação com protocolo comum** (6.1): os dois modelos saturam em ~7×10⁻⁶ em
   T = 0,1, sem falhas. A HQPINN converge com mais frequência, em cerca de metade dos
   passos e com cerca de metade dos parâmetros. Em T = 1 ela é até 14× mais precisa.
2. **HQPINN exata × N_col** (6.2): atinge o piso da FDM com N_col ≥ 30. Em T = 1 o limite
   é a largura, e não o número de pontos.
3. **Do simulador ruidoso ao hardware** (6.3–6.4): MSE de 7,7×10⁻⁴ no fake,
   1,06×10⁻³ no real com pesos do fake e 1,45×10⁻³ treinando no real. O estimador
   *straight-through* com polinômio calibrado viabiliza o gradiente no hardware.
4. **Comparação com orçamento igual** (6.5 + Figura 1): a principal evidência
   experimental do artigo.
5. (Opcional) **Problema inverso** (6.7) como extensão e limitação.

---

## 8. Limitações e inconsistências a conferir antes de submeter

1. **`run_quantum_ibm_faker.py` mudou depois do run salvo.**
   - Constantes atuais: T = 1, N_col = 20, LR com decaimento exponencial, β = (0,76;
     0,99), clip 2, média EMA dos pesos e 100 pontos de avaliação.
   - Configuração do log salvo: T = 0,1, N_col = 10, LR em degraus 0,1 → 0,01, clip 1,
     sem EMA e 200 pontos.
   - A fase 4 (predição no QPU real) está **comentada** no código atual, mas foi
     executada no run salvo (job `db4i61h2p4ds73bjt9n0`).
   - Para o artigo, use o JSON como fonte da configuração.
2. **`run_classical.py` atual** grava em `data/classical_seno/`, com `seno=True` e
   seeds 0–9. Os dados em `data/Forward/classical/` foram gerados por uma versão
   anterior: tanh e seeds 0–49. `data/Forward/classical2/` é outra varredura
   (LR 10⁻², 10 seeds).
3. **Caminhos desatualizados após a reorganização de `data/`.**

   | Script | Caminho no código | Caminho atual |
   |---|---|---|
   | `plote_erro_epocas_classico.py` | `data/classical` | `data/Forward/classical` |
   | `plote_inverse_classico_vs_collocation.py` | `data/classical_inverse` | `data/Inverso/classical_inverse` |
   | `plote_inverse_classico_vs_collocation.py` | `data/quantum_inverse_exact` | `data/Inverso/quantum_inverse` |
   | `run_classical_vs_qpu.py` | `data/classical_vs_qpu` | `data/Forward/classical_vs_qpu` |

   O lote fake de referência, `data/quantum_forward_hardware/`, não está mais em
   `data/`.
4. **Inverso quântico:** os checkpoints em `data/Inverso/quantum_inverse` são de
   `QuantumPINN` (simulação Qiskit, `linear_cnot`), não da versão exata/brickwork do
   problema direto. Além disso, `N_QUBITS_LIST = [3, 7, 15]` pede 15 qubits, que não
   existem nesses dados (só 1–10).
5. **Hiperparâmetros do sweep quântico exato** (notebook) não foram conferidos contra
   os do clássico (por exemplo, o LR). A comparação em 6.2 assume o mesmo orçamento
   de passos (teto de 20 050 nos dois). As pastas `fixed_*` (7, 10 e 15 qubits) e as
   sem prefixo (3 e 5 qubits) podem diferir no critério de parada.
6. **Estatística do hardware:** QPU real e predição real têm **uma seed cada**.
   Barras de erro só existem para o fake (5 seeds) e para o clássico.
7. **Perda monitorada.** Nos runs quânticos, a perda do resíduo na grade fixa de
   10 pontos termina **maior** do que no início (real: 0,041 → 0,356; fake:
   0,046 → 0,090), enquanto o MSE contra a FDM cai 27×. Com 10 pontos, o resíduo é
   um indicador fraco de convergência. Reporte o MSE contra a FDM como métrica
   principal.
8. **Mitigação de erro** desligada em todos os jobs. O resultado do hardware é um
   limite inferior do que a técnica pode atingir.

---

## 9. Como reproduzir

Ambiente: conda `qiskit` (Python 3.10, Qiskit 2.5.2, qiskit-ibm-runtime 0.50.0,
qiskit-aer 0.17.2, PyTorch 2.14.1 CPU). Conta IBM salva em `~/.qiskit`.

```bash
python run_classical.py                       # varredura clássica (longa, paralela)
python run_quantum_hardware_real.py --plan    # prepara o run no QPU real sem enviar jobs
python run_quantum_hardware_real.py           # QPU real (~1 h de Session)
python run_quantum_ibm_faker.py --no-real     # só o fake, sem jobs na IBM
python run_classical_vs_qpu.py                # controle clássico com o protocolo do QPU
python plote_erro_epocas_classico.py          # ajuste CLASSICAL_DIR antes (seção 8.3)
python plote_inverse_classico_vs_collocation.py  # ajuste os diretórios antes (seção 8.3)
python plot_FIGURA_1.py                       # Figura 1
```

As tabelas 6.1 foram obtidas carregando os 6 200 checkpoints de
`data/Forward/classical` e avaliando u(x,T) em 200 pontos contra a FDM, como em
`compute_classical_error_and_stopping_per_run`. As tabelas 6.2–6.7 vêm diretamente
dos CSV/JSON citados em cada seção.
