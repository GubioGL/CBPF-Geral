# Manuscrito: HQPINN para os problemas direto e inverso

Rascunho no formato APS (revtex4-2, PRA, duas colunas), figuras de publicação e o código que as reproduz. Abstract, introdução, considerações
finais e conclusão são rascunhos; o que falta rodar está marcado com `\TODO`.

Estrutura pedida pelo autor (05/11/2026): introdução; HQPINN em geral, servindo aos dois problemas; resultados do direto e do inverso, cada um em
simulação e hardware real; considerações finais; conclusão. Quatro apêndices, um por resultado (A direto/simulação, B direto/hardware, C inverso/simulação,
D inverso/hardware), com todos os detalhes técnicos. A parte central é curta e passa a mensagem; o detalhe fica nos apêndices.

Layout e comandos: `README.md` e `manuscript/README.md`.

## O que mudou na fusão dos repositórios (05/11/2026)

- `forward-problem-hqpinn` e `inverse-problem-hqpinn` entraram aqui (todos os arquivos de `data/` conferidos por SHA-256 antes de apagá-los). Histórico: GitHub e `archive/git_bundles/`. Os hashes citados nestas notas (`552d791`, `82a8056`) são desses repositórios.
- O código fundido reproduz bit a bit as runs clássicas do inverso nesta máquina (seed 0, K = 110 em T = 0,1 e K = 500 em T = 1). O emulador rodou com o código fundido (teste curto), mas não fixa a seed.
- **`data/derived/` é registro, não cache.** O replay do `ibm_berlin` reproduz os pesos exatamente só no Windows do run. No Linux, com o mesmo torch e as mesmas medidas, os ângulos diferem em 6×10⁻⁸ na 1ª época, o erro de arredondamento cresce até ordem 1 (resíduo por diferenças finitas em precisão simples) e o MSE final (surrogate) vai de 1,4×10⁻³ a 6,3×10⁻³. Conferido que acontece igual com o código original do repositório do direto. `make_all.py` não recalcula mais `data/derived/` sem `--refresh-derived`, e o replay recusa gravar se não reproduzir. Está no Apêndice B; **vale discutir se vai para o texto principal**, porque o treino é caótico nessa taxa de aprendizado.
- Conciliação física: o direto dizia que o perfil em T = 0,1 "já é afiado" e o inverso dizia "antes de empinar". Calculado: o tempo de quebra invíscido é 1/(2π) ≈ 0,159; em T = 0,1 há max|u_x| = 13,6 (contra 2π em t = 0) e nenhum choque; em T = 1 há choque viscoso, max u = 0,331, rms 0,212.
- O piso de 7×10⁻⁶ em T = 0,1 é o erro da própria FDM: contra Cole-Hopf, MSE = 6,9×10⁻⁶ em T = 0,1 e 4,0×10⁻⁶ em T = 1 (2000 pontos).
- Citação retirada: `PhysRevA.110.032607` ("PINNverse") era uma entrada vazia e não foi encontrada.
- `--real` em `Inverser_quantum_hardware_pinn_E110.py` agora grava em `data/Inverso_real/<backend>/`. Antes sobrescreveria as runs do emulador, que têm os mesmos nomes de arquivo.
- As cadeias do FakeBerlin gravadas em `Inverso_110` (33, 14, 9) vieram do `find_chains` original e não são reproduzidas pelo `find_chains` atual (31, 11, 7 mantidas). As do FakeAachen são (41, 14, 10). O que vale é a calibração em cache; sem ela, o FakeBerlin usaria outras cadeias.

## Figuras no texto: escolhas do autor (05/11/2026)

- Fig. 1: só a arquitetura, sem os painéis de resíduo e de fluxo de treino, sem tamanhos de rede e com a saída apenas `u_Θ`. O resíduo e o laço de treino estão no texto da Sec. II e no Apêndice B.
- Figs. 2(b,e) e 3(b,d) (09/11/2026): razão entre os passos médios da clássica e da HQPINN, em escala log; na Fig. 3, uma só cor (marrom) e a largura pelo marcador.
- Problema inverso no texto (10/11/2026): Fig. 5 = melhor run de cada modelo e T com 200 épocas, HQPINN no FakeAachen (`data/Inverso_200`). Apêndice C: Fig. 8 = erro × tamanho n como as Figs. 2(a,d); Fig. 9 = melhores runs com 20 000 épocas e circuito ideal. Não há análogo da Fig. 3: todas as runs do inverso usam N_col = 10. As pastas de 110 e 500 épocas foram apagadas pelo autor (commit a36240c96) e saíram do artigo.

## Organização dos resultados: problema direto

| Grupo | Conjunto | Dados | Saída |
|---|---|---|---|
| Simulação | varredura com protocolo idêntico, clássica × HQPINN ideal, n = 2–15, N_col ∈ {10, 100}, T ∈ {0,1; 1}, 50 seeds (5600 runs) | `data/Forward/CBPF_qubit_sweep_quantum_classical_fixed_validation_results/` | Figs. 2, 3 |
| Simulação | HQPINN ideal × N_col (10–100), n = 3, 5, 7, 10, 15, 50 seeds | `data/Forward/CBPF_quantum_exact_results_*` | Fig. 3 |
| Simulação | orçamento do hardware (110 épocas): HQPINN ideal e ideal + ruído de shot, seeds 0–4 | `data/derived/same_budget_sims.npz` | Fig. 4(b) |
| Simulação | PINN clássica com o protocolo do QPU (n = 1–10, seeds 0–4) | `data/Forward/classical_vs_qpu/T_0.1/` | Fig. 4 |
| QPU real, T = 0,1 | treino com o `ibm_berlin` no loop (n = 10, N_col = 10, seed 0) | `data/quantum_forward_hardware_real/T_0.1/` | Fig. 4(a,b), mapa do chip, tabela do dispositivo |
| QPU real, T = 1 | **ainda não rodado** | `data/quantum_forward_hardware_real/T_1.0/` | Fig. 4(c,d) |

| Script | Saída | O que faz |
|---|---|---|
| `paper_common.py` | — | caminhos, estilo, leitores de dados, lista `QPU_CASES` (o caso de QPU de cada T) |
| `build_paper_data.py` | `manuscript/data/forward/*.csv` | condensa os dados brutos nas tabelas do artigo |
| `fig2_sim_width.py` | `fig2_sim_width.pdf` | mediana do MSE (sombra: quartis) e média dos passos (sombra: desvio) × n, MSE × parâmetros; cor = modelo, marcador = N |
| `fig3_sim_points.py` | `fig3_sim_points.pdf` | mediana do MSE e média dos passos × N (T = 0,1 e T = 1) |
| `fig4_qpu.py` | `fig4_qpu.pdf` | QPU real: u(x,T) e MSE × época, um caso por T; desenha "pending" enquanto a rodada não existir |
| `figB1_chip.py` | `figB1_chip.pdf` | mapa do chip com as 8 cadeias (Apêndice B) |
| `replay_training.py` | `data/derived/learning_curves.npz` | refaz offline os treinos a partir das medidas gravadas e avalia u(x,T) a cada época |
| `simulate_same_budget.py` | `data/derived/same_budget_sims.npz` | HQPINN ideal e com ruído de shot, com o protocolo do hardware |
| `controls_hardware_budget.py` | `data/derived/controls_hardware_budget.npz` | clássica com outros cronogramas de LR e com seno |

Paleta fixa por entidade: azul = PINN clássica, laranja = HQPINN (QPU real e circuito ideal), aqua = HQPINN em emulador com ruído, preto = FDM. Exceção na Fig. 3, a pedido do autor: preto = clássica, uma cor por n = HQPINN, um símbolo por n nos dois.

## Por que a pasta `data/Forward/classical` não entra nas figuras

**Lembrete pedido pelo autor em 04/11/2026, para discutir depois.** Comparar essa pasta com a HQPINN não é justo, e na maior parte dos casos erra a favor do quântico. Os dados continuam em `data/Forward/classical*` e os scripts que os plotavam em `run/plots/legacy/`, fora do artigo:

- **Critério de parada diferente.** A pasta antiga monitora a grade `uniform_exact` de N_col pontos; a HQPINN monitora um
  conjunto de validação aleatório fixo. Por isso o alvo muda muito: em T = 1, N_col = 100 é 1,7×10⁻⁴ contra 3,9×10⁻³
  (medianas sobre as seeds) e nenhum run clássico antigo atinge o alvo.
- **Arquitetura diferente para n ≠ 10.** Na pasta antiga o bloco do meio é n→10→n (42n+71 parâmetros); na varredura é
  n→n→n (2n²+23n+61). Só coincidem em n = 10.

O mesmo modelo clássico nos dois protocolos, onde os dois existem (mediana de 50 seeds):

| T | N_col | n | MSE, pasta antiga | MSE, protocolo comum | alvo atingido, antiga | alvo atingido, comum |
|---|---|---|---|---|---|---|
| 0,1 | 10 | 5 | 3,6×10⁻² | 2,8×10⁻⁵ | 2 % | 14 % |
| 0,1 | 10 | 10 | 8,0×10⁻⁵ | 1,9×10⁻⁵ | 2 % | 56 % |
| 0,1 | 100 | 10 | 6,3×10⁻⁶ | 6,7×10⁻⁶ | 50 % | 64 % |
| 1 | 10 | 10 | 1,4×10⁻¹ | 2,7×10⁻³ | 0 % | 6 % |
| 1 | 100 | 10 | 2,2×10⁻⁴ | 3,0×10⁻⁴ | 0 % | 34 % |

Em T = 1 com N_col = 10 o clássico antigo fica no nível de "não evoluiu": 50 vezes pior só por causa do protocolo.

## Plots do autor (PDF "HQPINN 10_2026") e o protocolo

As análises do PDF entraram no artigo (Figs. 2 e 3) com as curvas clássicas do protocolo comum. No PDF, as curvas quânticas já são
do protocolo comum (n = 10, T = 1, N = 100: passos 13,3×10³) e as clássicas são da pasta antiga. Médias sobre as seeds; valores do PDF lidos do gráfico, n = 10:

| Caso | Clássica no PDF | Clássica, protocolo comum | HQPINN |
|---|---|---|---|
| T = 0,1, N = 10, MSE | ≈ 1,3×10⁻² | 2,95×10⁻⁵ | 1,46×10⁻⁵ |
| T = 1, N = 10, MSE | ≈ 1,4×10⁻¹ | 4,86×10⁻³ | 2,17×10⁻³ |
| T = 1, N = 100, MSE | ≈ 4,3×10⁻⁴ | 9,70×10⁻⁴ | 1,13×10⁻⁴ |
| T = 1, N = 100, passos | 20,0×10³ | 17,8×10³ | 13,3×10³ |

A diferença de centenas de vezes na página 3 do PDF é efeito do protocolo: com o mesmo protocolo é de cerca de 2 vezes. Em N = 100 o efeito inverte em T = 1: a clássica antiga
tem erro menor, porque nunca para antes do orçamento. O PDF tem n = 1, que não existe no protocolo comum (n = 2–15).

Saíram do artigo a pedido do autor as tabelas de comparação em n = 10, de menor rede por erro e de erros no orçamento do hardware
(`git show 82a8056:paper/main.tex`, no repositório do direto). `run_qubit_sweep.py --fixed-validation` roda a clássica nos outros N com o mesmo protocolo: conferido contra a varredura
publicada em 14 seeds (clássica, n = 10, T = 1, N = 100), com alvo idêntico e mesmo resultado nas 14, a cerca de 50 s por run nesta máquina.

## O que falta rodar

**Clássica em N = 20–90: feita** (4000 runs, terminou em 05/11/2026; a Fig. 3 está completa). Para repetir ou retomar: `bash run/run_classical_ncol.sh`, depois
`python manuscript/scripts/make_all.py`. Sem `raw_runs/`, o script refaz tudo.

Medidas nesta máquina (i7-13620H):
- **n = 10:** 800 runs em 1h08 com 10 processos, 47 s por run em média. Começou na bateria e terminou na tomada.
- **Paralelismo (teste de 3000 passos, na bateria):** 48 runs completos/h com 1 processo, 197 com 4, 318 com 8, 365 com 10, 384 com 12, 383 com 14, 377 com 16.
  Threads por processo não ajudam. Na tomada não medi com 12 processos; n = 10 deu cerca de 700 runs/h com 10.
- **GPU: não testada.** O módulo `nvidia` não carrega (Secure Boot ligado, módulo sem assinatura, causa provável), o torch instalado é só CPU e o código
  não tem opção de dispositivo. A expectativa de que não compense, com redes pequenas contra 12 processos, não foi medida.

**QPU real em T = 1 (o autor vai enviar).** Caso proposto, a confirmar: n = 10, N_col = 20, seed 0, 110 épocas (a configuração atual de
`run_quantum_ibm_faker.py`; o único teste em T = 1 no repositório são PNGs de n = 3, N_col = 10, seed 2 no FakeBerlin).

1. Rodar antes no FakeBerlin (`run_quantum_ibm_faker.py --no-real`) e olhar o erro. Fixar a seed antes de ver o QPU; se for
   a melhor de um lote, dizer isso no texto, como em T = 0,1.
2. Em `run/run_quantum_hardware.py`, ajustar `TFIN` e `N_COLLOCATION` (estão fixos em 0,1 e 10) e usar o mesmo otimizador
   do teste no FakeBerlin. `run_quantum_hardware_real.py --plan --seed <s>` mostra o que vai rodar sem gastar job.
3. Rodar `run_quantum_hardware_real.py --seed <s>`. Saída em `data/quantum_forward_hardware_real/T_1.0/`.
4. Controle clássico com o mesmo protocolo: `run_classical_vs_qpu.py` com `REAL_RUN` apontando para a rodada nova.
5. Replay e simulação sem ruído: `replay_training.py` e `simulate_same_budget.py` têm T = 0,1 fixo; adaptar e salvar como
   `data/derived/learning_curves_T1.npz` e `same_budget_sims_T1.npz`. Fazer na mesma máquina do run: fora dela o replay não reproduz.
6. Conferir o caso em `QPU_CASES` (`paper_common.py`) e rodar `make_all.py`: a Fig. 4(c,d) se preenche sozinha.
7. Escrever os números no parágrafo "T = 1" da Sec. "Real quantum processor" e tirar os `\TODO`.

## Principais números (todos conferidos nos dados; mediana do MSE e média dos passos, 50 seeds)

- **T = 0,1, N = 100:** mesmo erro para n ≥ 5 (MSE ≈ 7×10⁻⁶). A HQPINN usa cerca de metade dos passos (fator 1,95–2,16 para n ≥ 7; n = 10:
  7,6×10³ contra 1,52×10⁴) e atinge o alvo em 98 % dos runs com n ≥ 5, contra 55 %. MSE de 8×10⁻⁶: n = 3 (124 parâmetros) contra n = 5 (226).
- **T = 0,1, N = 10:** MSE menor em todo n (fator 1,2–4,5). MSE de 3×10⁻⁵: n = 2 (103) contra n = 5 (226).
- **Clássica × HQPINN, n = 3, 5, 7, 10, 15, N = 10–100 (Fig. 3; 50 configurações por T, alvos idênticos nos 4000 pares novos):**
  - T = 0,1: a clássica usa mais passos nas 50, por um fator de 1,06–1,71 (n ≤ 5) e 1,50–2,08 (n ≥ 7), e atinge menos o alvo nas 50. MSE clássico ÷ HQPINN: 0,79–1,50 para n ≥ 5 e N ≥ 20; 1,14–2,79 em n = 3. A clássica tem mediana menor em 8 das 50, em geral por poucos por cento (a maior diferença é n = 15, N = 40: 21 %).
  - T = 1: MSE clássico maior nas 50: 1,73–6,24 (n ≤ 10) e 4,73–17,65 (n = 15). Melhor mediana clássica entre as cinco larguras: 2,1×10⁻⁴ (n = 10, N = 80, 491 parâmetros); a HQPINN fica abaixo com n = 7, N = 30 (1,95×10⁻⁴, 208 parâmetros). Na Fig. 2, n = 9 em N = 100 chega a 1,1×10⁻⁴, por isso o texto diz "entre as cinco larguras".
  - T = 1, passos (clássica ÷ HQPINN): até 1,02 em n = 3, 1,09 em n = 5, 1,19 em n = 7, 1,33 em n = 10, 1,61 em n = 15. A razão fica abaixo de 1 em quatro casos, nenhum abaixo de 0,99. Alvo: a HQPINN atinge pelo menos tanto quanto a clássica em 49 de 50; a exceção é n = 10, N = 20 (22 % contra 24 %).
  - Não entrou no texto: a mediana clássica em T = 1 é maior em n = 15 do que em n = 10 em todo N (1,05–5,35 vezes), mas a diferença não se sustenta bem por N (Wilcoxon pareado por seed: p = 0,26, 0,036 e 0,089 em N = 20, 50 e 80), e as medianas clássicas oscilam com n (N = 100: 1,1×10⁻⁴ em n = 9, 3,0×10⁻⁴ em n = 10, 1,6×10⁻⁴ em n = 12).
- **T = 1, N = 100:** MSE menor em todo n (até 14 vezes). n = 10: 4,8×10⁻⁵ contra 3,0×10⁻⁴, alvo em 92 % contra 34 %. MSE de 10⁻⁴: n = 6 (187);
  a clássica não chega até n = 15 (856). Passos: fator 1,16–1,6 para n ≥ 7.
- **T = 1, N = 10:** MSE menor em todo n (até 5,5 vezes). Nenhum modelo atinge o alvo na maioria dos runs (no máximo 20 % e 12 %).
- **Média dos passos:** run que não atinge o alvo conta no orçamento, então é limite inferior, e subestima mais a clássica.
- **Mesmo setup:** `build_paper_data.py` para com erro se os dois modelos não tiverem o mesmo critério de parada e o mesmo alvo por seed.
- **QPU real, T = 0,1 (n = 10, 110 épocas):** 5,7 % (MSE 1,45×10⁻³), inicial 29,4 %. Clássica, mesma seed e largura: 31,2 %. Mediana das 50
  clássicas: 24,7 %; só 1 de 50 termina abaixo. HQPINN ideal: 3,9–15,1 % (mediana 9,1 %). Clássica com o melhor cronograma: 11,4 % (6,2–19,5 %).
- **FakeBerlin:** seed 0 (4,13 %), a mesma rede no `ibm_berlin` (4,83 %) e com o circuito ideal (3,18 %) conferidos em `data/quantum_ibm_faker/T_0.1/run_*.json`, restaurado do commit `daebd09` deste repositório.
  **Não reconferido:** seeds 1–4 (6,9–8,1 %): os arquivos brutos não estão nem no histórico. Vêm da versão anterior destas notas; há um `\TODO` no Apêndice B, e daí depende a frase "seed 0 foi a melhor de 5".

## Problema inverso

| Grupo | Conjunto | Dados | Saída |
|---|---|---|---|
| Simulação | 200 épocas, clássica × HQPINN no FakeAachen, n = 10, seeds 0–9 (10 pares por T) | `data/Inverso_200/` | Fig. 5, Tabs. II, IX |
| Simulação | 20 000 épocas, circuito ideal, n = 3, 7, 10, seeds 0–10 (33 pares por T) | `data/Inverso/` | Figs. 8, 9, Tabs. II, X |
| QPU real | **não existe nenhuma rodada** | `data/Inverso_real/<backend>/` | Fig. 7, Sec. IV B, Apêndice D |

Todos os números abaixo saem de `manuscript/data/inverse/` (conferidos em 10/11/2026):

- **K = 200, T = 0,1:** ε_ν mediano 1,9 % (HQPINN) × 4,6 %; p = 0,43; HQPINN melhor em 6/10. L2 0,011 × 0,009 (clássica melhor em 6/10, p = 0,32).
- **K = 200, T = 1:** ε_ν 36,9 % × 18,0 %; p = 0,23; clássica melhor em 7/10. L2 0,063 × 0,050 (7/10, p = 0,11). ν subestimado em 10/10 e 7/10. Oscilação de ν nas últimas 20 épocas ≈ 20 % (2 % em T = 0,1).
- **K = 200, mesmos pesos com o circuito ideal:** T = 0,1: pior em 9/10, fator mediano 1,09. T = 1: 4/10, fator 0,99.
- **K = 20 000 (n = 3, 5, 7, 10, 12, 15; 66 pares):** T = 0,1: 0,9 % × 0,8 % (HQPINN melhor em 34/66, p = 0,71); T = 1: 2,5 % × 1,7 % (clássica melhor em 38/66, p = 0,034). L2 0,004 × 0,003 e 0,011 × 0,007. Em T = 1 a HQPINN melhora com n: 11,7 % (n = 3), 4,6 % (n = 5), 1,2–2,2 % (n ≥ 7).
- **Clássica, só as épocas (n = 10):** T = 1: 18,0 % (K = 200) → 1,8 % (K = 20 000).
- **Não reconferido (dados apagados, não estão no histórico):** os 147 testes de treino e o proxy (38,5 → 12,7 %, 0,23 → 0,067, 32,7 → 20,7 %, 10,5 % e 4,2 %), e os 6,7 % × 4,0 % da seleção de hiperparâmetros. Vêm de `data/Inverso_110/README.md` e do rascunho; há `\TODO` no Apêndice C. O README cita 0,065 para o L2 só com a IC fraca e o rascunho, 0,067 com os três ajustes.

**Para rodar no QPU real.** Casos propostos em `INVERSE_QPU_CASES` (n = 10, N = 10, seed 0, 110 épocas, T = 0,1 e 1); custo estimado na Tab. XI (112 jobs, cerca de 1 h por run). Decisões abertas no Apêndice D: mitigação (o padrão do script é `suppress`, o direto foi sem mitigação), qual QPU, seed fixada antes, e quantas seeds, porque uma só não se compara em ν.

```bash
source run/env.sh
python run/Inverser_quantum_hardware_pinn_E110.py --real --tfin 0.1 --nu 0.01 --qubits 10 --seeds 0   # -> data/Inverso_real/ibm_berlin/
python manuscript/scripts/make_all.py                                                                  # a Fig. 7 se preenche sozinha
```

**Falta para fechar a comparação com o direto:** o inverso com HQPINN sem ruído e clássica no mesmo protocolo, e o controle clássico com produto de cossenos. Sem isso, o efeito da arquitetura não se separa do efeito do ruído (Sec. V).

## Pontos que exigem atenção antes da submissão

1. **A "bimodalidade" da PINN clássica com N_col = 10 não aparece no protocolo comum.** (As seções
   1.4, 6.1, 6.2 e 7 de `docs/results_report_pt.md` já foram corrigidas.) Ela vem da varredura antiga `data/Forward/classical`, cujo monitor
   de parada e de melhor snapshot é a grade `uniform_exact` de 10 pontos. Na varredura com protocolo
   idêntico (validação fixa aleatória), nenhum run clássico falha: o MSE máximo é 2,2×10⁻³ em 1400
   runs com T = 0,1. Por isso o manuscrito usa só a varredura de protocolo comum. A frase "nenhuma
   falha em 50 seeds no quântico, 36 % no clássico" compara protocolos diferentes e não deve entrar
   no artigo.
2. **Seleção da seed do hardware:** a seed 0 foi a melhor de 5 no lote FakeBerlin. Isso está dito no
   texto.
3. **Controle clássico no orçamento do hardware:** usa o cronograma de LR escolhido para a HQPINN. O melhor de dez
   cronogramas (11,4 %) está no texto e no apêndice.
4. **Script da varredura:** os dados de `CBPF_qubit_sweep_...` foram gerados por uma versão
   modificada (`Experiment1Modified`, protocolo `fixed_validation_v2`) que não está no repositório.
   O `run_qubit_sweep.py` local, sem flag, monitora o lote de treino; com `--fixed-validation` segue o mesmo protocolo.
   Ainda vale arquivar o script original.
5. **Arquitetura clássica:** na varredura o bloco do meio é n→n→n (2n²+23n+61 parâmetros). No
   controle com o protocolo do QPU é n→10→n (42n+71). Os dois coincidem em n = 10 (491).
   Está documentado na tabela de arquiteturas do apêndice.
6. **Problema inverso:** ver a seção própria abaixo.
7. **Referências a conferir:** volume/página do QCPINN (Farea et al.), autores do arXiv:2606.04679.
   A citação "PINNverse, PRA 110, 032607" do rascunho antigo não foi encontrada.

## Literatura: artigos-modelo (estrutura e estilo)

| Artigo | Por que serve de modelo |
|---|---|
| Wright *et al.*, *Nature* **601**, 549 (2022), "Deep physical neural networks trained with backpropagation" | Análogo metodológico mais próximo: forward no sistema físico e backward por um modelo digital dele (*physics-aware training*), como o nosso estimador straight-through. Modelo de narrativa e de figura para a parte de hardware. |
| Havlíček *et al.*, *Nature* **567**, 209 (2019), "Supervised learning with quantum-enhanced feature spaces" | Feature map quântico fixo em hardware IBM. Estrutura compacta: teoria do mapa, experimento e discussão do ruído. |
| Kyriienko, Paine e Elfving, *Phys. Rev. A* **103**, 052416 (2021), "Solving nonlinear differential equations with differentiable quantum circuits" | Referência APS para circuitos quânticos resolvendo EDPs não lineares. Modelo de notação e de organização da seção teórica. |
| Karniadakis *et al.*, *Nat. Rev. Phys.* **3**, 422 (2021), "Physics-informed machine learning" | Enquadramento das PINNs para a introdução e a seção de PINN clássica. |
| Schreiber, Eisert e Meyer, *Phys. Rev. Lett.* **131**, 100803 (2023), "Classical surrogates for quantum learning models" | Fundamenta o surrogate e o enquadramento honesto sobre vantagem quântica (Remark 1). |
| Kim *et al.*, *Nature* **618**, 500 (2023), "Evidence for the utility of quantum computing before fault tolerance" | Modelo de relato de experimento em hardware IBM: caracterização do dispositivo, mapa do chip, comparação com referência clássica e mitigação. |

Posicionamento (QPINNs, todos com circuito treinável e em simulação): Sedykh *et al.*, MLST **5**,
025045 (2024); Berger, Hosters e Möller, *Sci. Rep.* **15**, 18823 (2025); Farea *et al.*, MLST (2025);
Lantigua, Giraldi e Portugal, *PRA* **113**, 042446 (2026); Zabihi *et al.*, arXiv:2606.04679 (2026).

## Outras ideias de figuras (não implementadas)

- **Mapa espaço-tempo do erro** |u − u_FDM|(x, t) da rede treinada no QPU, pelo surrogate: mostra
  onde e quando o erro nasce (frente de choque). É barato, porque a FDM já guarda todos os tempos.
- **Surrogate por cadeia (ablação offline):** ajustar coeficientes por cadeia com os dados da
  calibração e medir quanto cai o RMS de Δ. Testa diretamente a conclusão da Fig. 5.
- **Varredura de LR do controle clássico** no orçamento de 110 épocas (CPU, minutos).
- **Controle "produto de cossenos"/seno** (`seno=True`): separa o efeito da forma funcional do efeito
  de a camada ser fixa.
- **Custo × precisão:** erro final × shots por circuito com o proxy de ruído de shot
  (`simulate_same_budget.py` com S variável).
- **Fidelidade do gradiente:** comparar ∂f̂/∂θ com o parameter-shift medido em alguns pontos.
  Exige alguns jobs extras no QPU.
