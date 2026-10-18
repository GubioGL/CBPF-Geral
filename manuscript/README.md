# Manuscrito: PINN híbrida com camada quântica fixa treinada no `ibm_berlin`

Rascunho no formato APS (revtex4-2, PRA, duas colunas), figuras de publicação e o código
que as reproduz. Abstract, introdução e conclusão estão como placeholders (`\TODO`).

```
manuscript/
  main.tex              texto (teoria, métodos, resultados, discussão-rascunho, apêndices)
  references.bib        bibliografia (entradas "VERIFY" precisam de conferência)
  figures/schematic.tex Fig. 1 em TikZ (entra no main.tex por \input)
  figures/*.pdf, *.png  figuras geradas pelos scripts
  scripts/              código das figuras (ver abaixo)
  cache/                resultados intermediários (replay do treino, simulações de mesmo orçamento)
  build.sh              compilação local (MiKTeX); no Overleaf basta compilar main.tex
```

**Overleaf:** envie `main.tex`, `references.bib`, `figures/schematic.tex` e `figures/*.pdf`
(ou o arquivo `manuscript_overleaf.zip`). Compilador pdfLaTeX, BibTeX.

## Como reproduzir as figuras

Ambiente conda `qiskit`. Nenhum job no QPU e nenhuma simulação Aer é executada.

```bash
C:/Users/Gubio/anaconda3/envs/qiskit/python.exe manuscript/scripts/make_all.py
```

`make_all.py --skip-cache` regenera só as figuras, reaproveitando `cache/`.

## Organização dos resultados: simulações × QPU real

| Grupo | Conjunto | Dados | Figura |
|---|---|---|---|
| Simulação | varredura com protocolo idêntico, clássica × HQPINN ideal, n = 2–15, N_col ∈ {10, 100}, T ∈ {0,1; 1}, 50 seeds (5600 runs) | `data/Forward/CBPF_qubit_sweep_quantum_classical_fixed_validation_results/` | Fig. 2, Tab. I |
| Simulação | HQPINN ideal × N_col (10–500), n = 3, 5, 7, 10, 15 | `data/Forward/CBPF_quantum_exact_results_*` | Fig. S1 |
| Simulação | orçamento do hardware (110 épocas): HQPINN ideal e ideal + ruído de shot, seeds 0–4 | gerado por `simulate_same_budget.py` → `cache/` | Fig. 3d, 4, Tab. II |
| Simulação | HQPINN treinada no FakeBerlin (mesmas cadeias do run real) | `data/quantum_ibm_faker/T_0.1/` | Fig. 3, 4, 5 |
| Simulação | PINN clássica com o protocolo do QPU (n = 1–10, seeds 0–4) | `data/Forward/classical_vs_qpu/T_0.1/` | Fig. 3, 4 |
| QPU real | treino com o `ibm_berlin` no loop (seed 0) | `data/quantum_forward_hardware_real/T_0.1/` | Fig. 3, 4, 5, Tab. II, IV |
| QPU real | rede treinada no FakeBerlin avaliada no `ibm_berlin` | `data/quantum_ibm_faker/T_0.1/prediction_real_*.npz` | Fig. 3, 4 |

| Script | Saída | O que faz |
|---|---|---|
| `paper_common.py` | — | caminhos, estilo (larguras APS, paleta), leitores de dados |
| `replay_training.py` | `cache/learning_curves.npz` | refaz offline os treinos do QPU real, do FakeBerlin e do controle clássico a partir das medidas gravadas; confere bit a bit com os pesos salvos (diferença 0) e avalia u(x,T) a cada época |
| `simulate_same_budget.py` | `cache/same_budget_sims.npz` | HQPINN ideal e ideal + ruído binomial (1024 shots) com o protocolo do hardware |
| `fig2_simulations.py` | `fig2_simulations.pdf` | MSE × n, distribuição das 50 seeds, fração que atinge o alvo físico × passos |
| `fig3_hardware.py` | `fig3_hardware.pdf` | mapa do chip com as 8 cadeias, u(x,T), erro pontual com a dispersão entre cadeias, curva de aprendizado por época |
| `fig4_noise_ladder.py` | `fig4_noise_ladder.pdf` | "escada de ruído": erro L2 do modelo ideal até o QPU real, no mesmo orçamento |
| `fig5_device_model.py` | `fig5_device_model.pdf` | resíduos da calibração × ruído de shot, contraste por qubit, desajuste por cadeia, RMS de Δ por época |
| `figS1_exact_ncol.py` | `figS1_exact_ncol.pdf` | HQPINN × N_col e eficiência em parâmetros |

Paleta fixa por entidade: azul = PINN clássica, laranja = HQPINN (QPU real e circuito ideal),
verde-água = FakeBerlin, preto = FDM. Marcadores abertos = mesmos pesos avaliados no circuito ideal.

## Principais números (todos conferidos nos dados)

- **Simulação, mesmo protocolo (n = 10):** em T = 0,1 os dois modelos chegam ao piso da FDM
  (MSE ≈ 7×10⁻⁶) com N_col = 100. A HQPINN atinge o alvo físico em 100 % dos runs, contra 64 %,
  e com cerca de metade dos passos e 271 contra 491 parâmetros. Em T = 1, N_col = 100: MSE 4,8×10⁻⁵
  contra 3,0×10⁻⁴ (L2 3,3 % contra 8,2 %).
- **Orçamento do hardware (110 épocas):** HQPINN ideal 3,9–15,1 % (mediana 9,1 %); com ruído de shot
  3,8–16,2 %; FakeBerlin 4,1 %; FakeBerlin → `ibm_berlin` 4,8 %; treinada e avaliada no `ibm_berlin`
  5,7 % (MSE 1,45×10⁻³). Clássica com o mesmo protocolo: mediana 24,7 %.
- **Conclusão física:** nesse orçamento o ruído do hardware não é o fator limitante; o resultado do
  QPU real fica dentro da dispersão entre seeds do modelo sem ruído.
- **Modelo do dispositivo:** no FakeBerlin o resíduo do polinômio fica no nível do ruído de shot
  (0,030). No `ibm_berlin` fica em ≈ 0,045 durante todo o treino, sem drift sistemático, e cai para
  0,013 com a média sobre as 8 cadeias (piso 0,011). O excesso vem principalmente da variação entre
  cadeias.

## Pontos que exigem atenção antes da submissão

1. **A "bimodalidade" da PINN clássica com N_col = 10 não aparece no protocolo comum.** (As seções
   1.4, 6.1, 6.2 e 7 do README principal já foram corrigidas.) Ela vem da varredura antiga `data/Forward/classical`, cujo monitor
   de parada e de melhor snapshot é a grade `uniform_exact` de 10 pontos. Na varredura com protocolo
   idêntico (validação fixa aleatória), nenhum run clássico falha: o MSE máximo é 2,2×10⁻³ em 1400
   runs com T = 0,1. Por isso o manuscrito usa só a varredura de protocolo comum. A frase "nenhuma
   falha em 50 seeds no quântico, 36 % no clássico" compara protocolos diferentes e não deve entrar
   no artigo.
2. **Seleção da seed do hardware:** a seed 0 foi a melhor de 5 no lote FakeBerlin. Isso está dito no
   texto.
3. **Controle clássico no orçamento do hardware:** usa o cronograma de LR escolhido para a HQPINN.
   Uma varredura de LR para a clássica é barata (CPU) e fortaleceria a comparação (`\TODO` no texto).
4. **Script da varredura:** os dados de `CBPF_qubit_sweep_...` foram gerados por uma versão
   modificada (`Experiment1Modified`, protocolo `fixed_validation_v2`) que não está no repositório.
   O `run_qubit_sweep.py` local monitora o lote de treino. É preciso arquivar o script usado.
5. **Arquitetura clássica:** na varredura o bloco do meio é n→n→n (2n²+23n+61 parâmetros). No
   controle com o protocolo do QPU é n→10→n (42n+71). Os dois coincidem em n = 10 (491).
   Está documentado na Tab. V.
6. **Problema inverso:** o conteúdo do rascunho anterior (ν = 0,1, FakeBerlin) ficou fora deste
   manuscrito, que trata do problema direto. Pode voltar como seção de extensão.
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
