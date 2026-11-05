# Manuscrito: HQPINN para os problemas direto e inverso

Formato APS (revtex4-2, PRA, duas colunas). Texto em inglês; os `\TODO` em vermelho são notas de trabalho
(`\showtodosfalse` no preâmbulo esconde todos). Notas, números conferidos e pendências: `docs/manuscript_notes_pt.md`.

Estrutura: introdução; a HQPINN em geral, para os dois problemas (Sec. II); resultados do problema direto (Sec. III) e do
inverso (Sec. IV), cada um em simulação e hardware real; considerações finais; conclusões. Quatro apêndices, um por
resultado, com todos os detalhes técnicos: A (direto, simulação), B (direto, hardware), C (inverso, simulação), D (inverso, hardware).

```
manuscript/
  main.tex, references.bib   texto e bibliografia (entradas "VERIFY" precisam de conferência)
  figures/                   schematic.tex (Fig. 1, TikZ) e os PDFs gerados pelos scripts
  tables/                    tabelas LaTeX geradas dos dados (\input no main.tex): não edite à mão
  data/                      tabelas CSV pequenas, de onde as figuras saem (ver data/README.md)
  scripts/                   código das tabelas e figuras
  build.sh                   compilação local
```

## Como reproduzir

Nenhum job no QPU e nenhuma simulação Aer é executada.

```bash
python manuscript/scripts/make_all.py      # dados brutos -> manuscript/data/ e tables/ -> manuscript/figures/
bash manuscript/build.sh                   # main.pdf (precisa de revtex4-2 e tikz)
python manuscript/scripts/make_overleaf_zip.py   # manuscript_overleaf.zip, para subir no Overleaf
```

No Debian, `build.sh` precisa de `sudo apt install texlive-publishers texlive-pictures texlive-latex-recommended texlive-latex-extra`.

**Não rode `make_all.py --refresh-derived` no Linux sem motivo.** `data/derived/` é o registro feito no Windows do autor: o
treino é caótico no nível do arredondamento, e em outra plataforma os resultados mudam (`replay_training.py` recusa gravar se não
reproduzir o run; os outros dois recalculam). Ver Apêndice B, "Offline replay".

## Figuras

| Figura | Arquivo | Script | Dados |
|---|---|---|---|
| 1 | `figures/schematic.tex` | TikZ | -- |
| 2 | `fig2_sim_width.pdf` | `fig2_sim_width.py` | `data/forward/simulation/width_sweep.csv` |
| 3 | `fig3_sim_points.pdf` | `fig3_sim_points.py` | `data/forward/simulation/points_sweep.csv` |
| 4 | `fig4_qpu.pdf` | `fig4_qpu.py` | `data/forward/qpu/` (T = 1 pendente) |
| 5, 9 | `fig5_inv_nu_E500.pdf`, `figC1_inv_nu_E110.pdf` | `fig5_inv_nu.py` | `data/inverse/nu_curves.csv`, `runs.csv` |
| 6 | `fig6_inv_field.pdf` | `fig6_inv_field.py` | `data/inverse/field.csv`, `runs.csv` |
| 7 | `fig7_inv_qpu.pdf` | `fig7_inv_qpu.py` | `data/inverse/qpu_cases.csv` (tudo pendente) |
| 8 | `figB1_chip.pdf` | `figB1_chip.py` | `data/forward/qpu/T_0.1/chains.csv` |

Paleta fixa por entidade: azul = PINN clássica, laranja = HQPINN (sem ruído e no QPU real), aqua = HQPINN em emulador com ruído,
preto = referência. Exceção na Fig. 3, a pedido do autor: preto = clássica, uma cor por n = HQPINN.
