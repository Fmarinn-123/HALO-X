# HALO-X

Design tools for the gradient amplifier's current loop: an interactive viewer of the closed-loop polynomial and the loop equations behind it.

| File | What it is |
| --- | --- |
| `polinomio.py` | Interactive viewer of the closed-loop denominator D(s) |
| `Diagramas_Gradientes/ecuaciones.tex` | Loop equations (1)–(3), built to `.pdf` and `.svg` |

## Quick start

```bash
pip install numpy matplotlib
python polinomio.py
```

Move the sliders (R3, RF, C1, CF), or type in the boxes (L, R_L, emitter resistors, G) and press Enter. The plot, the poles, the readings panel and the component table update right away.

## The model

![Loop equations](Diagramas_Gradientes/ecuaciones.svg)

- **Gain.** G = R3/(R4·R5) [A/V], so G = 1 A/V when R5 = R3/R4. With the 0.1 Ω shunt, that means R5 = 10·R3. You type G and the tool computes R5. G scales the current but does not move the poles.
- **Series resistance.** The coil sees R_tot = R_L + R4 + Re_eq, where Re_eq is the parallel combination of the emitter resistors. N equal branches of Re give Re_eq = Re/N. Type one Re per branch, comma-separated, e.g. `0.1, 0.1`.
- **Pole-zero cancellation.** RF and CF are coupled so that CF·RF = L/R_tot. The denominator then becomes D(s) = a·s² + b·s + 1, with K = R3·R_tot/R4, a = K·C1·CF·RF and b = K·(C1 + CF). Its roots are the closed-loop poles.

The full derivation is in the header of `polinomio.py`.

## Things to keep in mind

- The polynomial is only valid while CF·RF = L/R_tot. If a slider hits its limit the cancellation breaks, and the panel says so.
- Op-amps are treated as ideal, and the output stage as a unity-gain follower.
- R4, the 0.1 Ω current shunt, is a constant at the top of `polinomio.py`. Change it there if your shunt is different.

## Rebuilding the equations

```bash
cd Diagramas_Gradientes
pdflatex ecuaciones.tex
latex ecuaciones.tex
dvisvgm --no-fonts --exact-bbox ecuaciones.dvi
```

## License

Free to use and modify for educational and engineering purposes.
