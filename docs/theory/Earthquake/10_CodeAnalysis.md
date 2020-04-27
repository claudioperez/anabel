# Code Analysis Procedures

<!-- $$\begin{array}{|l|l|l|l|l|l|l|l|}
\hline \multicolumn{3}{|c|} {\text { Structural Characteristics }} & \multicolumn{3}{|c|} {\text { Analytical Procedure }} \\
\hline \begin{array}{l}
\text { Performance } \\
\text { Level }
\end{array} & \begin{array}{l}
\text { Fundamental } \\
\text { Period, } T
\end{array} & \text { Regularity } & \begin{array}{l}
\text { Ratio of Column to } \\
\text { Beam Strength }
\end{array} & \begin{array}{l}
\text { Linear } \\
\text { Static }
\end{array} & \begin{array}{l}
\text { Linear } \\
\text { Dynamic }
\end{array} & \begin{array}{l}
\text { Nonlinear } \\
\text { Static }
\end{array} & \begin{array}{l}
\text { Nonlinear } \\
\text { Dynamic }
\end{array} \\
\hline \text { Immediate } & T \leq 3.5 T_{s}^{1} & \begin{array}{l}
\text { Regular or } \\
\text { Irregular }
\end{array} & \text { Any Condition } & \text { Permitted } & \text { Permitted } & \text { Permitted } & \text { Permitted } \\
\hline & T>3.5 T_{s}^{1} & \begin{array}{l}
\text { Regular or } \\
\text { Irregular }
\end{array} & \text { Any Conditions } & \begin{array}{l}
\text { Not } \\
\text { Perrmitted }
\end{array} & \text { Permitted } & \begin{array}{l}
\text { Not } \\
\text { Permitted }
\end{array} & \text { Permitted } \\
\hline \text { Collapse } & T \leq 3.5 T_{s}^{1} & \text { Regular }^{2} & \text { Strong Column }^{3} & \text { Permitted } & \text { Permitted } & \text { Permitted } & \text { Permitted } \\
\hline \text { Prevention } & & \text { Weak Column }^{3} & \begin{array}{l}
\text { Not } \\
\text { Permitted }
\end{array} & \begin{array}{l}
\text { Not } \\
\text { Permited }
\end{array} & \text { Permitted } & \text { Permitted } \\
\hline & \text { Irregular }^{2} & \text { Any Conditions } & \begin{array}{l}
\text { Not } \\
\text { Permitted }
\end{array} & \begin{array}{l}
\text { Not } \\
\text { Permitted }
\end{array} & \text { Permitted } & \text { Permitted } \\
\hline T>3.5 T_{s} & \text { Regular } & \text { Strong Column }^{3} & \begin{array}{l}
\text { Not } \\
\text { Permitted }
\end{array} & \text { Permitted } & \begin{array}{l}
\text { Not } \\
\text { Permitted }
\end{array} & \text { Permitted } \\
\hline & \text { Weak Column }^{3} & \begin{array}{l}
\text { Not } \\
\text { Permitted }
\end{array} & \begin{array}{l}
\text { Not } \\
\text { Permitted }
\end{array} & \begin{array}{l}
\text { Not } \\
\text { Permitted }
\end{array} & \text { Permitted } \\
\hline & \text { Irregular }^{2} & \text { Any Conditions } & \begin{array}{l}
\text { Not } \\
\text { Permitted }
\end{array} & \begin{array}{l}
\text { Not } \\
\text { Permitted }
\end{array} & \begin{array}{l}
\text { Not } \\
\text { Permitted }
\end{array} & \text { Permitted } \\
\hline
\end{array}$$ -->

<!-- \begin{tabular}{|l|l|l|l|l|l|l|l|}
\hline \multicolumn{3}{|c|} { Structural Characteristics } & \multicolumn{3}{|c|} { Analytical Procedure } \\
\hline Performance Level & Fundamental Period, $T$ & Regularity & Ratio of Column to Beam Strength & Linear Static & Linear Dynamic & Nonlinear Static & Nonlinear Dynamic \\
\hline Immediate & $T \leq 3.5 T_{s}^{1}$ & Regular or Irregular & Any Condition & Permitted & Permitted & Permitted & Permitted \\
\hline & $T>3.5 T_{s}^{1}$ & Regular or Irregular & Any Conditions & Not Perrmitted & Permitted & Not Permitted & Permitted \\
\hline Collapse & $T \leq 3.5 T_{s}^{1}$ & Regular $^{2}$ & Strong Column $^{3}$ & Permitted & Permitted & Permitted & Permitted \\
\hline Prevention & & Weak Column $^{3}$ & Not Permitted & Not Permited & Permitted & Permitted \\
\hline & Irregular $^{2}$ & Any Conditions & Not Permitted & Not Permitted & Permitted & Permitted \\
\hline$T>3.5 T_{s}$ & Regular & Strong Column $^{3}$ & Not Permitted & Permitted & Not Permitted & Permitted \\
\hline & Weak Column $^{3}$ & Not Permitted & Not Permitted & Not Permitted & Permitted \\
\hline & Irregular $^{2}$ & Any Conditions & Not Permitted & Not Permitted & Not Permitted & Permitted \\
\hline
\end{tabular} -->

## ASCE 7

- **Risk Category/Importance:** I - IV
- **Seismic Design Category:** $f: I_e,S_{DS},S_{D1} \mapsto A - D$
  - Imposes limitations on framing system selection

1. **Linear Static (ELF):** 
   - Structures where first mode is expected to control.
   - Some irregularities, but NOT Torsional/ soft story.
   - < 2 Stories
   - $h<160$ft
2. **Linear Dynamic (RHA/RSA)**
3. **Nonlinear Dynamic**
   - s

  Note: Nonlinear static only included in ASCE 41

### Quantify a Torsional Iregularity 

- apply $F_x$ at $\pm 5\%$
- measure $\delta_A,\delta_B$ at far ends.
- $\delta_{avg}=\frac{\delta_{A}+\delta_{B}}{2} < 1.2$



