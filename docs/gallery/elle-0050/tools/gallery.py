
nm = 5
G = [2,5,8,13,19]

print("### Plots of Finite Element Solutions\n\n")

for pblm in ["", "-b"]:
    for m in range(1,nm+1):
        for g in G:
            print(f"![Finite element solution for problem 1 over mesh number {m} and order-{g} numerical integration.](../img/mesh{m}-gauss{g:02}{pblm}.png){{width=50%}}\n")

print("### Errors in the $H^1$ and $L2$ norms\n\n")

NORMS = ["H^1", "L2"]

for pblm in ["", "-b"]:
    for m in range(1,nm+1):
        for g in G:
            print(f"""
![](../img/mesh{m}-gauss{g:02}{pblm}-L2.png){{width="50%"}}
![](../img/mesh{m}-gauss{g:02}{pblm}-H1.png){{width="50%"}}
\\begin{{figure}}
\\caption{{Finite element error in the L2 and H1 norms/seminorms, respectively for problem 1 over mesh number {m} using order {g} quadrature.}}
\\end{{figure}}

""")

#<figure>
#  <img src="../img/mesh{m}-gauss{g:02}{pblm}-L2.png" width="50%"/>
#  <img src="../img/mesh{m}-gauss{g:02}{pblm}-H1.png" width="50%"/>
#  <figcaption>
#  </figcaption>
#</figure>



