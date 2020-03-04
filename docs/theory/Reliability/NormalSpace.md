# Multivariate Normal

## Properties of the standard normal space

1. Rotational symmetry wrt origin.

2. Probability density decreases exponentially with the square of the distance in the radial direction from the origin. It follows that for any domain ℱ shown in Figure 6.1a, the point nearest the origin, denoted 𝐮∗, has the highest probability density among all points within ℱ. Furthermore, if ℱ is a failure domain, 𝐮∗ can be considered as the most likely outcome, if failure is to occur.

3. Consider the hyperplane 𝛽−𝛂̂𝐮=0, where 𝛽 is the distance from the origin and 𝛂̂ is the unit normal row vector directed away from the origin (see Figure 6.1b for the two-dimen-sional case). The probability density of points located on this plane decays exponentially with the square of the distance as we move away from the foot of the normal from the origin, 𝐮∗. This has to do with the fact that a set of 𝑛 jointly normal random variables when conditioned on a linear function of any subset of the variables is jointly normal in the 𝑛−1 dimensiona.
4. The probability content of the half space 𝛽−𝛂̂𝐔≤0 is given by
𝑝1=Pr(𝛽−𝜶̂𝐔≤0)=Φ(−𝛽)
 This is evident from the fact that E[𝛽−𝜶̂𝐔]=𝛽 and Var[𝛽−𝜶̂𝐔]=1 and that 𝛽−𝜶̂𝐔 being a linear function of normal random varia-bles is normally distributed. It follows that 𝑝1=Φ[0−𝛽1]=Φ(−𝛽).
1. Consider the parabolic surface 𝛽−𝑢𝑛+12Σ𝜅𝑖𝑢𝑖2=0𝑛−1𝑖=1, where 𝛽 is the distance from the origin to the vertex 𝐮∗ and 𝜅𝑖 are the principal curvatures (see Figure 6.1c for the two-dimensional case). Tvedt (1990) has shown that the probability content of the parabolic domain 𝛽−𝑢𝑛+12Σ𝜅𝑖𝑢𝑖2≤0𝑛−1𝑖=1 is given by
𝑝2=Pr(𝛽−𝑢𝑛+12Σ𝜅𝑖𝑢𝑖2≤0𝑛−1𝑖=1) =𝜙(𝛽)Re{i√2𝜋∫1𝑠exp[(𝑠+𝛽)22]Π1√1+𝜅𝑖𝑠𝑑𝑠𝑛−1𝑖=1i∞0}
(6.3)
where 𝜙(.) denotes the standard normal PDF, i denotes the imaginary unit and the inte-gral inside brackets is computed along the imaginary axis. Breitung (1984) derived the following approximation of the above probability
3
𝑝2≅Φ(−𝛽)Π1√1+𝜓(𝛽)𝜅𝑖𝑛−1𝑖=1
where 𝜓(𝛽)=𝜙(𝛽)/Φ(−𝛽). One can show that 𝜓(𝛽)≅𝛽 so that a simpler approxima-tion is 
𝑝2≅Φ(−𝛽)Π1√1+𝛽𝜅𝑖𝑛−1𝑖=1
The above two approximations work well as long as −1<𝜓(𝛽)𝜅𝑖 and −1<𝛽𝜅𝑖, respectively, for each 𝑖.
1. Consider the hypersphere 𝛽2−Σ𝑢𝑖2=0𝑛𝑖=1 of radius 𝛽. Owing to the fact that the sum of 𝑛 squared standard normal random variables has the chi-square distribution with 𝑛 degrees of freedom, the probability content outside the hypersphere (see Figure 6.1d for the two-dimensional case) is given by