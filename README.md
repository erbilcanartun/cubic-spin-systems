# Cubic-Spin Systems

This repository contains an implementation of renormalization group (RG) calculations for cubic-spin systems, focusing on phase transitions, and phase diagram calculations, particularly in three spatial dimensions. The models explored include the Ising, Potts, and clock spin models. The system is studied under various configurations, including the presence of antiferromagnetic bonds and external magnetic fields. Additionally, for Ising system, phase diagram calculations, and spin-glass chaos analysis are provided. The spin-glass chaos analysis includes chaotic RG flows, Lyapunov exponent calculations, and runaway exponent calculations.

## Theoretical Background

1. Cubic-Spin Systems

Cubic-spin systems generalize the well-known Ising model to include multiple spin components. In the case of the Ising model, spins are binary ($+1$ (up) or $-1$ (down)), while in the Potts and clock models, spins can take on more states or directions. These models are important for understanding complex ordering phenomena, including:

- Spin-glass phase: Disordered systems where the magnetic moments of atoms or spins freeze into a random state.
- Nematic phase: A liquid-crystal-like phase where spins align along certain axes but remain disordered in other aspects.

There are also several other ordered phases that can be found on the phase diagrams.

2. Hamiltonian

The Hamiltonian governing the cubic-spin Ising system is expressed as:

$$\begin{align}
-\beta \mathcal{H} &= \sum_{\langle ij\rangle} {[ J_{ij}\vec{s_i} \cdot \vec{s_j} + \vec{H}\cdot(\vec{s_i} + \vec{s_j})]}
\end{align}$$

where $J_{ij}$ represents the interaction between neighboring spins (either ferromagnetic or antiferromagnetic), $\vec{s_i}$ is a vector in spin space that takes on $2n$ different states $\pm\hat{u}$ at each site $i$, $\hat{u}$ being orthogonal unit vectors in spin space. $\vec{H}$ is the $n'$-component uniform external magnetic field is
$\vec{H}= H (\hat{u_1} +...+ \hat{u_{n'}})$, with of course $n′\leq n$. The sum is over nearest-neighbor pairs of site $\langle ij\rangle$, and the system is treated using renormalization-group theory on hierarchical lattices. The
interaction $J_{ij}$ is ferromagnetic $+J > 0$ or antiferromagnetic
$-J$ with probabilities $1−p$ and $p$, respectively.

3. Renormalization Group Method

This approach involves rescaling the system, reducing the number of spins, and calculating new effective interactions at larger length scales. The RG flows provide insight into the stability and transitions between different phases, such as spin-glass, nematic, and disordered phases. The critical properties of the system, including phase boundaries and chaotic behavior, are derived from this method.

4. Phases and Phenomena

- Spin-glass chaos: Spin glasses exhibit chaos under scale change, which can be quantified through the Lyapunov exponent. This repository includes calculations for the Lyapunov exponent, which measures the sensitivity to initial conditions.
- Nematic phase: For cubic Ising systems with $n \geq 3$ components, a nematic phase emerges between the low-temperature spin-glass phase and the high-temperature disordered phase.
- Phase diagram topologies: External fields (axial, planar-diagonal, body-diagonal) result in a variety of ordered phases. In some cases, the spin-glass phase is replaced by nematic or other complex orderings under field influence.

## Research Papers

This repository is based on results from the following research papers:

1. [Nematic phase of the n-component cubic-spin spin glass in d = 3: Liquid-crystal phase in a dirty magnet](https://www.sciencedirect.com/science/article/abs/pii/S0378437124002188?via%3Dihub). This paper discusses the emergence of the nematic phase in cubic-spin spin-glass systems and chaotic RG flows.
1. [Axial, planar-diagonal, body-diagonal fields on the cubic-spin spin glass in 𝑑=3: A plethora of ordered phases under finite fields](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.110.034123). This paper examines the effects of different uniform magnetic fields on the cubic-spin systems, revealing a plethora of ordered phases and novel phase diagram topologies.

## Repository Contents

1. `cubic_potts_system.ipynb`: This notebook performs renormalization group calculations for cubic-spin system with Potts spins.
2. `cubic_clock_system.ipynb`: This notebook performs renormalization group calculations for cubic-spin system with clock spins.
3. `cubic_ising_system.ipynb`: This notebook performs renormalization group calculations for cubic-spin system with Ising spins.
4. `cubic_ising_spin-glass_chaos.ipynb`: A set of tools for studying chaos in the spin-glass phase, including: Lyapunov exponent calculations, and runaway exponent calculations for the spin-glass phase, showing the degree of saturation of spin-glass order.
5. `cubic_ising_phase_diagram.ipynb`: A notebook for exploring phases and phase diagrams.

## License

This project is licensed under the GNU Affero General Public License (AGPL) v3 for non-commercial and academic use. For commercial use, a separate license agreement is required.