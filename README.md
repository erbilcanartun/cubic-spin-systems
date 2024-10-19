# Cubic-Spin Systems: Ising, Potts, and Clock Spin Models

This repository contains a set of Python codes for simulating and analyzing cubic-spin systems, particularly in three spatial dimensions. The models explored include the Ising, Potts, and clock spin models, and the results are derived using renormalization group (RG) theory. The system is studied under various configurations, including the presence of antiferromagnetic bonds and external magnetic fields. Additionally, for Ising system, phase diagram calculations, and spin-glass chaos analysis are provided. The spin-glass chaos analysis includes chaotic RG flows, Lyapunov exponent calculations, and runaway exponent calculations.

## Theoretical Background

1. Cubic-Spin Systems
Cubic-spin systems generalize the well-known Ising model to include multiple spin components. In the case of the Ising model, spins are binary (±1), while in the Potts and clock models, spins can take on more states or directions. These models are important for understanding complex ordering phenomena, including:

- Spin-glass phase: Disordered systems where the magnetic moments of atoms or spins freeze into a random state.
- Nematic phase: A liquid-crystal-like phase where spins align along certain axes but remain disordered in other aspects.

2. Hamiltonian
The Hamiltonian governing the cubic-spin systems is expressed as:

[Hamiltonian]

where:
- J_{ij} represents the interaction between neighboring spins (either ferromagnetic or antiferromagnetic).
- s_i is a vector in spin space that takes on different states depending on the model (Ising, Potts, or clock).
- \H is an external magnetic field (for models under field influence). The sum is over nearest-neighbor pairs, and the system is treated using renormalization-group theory on hierarchical lattices.

3. Renormalization Group (RG) Method
This approach involves rescaling the system, reducing the number of spins, and calculating new effective interactions at larger length scales. The RG flows provide insight into the stability and transitions between different phases, such as spin-glass, nematic, and disordered phases. The critical properties of the system, including phase boundaries and chaotic behavior, are derived from this method.

4. Phases and Phenomena
- Spin-glass chaos: Spin glasses exhibit chaos under scale change, which can be quantified through the Lyapunov exponent. This repository includes calculations for the Lyapunov exponent, which measures the sensitivity to initial conditions.
- Nematic phase: For systems with n≥3 components, a nematic phase emerges between the low-temperature spin-glass phase and the high-temperature disordered phase
- Phase diagram topologies: External fields (axial, planar-diagonal, body-diagonal) result in a variety of ordered phases. In some cases, the spin-glass phase is replaced by nematic or other complex orderings under field influence.

# Research Papers

This repository is based on results from the following research papers:

1. [Nematic phase of the n-component cubic-spin spin glass in d = 3: Liquid-crystal phase in a dirty magnet](https://www.sciencedirect.com/science/article/abs/pii/S0378437124002188?via%3Dihub). This paper discusses the emergence of the nematic phase in cubic-spin spin-glass systems and chaotic RG flows.
1. [Axial, planar-diagonal, body-diagonal fields on the cubic-spin spin glass in 𝑑=3: A plethora of ordered phases under finite fields](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.110.034123). This paper examines the effects of different uniform magnetic fields on the cubic-spin systems, revealing a plethora of ordered phases and novel phase diagram topologies.

# Repository Contents

1. cubic_potts_system.ipynb: This script performs renormalization group calculations for cubic-Potts system.
2. cubic_clock_system.ipynb: This script performs renormalization group calculations for cubic-clock system.
3. cubic_ising_system.ipynb: This script performs renormalization group calculations for cubic-Ising system.
4. cubic_ising_spin-glass_chaos.ipynb: A comprehensive set of tools for studying chaos in the spin-glass phase, including: Lyapunov exponent calculations, and runaway exponent calculations for the spin-glass phase, showing the degree of saturation of spin-glass order.
5. cubic_ising_phase_diagram.ipynb: A notebook for exploring phases and phase diagrams.

## License

This project is licensed under the GNU Affero General Public License (AGPL) v3 for non-commercial and academic use. For commercial use, a separate license agreement is required.



