### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ c478a7cc-42b0-11f0-1c45-919167ce835a
md"
# The shifted QR iteration
"

# ╔═╡ 17dbbafe-fe46-4001-a1a0-5636395e89d8
md"
Plan:
1. Introducción
 * Nociones
 * Motivación
2. Teoría
 * Recap of eigenvalue problems.
 * The basic QR algorithm.
 * The shifted QR method (Wilkinson shift, etc.).
 * Convergence behavior and complexity.
3. Implementación

 * Code for the unshifted QR algorithm (for comparison).

 * Code for the shifted QR algorithm.

 * Optional: Use of Hessenberg reduction to optimize performance.
4. Experimentos
5. Análisis
"

# ╔═╡ 510c043e-a9dc-48ed-a91d-0821231c93d5


# ╔═╡ Cell order:
# ╠═c478a7cc-42b0-11f0-1c45-919167ce835a
# ╠═17dbbafe-fe46-4001-a1a0-5636395e89d8
# ╠═510c043e-a9dc-48ed-a91d-0821231c93d5
