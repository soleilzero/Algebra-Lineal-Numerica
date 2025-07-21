### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ bf0426ff-6f2c-4bb7-b3bb-0017d99e81c7
using LinearAlgebra

# ╔═╡ 9c09483c-6619-11f0-339c-3132d4ffd563
md"""
# Cálculo Numérico Ingenuo de la SVD

## Índice
1. Objetivos
1. Introducción
1. Fundamentos teóricos
3. Algoritmos

## Objetivos

**General**

Comprender, implementar y analizar el algoritmo numérico ingenuo para calcular la descomposición en valores singulares (SVD).

**Específicos**
* Exponer las conexiones teóricas entre la SVD y los problemas de autovalores simétricos,
* implementar el algoritmo utilizando los algoritmos QR de las notas de clase,
* analizar el comportamiento numérico del algoritmo clásico frente a métodos mal condicionados,
* discutir posibles estrategias prácticas.

En otro trabajo se puede aplicar la SVD computada numéricamente a un problema práctico.
"""

# ╔═╡ 71039e0e-48ff-4953-b400-ce22494ff56a
md"
## Introducción 

La descomposición en valores singulares (SVD) es una de las herramientas más poderosas y versátiles del álgebra lineal numérica.
No solo existe para cualquier matriz y proporciona información estructural sobre ella —como su rango, condición y modos principales de acción—, sino que también constituye la base de aplicaciones fundamentales en análisis de datos, compresión, métodos de mínimos cuadrados, y problemas inversos.

"

# ╔═╡ 69ad41a3-b775-4168-a3c7-218d45eaa0d5
md"## Teoría"

# ╔═╡ 8c4925b3-70c8-4d63-9140-1ba1a541f14d
md"## Implementación"

# ╔═╡ eccf8824-6d32-491d-9391-9b60e20b14f2
begin
	A=rand(5,5)
	C=A'*A
	display(A)
	epsilon=1E-10
end

# ╔═╡ 11f75b56-81b0-4ea8-9689-4721e2c29a34
struct SVDReconstruction
    U::Matrix{Float64}
    S::Vector{Float64}
    V::Matrix{Float64}
end


# ╔═╡ f479bb41-5a99-424c-9aab-121ef5b6b51e
"""
Reconstruye la matriz original A a partir de su descomposición SVD.
Recibe un objeto F retornado por la función svd(A).
Devuelve A ≈ U * Σ * Vᵀ
"""
function reconstruct_from_svd(F)
    m, n = size(F.U, 1), size(F.V, 1)
    Σ = zeros(m, n)
    for i in 1:length(F.S)
        Σ[i, i] = F.S[i]
    end
    return F.U * Σ * F.V'
end

# ╔═╡ c4c38a38-9298-4cda-998e-03f4610cd33a
"""
Valida una descomposición SVD comparando la matriz original A
con su reconstrucción desde la tupla o estructura SVD `F`.

Imprime la norma del error ‖A - UΣVᵀ‖₂.
"""
function validate_svd(A::Matrix{Float64}, F)
    A_hat = reconstruct_from_svd(F)
    error = norm(A - A_hat)
    println("Error de reconstrucción ‖A - UΣVᵀ‖₂ = ", error)
	display(A)
	display(A_hat)
    return error
end


# ╔═╡ 7066f320-5339-40db-bdda-1e1c396e6fc1
md"### SVD usando un QR de Julia"

# ╔═╡ 293b240e-fd3b-48e4-ba3c-d5e337fd4d2f
"""
Calcula la SVD de A utilizando el enfoque clásico basado en formar C = AᵀA.
Paso 1: Formar C = AᵀA
Paso 2: Calcular la descomposición espectral de C = VΛVᵀ usando QR simétrico
Paso 3: Calcular AV = QR para obtener U
Devuelve U, Σ, Vᵀ
"""
function svd_via_ata(A::Matrix{Float64})
    m, n = size(A)

    # Step 1: Form the symmetric matrix C = AᵀA
    C = A' * A

    # Step 2: Compute eigen-decomposition of C = VΛVᵀ
    # This corresponds to applying the symmetric QR algorithm to C
    quadratic_eigenvalues, V = eigen(Symmetric(C))
    eigenvalues = sqrt.(clamp.(quadratic_eigenvalues, 0.0, Inf))

    # Step 3: Compute AV = QR to obtain U
    AV = A * V
    Q, _ = qr(AV)  # QR
    U = collect(Q) # Get full Q

    return SVDReconstruction(U, eigenvalues, V)
end

# ╔═╡ 59e48f61-049f-4708-a474-293c289f4539
begin
	F2 = svd_via_ata(A)
	reconstruct_from_svd(F2)
	validate_svd(A,F2)
end

# ╔═╡ d0cfe9c6-04ba-47af-b466-3763cbae0364
md"
### Usando un QR propio
"

# ╔═╡ 36cc1bce-5264-4f14-8e79-ab83320e083d
md"
De acuerdo con el libro el pseudocódigo es el siguiente:
1. Form $C=A^TA$
2. Use the symmetric QR algorithm to compute $V^TCV=diag(o²)$
3. Apply QR with column pivoting to $AV$ to obtain $U^T (AV)N =R$
"

# ╔═╡ 921e6808-b9e7-493f-8042-6f8e5419a117
md"#### Paso 2: Descomposición de Schur"

# ╔═╡ aae50fe4-6d55-4e6e-803c-1dde61f1947f
md"Utilizamos la descomposición de Schur para obtener los autovalores de $C$, veamos que funciona en un ejemplo:"

# ╔═╡ e2df08c4-245e-48d8-b680-539d9b12adda
md"Ahora veamos el código:"

# ╔═╡ 9745ad15-080c-493e-9c5e-3a88aebf1068
function Housev(x)
    n = length(x)
    v = ones(size(x))
    v[2:n] = x[2:n]
    σ = norm(x[2:n])^2
    if σ == 0
        β = 0
    else 
        μ = √(x[1]^2+σ)
        if x[1] ≤ 0
            v[1] = x[1] - μ
        else
            v[1] = -σ/(x[1]+μ)
        end
        β = 2*v[1]^2/(σ+v[1]^2)
        v = v/(v[1])
    end
    return v, β
end

# ╔═╡ b32ef216-4e64-44aa-9e84-1bedbccabac3
begin
	function HessenbergForm(A)
    n = size(A)[1]
    H = copy(A)
    U = Matrix(1.0*I, n, n)
    for k = 1:n-2
        v, β = Housev(H[k+1:n, k])
        H[k+1:n, k:n] = (I - β*v*v')*H[k+1:n, k:n]
        H[1:n, k+1:n] = H[1:n, k+1:n]*(I - β*v*v')
        
        #U es necesaria para la verificar que la función devuelve los resultados correctos
        U[1:n, k+1:n] = U[1:n, k+1:n]*(I - β*v*v') 
    end
    return H, U
end
end

# ╔═╡ 5a22a26a-1e35-4200-b5b1-4462e4878b74
function Givens(a,b)
    if b==0
        c = 1
        s = 0
    else
        if abs(b)>abs(a)
            τ=-a/b
            s=-1/sqrt(1+τ^2)
            c=s*τ
        else
            τ=-b/a
            c=1/sqrt(1+τ^2)
            s=c*τ
        end
    end
    return c,s
end

# ╔═╡ 9affa8ea-51cd-4a7b-95e4-e79eedd291fa
begin
	function HessenbergQR(H)
    n = size(H)[1]
    H2 = copy(H)
    C, S = zeros(n-1), zeros(n-1)
    
    #Factorización QR de H
    for k = 1:n-1
        C[k], S[k] = Givens(H2[k,k], H2[k+1, k])
        H2[k:k+1,k:n] = [C[k] -S[k]; S[k] C[k]]*H2[k:k+1,k:n]
        H2[k+1,k] = 0
        #display([C[k] S[k]])
    end
    
    #Matriz RQ
    for k = 1:n-1
        H2[1:k+1, k:k+1] = H2[1:k+1, k:k+1]*[C[k] S[k]; -S[k] C[k]]
    end
    
    return H2 #RQ Hessenberg superior
end
end

# ╔═╡ f2647a91-cb20-4740-97ef-dcb6efc38e88
function RealSchur(A, iteraciones = 10000)
    H0 = A
    H1 = HessenbergForm(A)[1]
    δ = 10
    for k = 1:iteraciones
        H0 = H1
        H1 = HessenbergQR(H1)
    end
    return H1
    
end

# ╔═╡ e16a2c37-32fa-4476-a6c2-69889d532e16
RealSchur(C)

# ╔═╡ 21095a5d-1f28-4c4c-a134-e2a83e20573d
begin
	println("Autovalores de C:")
	display(eigen(Symmetric(C)).values)
	H_2=RealSchur(C)
	println("\nAutovalores por descomposición de Schur:")
	display(Diagonal(H_2))
end

# ╔═╡ 17ff8c7b-b0aa-4490-8d53-20eb82e9764d
md"### Paso 3: Apply QR with column pivoting to $AV$ to obtain $U^T (AV)N =R$ "

# ╔═╡ ce61ed2c-354e-461b-b654-aaf02202fc7e
md"### Final"

# ╔═╡ 596ea23f-236c-4abf-8c06-1e3682bc94dd
function naiveSVD_classic(A::Matrix{Float64})
    C = transpose(A) * A  # Paso 1: matriz simétrica

    T, V = Schur(C)  # Paso 2: tu algoritmo Schur (QR iterado con Gram-Schmidt)

    λ = diag(T)
    σ = sqrt.(abs.(λ))  # valores singulares

    # Ordenar por valores singulares decrecientes
    orden = sortperm(σ, rev=true)
    σ = σ[orden]
    V = V[:, orden]

    # Paso 3: calcular U = AV / σ
    m, n = size(A)
    U = zeros(m, n)
    for i in 1:n
        vi = V[:, i]
        Avi = A * vi
        normi = norm(Avi)
        if normi > 1e-10
            U[:, i] = Avi / normi
        end
    end

    return U, σ, V
end


# ╔═╡ 79b9956c-b3b5-4747-b919-dd4b9fc604bb
md"
**To Do:**

utilizar ortogonalización e iteración QR propio (en classroom - matrices simétricas)
dos formas: con A_TA y la otra, por bloques 2x2 con primer bloque siendo [0 A // A^T 0]
en ambas calcular los valores propios de la matriz
- Reemplazar `eigenvalues` por qr propio
- Reemplazar `Q, _ = qr(AV)` por qr propio
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "ac1187e548c6ab173ac57d4e72da1620216bce54"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"
"""

# ╔═╡ Cell order:
# ╟─9c09483c-6619-11f0-339c-3132d4ffd563
# ╟─71039e0e-48ff-4953-b400-ce22494ff56a
# ╠═69ad41a3-b775-4168-a3c7-218d45eaa0d5
# ╟─8c4925b3-70c8-4d63-9140-1ba1a541f14d
# ╠═bf0426ff-6f2c-4bb7-b3bb-0017d99e81c7
# ╠═eccf8824-6d32-491d-9391-9b60e20b14f2
# ╠═11f75b56-81b0-4ea8-9689-4721e2c29a34
# ╠═f479bb41-5a99-424c-9aab-121ef5b6b51e
# ╠═c4c38a38-9298-4cda-998e-03f4610cd33a
# ╟─7066f320-5339-40db-bdda-1e1c396e6fc1
# ╠═293b240e-fd3b-48e4-ba3c-d5e337fd4d2f
# ╠═59e48f61-049f-4708-a474-293c289f4539
# ╟─d0cfe9c6-04ba-47af-b466-3763cbae0364
# ╠═36cc1bce-5264-4f14-8e79-ab83320e083d
# ╟─921e6808-b9e7-493f-8042-6f8e5419a117
# ╟─aae50fe4-6d55-4e6e-803c-1dde61f1947f
# ╠═e16a2c37-32fa-4476-a6c2-69889d532e16
# ╠═21095a5d-1f28-4c4c-a134-e2a83e20573d
# ╟─e2df08c4-245e-48d8-b680-539d9b12adda
# ╠═f2647a91-cb20-4740-97ef-dcb6efc38e88
# ╠═b32ef216-4e64-44aa-9e84-1bedbccabac3
# ╠═9745ad15-080c-493e-9c5e-3a88aebf1068
# ╠═5a22a26a-1e35-4200-b5b1-4462e4878b74
# ╠═9affa8ea-51cd-4a7b-95e4-e79eedd291fa
# ╠═17ff8c7b-b0aa-4490-8d53-20eb82e9764d
# ╟─ce61ed2c-354e-461b-b654-aaf02202fc7e
# ╠═596ea23f-236c-4abf-8c06-1e3682bc94dd
# ╠═79b9956c-b3b5-4747-b919-dd4b9fc604bb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
