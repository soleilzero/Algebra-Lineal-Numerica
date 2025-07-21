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
md"## Implementación del método ingenuo"

# ╔═╡ 36cc1bce-5264-4f14-8e79-ab83320e083d
md"
De acuerdo con el libro el pseudocódigo es el siguiente:
1. Form $C=A^TA$
2. Use the symmetric QR algorithm to compute $V^TCV=diag(o²)$
3. Apply QR with column pivoting to $AV$ to obtain $U^T (AV)N =R$
"

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


# ╔═╡ 88697440-08ac-4748-8224-863ede4a8a22
function relative_frobenius_error(A, Â)
    return norm(A - Â, Fro) / norm(A, Fro)
end


# ╔═╡ f479bb41-5a99-424c-9aab-121ef5b6b51e
"""
Reconstruye la matriz original A a partir de su descomposición SVD.
Recibe un objeto F retornado por la función svd(A).
Devuelve A ≈ U * Σ * Vᵀ
"""
function reconstruct_from_svd(F; V_transposed=false)
    n = size(F.S, 1)
    Σ = zeros(n, n)
    for i in 1:length(F.S)
        Σ[i, i] = F.S[i]
    end
	if V_transposed
    	return F.U * Σ * F.V
	else
		return F.U * Σ * F.V'
	end
end

# ╔═╡ c4c38a38-9298-4cda-998e-03f4610cd33a
"""
Valida una descomposición SVD comparando la matriz original A
con su reconstrucción desde la tupla o estructura SVD `F`.

Imprime la norma del error ‖A - UΣVᵀ‖₂.
"""
function validate_svd(A::Matrix{Float64}, F; V_transposed=false)
    A_hat = reconstruct_from_svd(F;V_transposed=V_transposed)
    error = norm(A - A_hat)
    println("Error de reconstrucción ‖A - UΣVᵀ‖₂ = ", error)
	#println("Error de Frobenius ‖A - UΣVᵀ‖₂ = ", relative_frobenius_error(A,A_hat))

    return error
end


# ╔═╡ bccf578b-da57-4df2-b5cb-cf5a95c80ff4
md"Hay un error en la validación del SVD. Funciona mejor con el V_transposed contrario. Igualmente los errores son grandes.
Idea: usar error contra svd normal
"

# ╔═╡ 33b3d729-7315-4c60-9854-48d9a246cd32
s1=svd(A)

# ╔═╡ 578c849f-0760-41a6-b6f2-0a225afddeb8
validate_svd(A, s1)

# ╔═╡ 8c02d86c-9479-4df8-a94f-8489406a5356
validate_svd(A, s1; V_transposed=true)

# ╔═╡ 41c1603c-6101-4bee-b66a-3f88dacce14d
md"### SVD de Julia"

# ╔═╡ 7618b8dc-1f91-40a3-9ea3-a8248b9efe28
begin
	A1 = svd(A)
	validate_svd(A,A1)
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
    quadratic_σ, V = eigen(Symmetric(C))
    σ = sqrt.(clamp.(quadratic_σ, 0.0, Inf))

	# Orden
	orden = sortperm(σ, rev=true)
    σ = σ[orden]
    V = V[:, orden]

    # Step 3: Compute AV = QR to obtain U
    AV = A * V
    Q, _ = qr(AV)  # QR
    U = Matrix(Q[:, 1:n]) # Get full Q

    return SVDReconstruction(U, σ, V)
end

# ╔═╡ df587928-e939-4d20-a959-eb4128980bed
s2=svd_via_ata(A)

# ╔═╡ 210b22b9-13e6-4d29-a1f4-6d0e85bf45a6
validate_svd(A, s2)

# ╔═╡ 83cb9d8a-b759-473e-9d2e-52b6fb5f2054
validate_svd(A, s2; V_transposed=true)

# ╔═╡ 59e48f61-049f-4708-a474-293c289f4539
begin
	F2 = svd_via_ata(A)
	validate_svd(A,F2)
end

# ╔═╡ d0cfe9c6-04ba-47af-b466-3763cbae0364
md"
### Usando un QR propio
"

# ╔═╡ 921e6808-b9e7-493f-8042-6f8e5419a117
md"#### Paso 2: Descomposición de Schur"

# ╔═╡ aae50fe4-6d55-4e6e-803c-1dde61f1947f
md"Utilizamos la descomposición de Schur para obtener los autovalores y los autovectores ($V$) de $C$, veamos que funciona en un ejemplo:"

# ╔═╡ 9e7998ae-f91b-424b-bca8-05fcd71064ec
md"**Autovalores:**"

# ╔═╡ ce85b232-ec3d-4d1c-84d5-420f9911ac24
eigen(Symmetric(C)).values

# ╔═╡ e8606e61-cbaf-4062-8c4f-abb653737919
md"**Autovectores**"

# ╔═╡ b88b2e64-6c79-461c-a89e-ee816bd4a2ba
eigen(Symmetric(C)).vectors

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

# ╔═╡ 469bb3f0-bdc9-46a5-b3ab-73149ddf9bbd
function RealSchurEig(C::Matrix{Float64}, iteraciones::Int = 10000)
    H, Q = HessenbergForm(C)  # Q inicial de Householder
    n = size(C, 1)

    for k = 1:iteraciones
        Cg, Sg = zeros(n-1), zeros(n-1)
        # Aplicamos rotaciones de Givens
        for i = 1:n-1
            c, s = Givens(H[i, i], H[i+1, i])
            G = [c -s; s c]
            H[i:i+1, i:end] = G * H[i:i+1, i:end]
            H[1:end, i:i+1] = H[1:end, i:i+1] * G'

            # Acumulamos la rotación en Q
            Q[:, i:i+1] = Q[:, i:i+1] * G'
        end
    end

    return H, Q  # H tiene los autovalores, Q tiene los autovectores
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
Diagonal(RealSchur(C))

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

# ╔═╡ 491e85bc-2104-4908-8797-194c89cf1f7f
md"### Versión nueva"

# ╔═╡ b0355544-b0a6-4f00-9381-cdddcc982b91
begin
	function Housev_(x)
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
	
	function HessenbergForm_(A)
	    n = size(A, 1)
	    H = copy(A)
	    Q = Matrix(1.0I, n, n)
	    for k = 1:n-2
	        v, β = Housev(H[k+1:n, k])
	        H[k+1:n, k:n] .= (I - β*v*v') * H[k+1:n, k:n]
	        H[1:n, k+1:n] .= H[1:n, k+1:n] * (I - β*v*v')
	        Q[:, k+1:n] .= Q[:, k+1:n] * (I - β*v*v')
	    end
	    return H, Q
	end
	
	function Givens_(a, b)
	    if b == 0
	        return 1.0, 0.0
	    else
	        if abs(b) > abs(a)
	            τ = -a / b
	            s = 1 / sqrt(1 + τ^2)
	            c = s * τ
	        else
	            τ = -b / a
	            c = 1 / sqrt(1 + τ^2)
	            s = c * τ
	        end
	        return c, s
	    end
	end
	
	function HessenbergQR_(H, Q)
	    n = size(H, 1)
	    for k = 1:n-1
	        c, s = Givens_(H[k, k], H[k+1, k])
	        G = [c -s; s c]
	        H[k:k+1, k:n] = G * H[k:k+1, k:n]
	        H[1:n, k:k+1] = H[1:n, k:k+1] * G'
	        Q[:, k:k+1] = Q[:, k:k+1] * G'
	    end
	    return H, Q
	end
	
	function RealSchur_(A, iteraciones = 10000)
	    H, Q = HessenbergForm_(A)
	    for _ = 1:iteraciones
	        H, Q = HessenbergQR_(H, Q)
	    end
	    return H, Q
	end
end

# ╔═╡ 042dcfd5-d944-417d-a18a-ed982874c185
Diagonal(RealSchur_(C)[1])

# ╔═╡ 21095a5d-1f28-4c4c-a134-e2a83e20573d
RealSchur_(C)[2]

# ╔═╡ cea1de1a-3a88-4eb4-b8fb-ac9e65eb56d6
function QRPivotado_(A::Matrix{Float64})
	    m, n = size(A)
	    Q = zeros(m, n)
	    R = zeros(n, n)
	    P = collect(1:n)
	    A_work = copy(A)
	    norms = [norm(A[:, j]) for j in 1:n]
	
	    for i in 1:n
	        max_col = argmax(norms[i:end]) + i - 1
	        A_work[:, [i, max_col]] = A_work[:, [max_col, i]]
	        P[[i, max_col]] = P[[max_col, i]]
	        norms[[i, max_col]] = norms[[max_col, i]]
	
	        v = A_work[:, i]
	        for j = 1:i-1
	            R[j, i] = dot(Q[:, j], v)
	            v -= R[j, i] * Q[:, j]
	        end
	        R[i, i] = norm(v)
	        Q[:, i] = v / R[i, i]
	    end
	
	    return Q, R, P
	end	

# ╔═╡ dcbd66b1-e3e9-4988-8785-e94219d56e32
function naiveSVD_classic_(A::Matrix{Float64})
	    C = transpose(A) * A
	    T, V = RealSchur_(C)
	    λ = diag(T)
	    σ = sqrt.(abs.(λ))
	    orden = sortperm(σ, rev=true)
	    σ = σ[orden]
	    V = V[:, orden]
	    AV = A * V
	    U, _, _ = QRPivotado_(AV)
	    return SVDReconstruction(U, σ, V)
	end

# ╔═╡ 6f6f3e46-7164-4200-88fb-3a437543986f
s3=naiveSVD_classic_(A)

# ╔═╡ c80c0ecf-1b44-445f-a990-d044e4d4280c
validate_svd(A, s3)

# ╔═╡ 5e2753e3-bb12-442c-992e-0ac11a147f57
validate_svd(A, s3; V_transposed=true)

# ╔═╡ 9779b71a-60cf-4a6d-8337-7d4421a9d8fe
eigen(C)

# ╔═╡ 3c82519a-9c07-49b8-a0b8-ba7d895077af
A2 = naiveSVD_classic_(C)

# ╔═╡ ea15f2f1-82c2-4060-9855-09eb495a56c6
validate_svd(C, A2)

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
# ╟─36cc1bce-5264-4f14-8e79-ab83320e083d
# ╠═bf0426ff-6f2c-4bb7-b3bb-0017d99e81c7
# ╠═eccf8824-6d32-491d-9391-9b60e20b14f2
# ╠═11f75b56-81b0-4ea8-9689-4721e2c29a34
# ╠═88697440-08ac-4748-8224-863ede4a8a22
# ╠═f479bb41-5a99-424c-9aab-121ef5b6b51e
# ╠═c4c38a38-9298-4cda-998e-03f4610cd33a
# ╠═bccf578b-da57-4df2-b5cb-cf5a95c80ff4
# ╠═33b3d729-7315-4c60-9854-48d9a246cd32
# ╠═df587928-e939-4d20-a959-eb4128980bed
# ╠═6f6f3e46-7164-4200-88fb-3a437543986f
# ╠═578c849f-0760-41a6-b6f2-0a225afddeb8
# ╠═8c02d86c-9479-4df8-a94f-8489406a5356
# ╠═210b22b9-13e6-4d29-a1f4-6d0e85bf45a6
# ╠═83cb9d8a-b759-473e-9d2e-52b6fb5f2054
# ╠═c80c0ecf-1b44-445f-a990-d044e4d4280c
# ╠═5e2753e3-bb12-442c-992e-0ac11a147f57
# ╠═41c1603c-6101-4bee-b66a-3f88dacce14d
# ╠═7618b8dc-1f91-40a3-9ea3-a8248b9efe28
# ╟─7066f320-5339-40db-bdda-1e1c396e6fc1
# ╠═293b240e-fd3b-48e4-ba3c-d5e337fd4d2f
# ╠═59e48f61-049f-4708-a474-293c289f4539
# ╟─d0cfe9c6-04ba-47af-b466-3763cbae0364
# ╟─921e6808-b9e7-493f-8042-6f8e5419a117
# ╟─aae50fe4-6d55-4e6e-803c-1dde61f1947f
# ╟─9e7998ae-f91b-424b-bca8-05fcd71064ec
# ╠═042dcfd5-d944-417d-a18a-ed982874c185
# ╠═e16a2c37-32fa-4476-a6c2-69889d532e16
# ╠═ce85b232-ec3d-4d1c-84d5-420f9911ac24
# ╟─e8606e61-cbaf-4062-8c4f-abb653737919
# ╠═21095a5d-1f28-4c4c-a134-e2a83e20573d
# ╠═b88b2e64-6c79-461c-a89e-ee816bd4a2ba
# ╟─e2df08c4-245e-48d8-b680-539d9b12adda
# ╠═469bb3f0-bdc9-46a5-b3ab-73149ddf9bbd
# ╠═f2647a91-cb20-4740-97ef-dcb6efc38e88
# ╠═b32ef216-4e64-44aa-9e84-1bedbccabac3
# ╠═9745ad15-080c-493e-9c5e-3a88aebf1068
# ╠═5a22a26a-1e35-4200-b5b1-4462e4878b74
# ╠═9affa8ea-51cd-4a7b-95e4-e79eedd291fa
# ╠═17ff8c7b-b0aa-4490-8d53-20eb82e9764d
# ╟─ce61ed2c-354e-461b-b654-aaf02202fc7e
# ╠═596ea23f-236c-4abf-8c06-1e3682bc94dd
# ╠═79b9956c-b3b5-4747-b919-dd4b9fc604bb
# ╟─491e85bc-2104-4908-8797-194c89cf1f7f
# ╠═b0355544-b0a6-4f00-9381-cdddcc982b91
# ╠═cea1de1a-3a88-4eb4-b8fb-ac9e65eb56d6
# ╠═dcbd66b1-e3e9-4988-8785-e94219d56e32
# ╠═9779b71a-60cf-4a6d-8337-7d4421a9d8fe
# ╠═3c82519a-9c07-49b8-a0b8-ba7d895077af
# ╠═ea15f2f1-82c2-4060-9855-09eb495a56c6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
