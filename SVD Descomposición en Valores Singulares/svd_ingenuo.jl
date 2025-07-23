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

# ╔═╡ b250760b-efa2-46df-a1e4-656547a6d3d9
md"""
## Reflexión y aprendizajes

El principal aprendizaje que me dejó este notebook fue la importancia de la comunicación, dado que inicialmente no tenía claras las instrucciones y estaba haciendo un trabajo innecesario.

## Declaración sobre el uso de inteligencia artificial y fuentes externas

Durante la realización de este trabajo se utilizó **asistencia de inteligencia artificial (IA)** para generar y mejorar partes del código y de los textos explicativos, con el fin de estructurar mejor el contenido y optimizar la implementación.

### Uso de IA:

Se utilizó **ChatGPT (OpenAI)** de forma activa para la generación y refactorización de código en Julia.

**Algunos prompts utilizados:**

* "¿A qué se debe este error: `MethodError: no method matching iterate(::Main.var"workspace#25".SVDReconstruction)`? "

---

### Fuentes externas consultadas:

* Golub, G., & Van Loan, C. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.

"""

# ╔═╡ 69ad41a3-b775-4168-a3c7-218d45eaa0d5
md"## Teoría"

# ╔═╡ 8c4925b3-70c8-4d63-9140-1ba1a541f14d
md"## Implementación"

# ╔═╡ 4c28ca44-c61b-4db4-87e1-eb00fb467246
md"### Set up"

# ╔═╡ 11f75b56-81b0-4ea8-9689-4721e2c29a34
struct SVDReconstruction
    U::Matrix{Float64}
    S::Vector{Float64}
    V::Matrix{Float64}
end


# ╔═╡ 41c1603c-6101-4bee-b66a-3f88dacce14d
md"### SVD de Julia"

# ╔═╡ a4858212-5dd0-406a-ac48-314dbb539517
svd

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

# ╔═╡ d0cfe9c6-04ba-47af-b466-3763cbae0364
md"
### Método ingenuo

De acuerdo con el libro el pseudocódigo es el siguiente:
1. Form $C=A^TA$
2. Use the symmetric QR algorithm to compute $V^TCV=diag(o²)$
3. Apply QR with column pivoting to $AV$ to obtain $U^T (AV)N =R$
"

# ╔═╡ 921e6808-b9e7-493f-8042-6f8e5419a117
md"
#### Paso 2: Obtención de autovalores y autovectores
Para obtener los autovalores y los autovectores ($V$) de $C$ utilizamos la descomposición de Schur. Utilizaremos una adaptación del método `RealSchur` del cuaderno `SchurFinal.jl`.

La adaptación es para guardar también `Q` en `HessenbergQR` y `RealSchur`.

Ahora, comparemos el método propio de Julia `eigen` con la función `RealSchur` para autovalores y autovectores.

**Autovalores:**
"

# ╔═╡ 29ede2b0-76b4-4d4c-be37-cd74dc285ec9
md"**Autovectores:**"

# ╔═╡ 70737453-1dd3-4832-8dcc-02924df3ee48
md"
Podemos ver que ambos métodos coinciden.
Ahora veamos el código de `Real_Schur`, junto con las funciones auxiliares `HessenbergForm` y `HessenbergQR` que transforman la matriz en una matriz Hessenberg y luego aplican la descomposición QR.
"

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
end

# ╔═╡ a66ef3b2-602e-4bf6-801f-b44e6af58404
begin
	function HessenbergQR(H,Q)
		n = size(H)[1]
		H2 = copy(H)
		C, S = zeros(n-1), zeros(n-1)
		
		# Factorización QR de H
		# Q^T*H o Givens por la izquierda
		for k = 1:n-1
			C[k], S[k] = Givens(H2[k,k], H2[k+1, k])
			H2[k:k+1,k:n] = [C[k] -S[k]; S[k] C[k]]*H2[k:k+1,k:n]
			H2[k+1,k] = 0
			Q[:, k:k+1] = Q[:, k:k+1] * [C[k] S[k]; -S[k] C[k]]  # actualizar Q
		end
		
		# Matriz RQ
		# H*Q o Givens por la derecha
		for k = 1:n-1
			H2[1:k+1, k:k+1] = H2[1:k+1, k:k+1]*[C[k] S[k]; -S[k] C[k]]
		end
		
		return H2,Q
	end
	
	begin
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
	end
end

# ╔═╡ f2647a91-cb20-4740-97ef-dcb6efc38e88
function RealSchur(A, iteraciones = 10000)
    H0 = A
    H1, Q = HessenbergForm(A)
    δ = 10
    for _ = 1:iteraciones
        H0 = H1
        H1,Q = HessenbergQR(H1,Q)
    end
    return H1, Q
    
end

# ╔═╡ 17ff8c7b-b0aa-4490-8d53-20eb82e9764d
md"
#### Paso 3: Obtener $U$
Aplicamos el algoritmo QR con pivoteo de columna a $AV$ para obtener $U^T (AV)N =R$.
"

# ╔═╡ de232181-3366-4949-a7a4-6dabbcc10cc2
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

# ╔═╡ ef1c9edb-f3e8-4972-8d8a-8cefa57dd544
function naiveSVD_classic_(A::Matrix{Float64})
	#Paso 1
	C = transpose(A) * A
	
	#Paso 2
	T, V = RealSchur(C)
	λ = diag(T)
	σ = sqrt.(abs.(λ))

	#Reorden
	orden = sortperm(σ, rev=true)
	σ = σ[orden]
	V = V[:, orden]

	#Paso 3
	AV = A * V
	U, _, _ = QRPivotado_(AV)
	return SVDReconstruction(U, σ, V)
end

# ╔═╡ b69f8c42-91a2-4b99-b180-b6481c30a6c6
md"### Evaluación
#### Funciones de evaluación
"

# ╔═╡ ba01fa78-0585-4deb-924b-1b815a9e32c0
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

# ╔═╡ adfcb17d-2de1-4581-9495-fed1359d8a1f
"""
Valida una descomposición SVD comparando la matriz original A
con su reconstrucción desde la tupla o estructura SVD `F`.

Imprime la norma del error ‖A - UΣVᵀ‖₂.
"""
function validate_svd(A::Matrix{Float64}, F; V_transposed=false)
    A_hat = reconstruct_from_svd(F;V_transposed=V_transposed)
    error = norm(A - A_hat)
	display(A_hat)
    println("Error de reconstrucción ‖A - UΣVᵀ‖₂ = ", error)

    return error
end

# ╔═╡ de203fde-bd13-4b46-a0cb-8af4407ef83f
md"
### Ejemplos
Veamos el error de reconstrucción para cada método. Podemos ver que para `svd` y para `naiveSVD_classic_` el error corresponde al error de máquina, lo cual es un buen signo.
"

# ╔═╡ ecf2900a-b0f7-4a65-bc20-0b0671ef30ad
begin
	A=rand(5,5)
	C=A'*A
	display(A)
end

# ╔═╡ 59e48f61-049f-4708-a474-293c289f4539
begin
	F2 = svd_via_ata(A)
	validate_svd(A,F2)
end

# ╔═╡ ce85b232-ec3d-4d1c-84d5-420f9911ac24
sort(eigen(C).values, rev=true)

# ╔═╡ e16a2c37-32fa-4476-a6c2-69889d532e16
Diagonal(RealSchur(C)[1])

# ╔═╡ b88b2e64-6c79-461c-a89e-ee816bd4a2ba
eigen(Symmetric(C)).vectors

# ╔═╡ 240d8bff-c900-4126-b906-3d7d7b9d0e56
RealSchur(C)[2]

# ╔═╡ 4806679c-71bf-431a-9464-93689e696a35
begin
	s1=svd(A)
	validate_svd(A, s1)
end

# ╔═╡ cb6b4738-a3c5-413a-9a28-108af85fb249
begin
	s2=svd_via_ata(A)
	validate_svd(A, s2)
end

# ╔═╡ 3abe59da-bf9d-4703-b584-d378dcbec887
begin
	s3=naiveSVD_classic_(A)
	validate_svd(A, s3)
end

# ╔═╡ 60d6c2cb-56c5-47b9-9959-7caf72d214e5
md"Y para matrices grandes, toma bastante tiempo"

# ╔═╡ 82d13c2a-f64d-447a-9b4b-371f2d3d50c2
begin
	D=rand(500,500)
	#s5=naiveSVD_classic_(D)
end

# ╔═╡ d1b359b3-bf0e-47a2-893e-6de182767fc0
svd(D)

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
# ╟─69ad41a3-b775-4168-a3c7-218d45eaa0d5
# ╟─8c4925b3-70c8-4d63-9140-1ba1a541f14d
# ╟─4c28ca44-c61b-4db4-87e1-eb00fb467246
# ╠═bf0426ff-6f2c-4bb7-b3bb-0017d99e81c7
# ╠═11f75b56-81b0-4ea8-9689-4721e2c29a34
# ╟─41c1603c-6101-4bee-b66a-3f88dacce14d
# ╠═a4858212-5dd0-406a-ac48-314dbb539517
# ╟─7066f320-5339-40db-bdda-1e1c396e6fc1
# ╠═293b240e-fd3b-48e4-ba3c-d5e337fd4d2f
# ╠═59e48f61-049f-4708-a474-293c289f4539
# ╟─d0cfe9c6-04ba-47af-b466-3763cbae0364
# ╠═ef1c9edb-f3e8-4972-8d8a-8cefa57dd544
# ╟─921e6808-b9e7-493f-8042-6f8e5419a117
# ╠═ce85b232-ec3d-4d1c-84d5-420f9911ac24
# ╠═e16a2c37-32fa-4476-a6c2-69889d532e16
# ╟─29ede2b0-76b4-4d4c-be37-cd74dc285ec9
# ╠═b88b2e64-6c79-461c-a89e-ee816bd4a2ba
# ╠═240d8bff-c900-4126-b906-3d7d7b9d0e56
# ╟─70737453-1dd3-4832-8dcc-02924df3ee48
# ╠═f2647a91-cb20-4740-97ef-dcb6efc38e88
# ╠═b32ef216-4e64-44aa-9e84-1bedbccabac3
# ╠═a66ef3b2-602e-4bf6-801f-b44e6af58404
# ╟─17ff8c7b-b0aa-4490-8d53-20eb82e9764d
# ╠═de232181-3366-4949-a7a4-6dabbcc10cc2
# ╠═b69f8c42-91a2-4b99-b180-b6481c30a6c6
# ╠═ba01fa78-0585-4deb-924b-1b815a9e32c0
# ╠═adfcb17d-2de1-4581-9495-fed1359d8a1f
# ╟─de203fde-bd13-4b46-a0cb-8af4407ef83f
# ╠═ecf2900a-b0f7-4a65-bc20-0b0671ef30ad
# ╠═4806679c-71bf-431a-9464-93689e696a35
# ╠═cb6b4738-a3c5-413a-9a28-108af85fb249
# ╠═3abe59da-bf9d-4703-b584-d378dcbec887
# ╠═60d6c2cb-56c5-47b9-9959-7caf72d214e5
# ╠═d1b359b3-bf0e-47a2-893e-6de182767fc0
# ╠═82d13c2a-f64d-447a-9b4b-371f2d3d50c2
# ╟─b250760b-efa2-46df-a1e4-656547a6d3d9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
