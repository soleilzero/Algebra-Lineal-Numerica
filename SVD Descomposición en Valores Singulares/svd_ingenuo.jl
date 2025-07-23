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
	display(A_hat)
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

# ╔═╡ b3b5bc5b-c93d-4f92-9d89-33165587bf1e
md"
Para el método `svd` de Julia:
el error de reconstrucción teniendo en cuenta que V es no transpuesta corresponde al error de máquina.
"

# ╔═╡ 578c849f-0760-41a6-b6f2-0a225afddeb8
validate_svd(A, s1) #sirve

# ╔═╡ 318ce551-45b6-4d04-ba66-4d622051807d
md"
Para el método `svd_via_ata` en el cual implementados el algoritmo SVD con el algoritmo QR de Julia, el error es de más de una unidad incluso si se tiene en cuenta que V no está transpuesta.
"

# ╔═╡ fa7fafbc-04d3-4c69-9aad-4ee86fcffba6
md"
Para el método `naiveSVD_classic_` en el cual implementados el algoritmo SVD con el algoritmo QR dado en clase, el error es de aproximadamente $5*10^{-1}$ incluso si se tiene en cuenta que V no está transpuesta.
"

# ╔═╡ a0236861-049d-40c3-9069-70029c13bc69
md"Para V_transposed=true ninguna sirve..."

# ╔═╡ 8c02d86c-9479-4df8-a94f-8489406a5356
validate_svd(A, s1; V_transposed=true) #no sirve

# ╔═╡ b460bc19-8641-47ba-89e6-06faf66f53ac
md"Y para matrices grandes, toma bastante tiempo"

# ╔═╡ 41c1603c-6101-4bee-b66a-3f88dacce14d
md"### SVD de Julia"

# ╔═╡ a4858212-5dd0-406a-ac48-314dbb539517
svd

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
md"
**Autovalores:**

Comparemos 3 métodos diferentes para obtener los autovalores: el de Julia, El del notebook `RealSchur` y el adaptado `RealSchur_` para devolver también $V$.

Podemos ver que el original `RealSchur` es mucho más preciso que el adaptado
"

# ╔═╡ ce85b232-ec3d-4d1c-84d5-420f9911ac24
sort(eigen(Symmetric(C)).values, rev=true)

# ╔═╡ e8606e61-cbaf-4062-8c4f-abb653737919
md"
**Autovectores**

Ahora comparemos dos métodos para obtener autovectores: el método propio de Julia con `RealSchur_`. Podemos ver que los resultados no son cercanos
"

# ╔═╡ b88b2e64-6c79-461c-a89e-ee816bd4a2ba
eigen(Symmetric(C)).vectors

# ╔═╡ 1e0bd262-f985-44c9-8e4f-136fc43b6390
md"**HessenbergQR**"

# ╔═╡ e2df08c4-245e-48d8-b680-539d9b12adda
md"Ahora veamos el código:"

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

# ╔═╡ a66ef3b2-602e-4bf6-801f-b44e6af58404
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

# ╔═╡ af05a8ab-8a80-49c3-b5c9-78a80d257490
begin
	H, Q = HessenbergForm(A)
	HessenbergQR(H,Q)
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

# ╔═╡ e16a2c37-32fa-4476-a6c2-69889d532e16
Diagonal(RealSchur(C)[1])

# ╔═╡ 240d8bff-c900-4126-b906-3d7d7b9d0e56
RealSchur(C)[2]

# ╔═╡ 17ff8c7b-b0aa-4490-8d53-20eb82e9764d
md"### Paso 3: Apply QR with column pivoting to $AV$ to obtain $U^T (AV)N =R$ "

# ╔═╡ ce61ed2c-354e-461b-b654-aaf02202fc7e
md"### Final"

# ╔═╡ 79b9956c-b3b5-4747-b919-dd4b9fc604bb
md"
**To Do:**

utilizar ortogonalización e iteración QR propio (en classroom - matrices simétricas)
dos formas: con A_TA y la otra, por bloques 2x2 con primer bloque siendo [0 A // A^T 0]
en ambas calcular los valores propios de la matriz
- Reemplazar `eigenvalues` por qr propio
- Reemplazar `Q, _ = qr(AV)` por qr propio

Opción 1: modificar Schur para que devuelva los vectores
Opción 2: armar V de la otra forma
"

# ╔═╡ 491e85bc-2104-4908-8797-194c89cf1f7f
md"
### Versión nueva
Queremos un Real Schur que devuelva también Q, la única diferencia con el original es que HessenberQR también debe devolver Q.
"

# ╔═╡ f2a6c197-13a8-468e-bb43-61f6010bdae3
md"#### Ahora...."

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
	C = transpose(A) * A
	T, V = RealSchur(C)
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
validate_svd(A, s3) #sirve más o menos

# ╔═╡ 5e2753e3-bb12-442c-992e-0ac11a147f57
validate_svd(A, s3; V_transposed=true)

# ╔═╡ 82d13c2a-f64d-447a-9b4b-371f2d3d50c2
begin
	D=rand(500,500)
	s5=naiveSVD_classic_(D)
end

# ╔═╡ e7a42900-4c04-4baa-94b6-a0a2254f5177
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
# ╟─b250760b-efa2-46df-a1e4-656547a6d3d9
# ╟─69ad41a3-b775-4168-a3c7-218d45eaa0d5
# ╟─8c4925b3-70c8-4d63-9140-1ba1a541f14d
# ╟─36cc1bce-5264-4f14-8e79-ab83320e083d
# ╠═bf0426ff-6f2c-4bb7-b3bb-0017d99e81c7
# ╠═eccf8824-6d32-491d-9391-9b60e20b14f2
# ╠═11f75b56-81b0-4ea8-9689-4721e2c29a34
# ╠═88697440-08ac-4748-8224-863ede4a8a22
# ╠═f479bb41-5a99-424c-9aab-121ef5b6b51e
# ╠═c4c38a38-9298-4cda-998e-03f4610cd33a
# ╟─bccf578b-da57-4df2-b5cb-cf5a95c80ff4
# ╠═33b3d729-7315-4c60-9854-48d9a246cd32
# ╠═df587928-e939-4d20-a959-eb4128980bed
# ╠═6f6f3e46-7164-4200-88fb-3a437543986f
# ╟─b3b5bc5b-c93d-4f92-9d89-33165587bf1e
# ╠═578c849f-0760-41a6-b6f2-0a225afddeb8
# ╟─318ce551-45b6-4d04-ba66-4d622051807d
# ╠═210b22b9-13e6-4d29-a1f4-6d0e85bf45a6
# ╠═fa7fafbc-04d3-4c69-9aad-4ee86fcffba6
# ╠═c80c0ecf-1b44-445f-a990-d044e4d4280c
# ╟─a0236861-049d-40c3-9069-70029c13bc69
# ╠═8c02d86c-9479-4df8-a94f-8489406a5356
# ╠═83cb9d8a-b759-473e-9d2e-52b6fb5f2054
# ╠═5e2753e3-bb12-442c-992e-0ac11a147f57
# ╟─b460bc19-8641-47ba-89e6-06faf66f53ac
# ╠═e7a42900-4c04-4baa-94b6-a0a2254f5177
# ╠═82d13c2a-f64d-447a-9b4b-371f2d3d50c2
# ╟─41c1603c-6101-4bee-b66a-3f88dacce14d
# ╠═a4858212-5dd0-406a-ac48-314dbb539517
# ╠═7618b8dc-1f91-40a3-9ea3-a8248b9efe28
# ╟─7066f320-5339-40db-bdda-1e1c396e6fc1
# ╠═293b240e-fd3b-48e4-ba3c-d5e337fd4d2f
# ╠═59e48f61-049f-4708-a474-293c289f4539
# ╟─d0cfe9c6-04ba-47af-b466-3763cbae0364
# ╟─921e6808-b9e7-493f-8042-6f8e5419a117
# ╟─aae50fe4-6d55-4e6e-803c-1dde61f1947f
# ╟─9e7998ae-f91b-424b-bca8-05fcd71064ec
# ╠═ce85b232-ec3d-4d1c-84d5-420f9911ac24
# ╠═e16a2c37-32fa-4476-a6c2-69889d532e16
# ╟─e8606e61-cbaf-4062-8c4f-abb653737919
# ╠═b88b2e64-6c79-461c-a89e-ee816bd4a2ba
# ╠═240d8bff-c900-4126-b906-3d7d7b9d0e56
# ╟─1e0bd262-f985-44c9-8e4f-136fc43b6390
# ╠═af05a8ab-8a80-49c3-b5c9-78a80d257490
# ╠═a66ef3b2-602e-4bf6-801f-b44e6af58404
# ╟─e2df08c4-245e-48d8-b680-539d9b12adda
# ╠═f2647a91-cb20-4740-97ef-dcb6efc38e88
# ╠═b32ef216-4e64-44aa-9e84-1bedbccabac3
# ╟─17ff8c7b-b0aa-4490-8d53-20eb82e9764d
# ╟─ce61ed2c-354e-461b-b654-aaf02202fc7e
# ╟─79b9956c-b3b5-4747-b919-dd4b9fc604bb
# ╟─491e85bc-2104-4908-8797-194c89cf1f7f
# ╟─f2a6c197-13a8-468e-bb43-61f6010bdae3
# ╠═de232181-3366-4949-a7a4-6dabbcc10cc2
# ╠═ef1c9edb-f3e8-4972-8d8a-8cefa57dd544
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
