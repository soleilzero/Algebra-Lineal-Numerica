### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 4fb5a702-eea7-4d84-b5cf-48f5f98c3a0d
begin
	using LinearAlgebra
	using SparseArrays
	using BenchmarkTools
	using Plots
	using PrettyTables
end

# ╔═╡ 4f16bf43-ed9a-4e2b-8e41-600b8a4b0ea4
md"""
# Ortogonalización de Householder con Permutaciones + Comparación con Givens en el caso disperso
"""

# ╔═╡ 0d4e1dfc-3232-11f0-1549-17244c3a3ae6
md"""
## Indicaciones
📢 Tarea: Ortogonalización de Householder con Permutaciones + Comparación con Givens en el caso disperso

### 🎯 Objetivos

*    Implementar el algoritmo de ortogonalización de Householder con permutaciones, capaz de manejar columnas casi linealmente dependientes.
*    Analizar la viabilidad práctica de usar Householder vs. Givens cuando se trabaja con matrices ralas (sparse).
*    Familiarizarse con estructuras y herramientas de Julia para trabajar con matrices dispersas.

### 🛠️ Parte 1: Implementación
Implemente en Julia el algoritmo de ortogonalización de Householder con permutaciones de columnas, es decir, el algoritmo de factorización QR con pivoteo columnar parcial:
A=QRZ.
🔍 Evalúe su algoritmo en presencia de columnas linealmente dependientes o casidependientes.

### 💬 Parte 2: Pregunta de Análisis
Para matrices grandes y dispersas (sparse), compare conceptualmente y computacionalmente los métodos de:

*    Householder
*    Rotaciones de Givens

Responda:

* ¿Cuál de los dos métodos considera más adecuado para mantener la dispersión en una factorización QR de una matriz rala? Justifique su respuesta en términos de la estructura de la matriz, operaciones necesarias, y el patrón de llenado (fill-in).

💡 Puede usar matrices dispersas generadas con sprand() o matrices reales simples de ejemplo, y puede visualizar el llenado con spy() del paquete SparseArrays o Plots.

📦 Sugerencias de herramientas en Julia

    LinearAlgebra
    SparseArrays
    BenchmarkTools
    Plots o Makie para visualización
"""

# ╔═╡ c6cbde37-2796-4867-b5f2-a918672749ad
md""" ## Set up"""

# ╔═╡ dce5fe61-66f5-4e43-a460-4299b8ce21a9
md"""
## Implementación de Householder QR con pivoteo
Implemente en Julia el algoritmo de ortogonalización de Householder con permutaciones de columnas, es decir, el algoritmo de factorización QR con pivoteo columnar parcial:
A=QRZ.
🔍 Evalúe su algoritmo en presencia de columnas linealmente dependientes o casidependientes.

### Código
"""

# ╔═╡ c269a7b8-93fc-4cdb-bfb9-9ab3ed3e1c1c
"""
    house(x) -> v, β

Algoritmo 5.1.1: Dado un vector `x`, devuelve el vector de Householder `v` (con v[1] = 1)
y el escalar `β` tal que `P = I - β * v * v'` es ortogonal y `P * x = ±‖x‖ e₁`.
"""
function house(x)
    m = length(x)
    σ = dot(x[2:end], x[2:end])
    v = [1.0; x[2:end]]

    if σ == 0.0
        if x[1] ≥ 0.0
            β = 0.0
        else
            β = -2.0
        end
    else
        μ = sqrt(x[1]^2 + σ)
        if x[1] ≤ 0
            v[1] = x[1] - μ
        else
            v[1] = -σ / (x[1] + μ)
        end
        β = 2 * v[1]^2 / (σ + v[1]^2)
        v /= v[1]
    end

    return v, β
end


# ╔═╡ e9d5027a-719a-460c-8529-7e2a5fbd73d3
"""
    qr_householder_pivoting(A) -> A_out, piv, r

Algoritmo 5.4.1: QR con pivoteo por columnas. Sobrescribe la matriz `A`:
- Parte superior triangular contiene `R`,
- Debajo de la diagonal: vectores de Householder compactados.
- `piv` codifica la permutación de columnas (Π).
- `r` es el rango estimado.
"""
function qr_householder_with_pivot(A)
    A = copy(A)
    m, n = size(A)
    c = [dot(A[:, j], A[:, j]) for j in 1:n]  # c(j) = ||A[:,j]||²
    piv = collect(1:n)
    r = 0
    τ = maximum(c)

    while τ > 0 && r < n
        r += 1
        # Encontrar el índice k con c(k) = τ, mínimo k
        k = findfirst(c[r:n] .== τ) + r - 1

        # Intercambiar columnas r y k en A, c, y piv
        A[:, [r, k]] = A[:, [k, r]]
        c[[r, k]] = c[[k, r]]
		piv[[r, k]] = piv[[k, r]]


        # Calcular Householder: A[r:end, r]
        v, β = house(A[r:end, r])

        # Aplicar la reflexión: A[r:end, r:n] -= β * v * (v' * A[r:end, r:n])
        A[r:end, r:n] .-= β * v * (v' * A[r:end, r:n])

        # Guardar vector v en A
        A[r+1:end, r] = v[2:end]

        # Actualizar normas c(i)
        for i in r+1:n
            c[i] -= A[r, i]^2
        end

        τ = maximum(c[r+1:end]; init=0.0)
    end

    return A, piv, r
end


# ╔═╡ fdd67542-eb97-4159-ac52-55235541cd89
begin
	B = sprand(100,100, 0.01)
	qr_householder_with_pivot(B)
end

# ╔═╡ 44c341fa-b702-48b7-afe4-5e329b3a8317
function reconstruct_Q(A_fact, r)
    m = size(A_fact, 1)
    Q = Matrix{Float64}(I, m, m)

    for j in r:-1:1
        v = [1.0; A_fact[j+1:end, j]]
        β = 2.0 / dot(v, v)
        Q[j:end, :] .-= β * v * (v' * Q[j:end, :])
    end

    return Q
end

# ╔═╡ 7aef6c03-114b-48b6-9fe7-3f52375b37ab
function qr_householder_with_pivot_results(A)
	A_fact, piv, rank_est = qr_householder_with_pivot(A)
	Q = reconstruct_Q(A_fact, rank_est)
	R = triu(A_fact)
	n = size(A, 2)
	Z = I(n)[:, piv]' 
	return Q, R, Z
end

# ╔═╡ ad100da0-c243-4335-9a59-947989b46da7
function qr_householder_pivoting(A; tol=1e-12)
    m, n = size(A)
    R = copy(A)
    Q = Matrix{Float64}(I, m, m)
    Z = Matrix{Float64}(I, n, n)
    col_norms = vec(norm.(eachcol(R)))

    for k in 1:min(m, n)
        # Elegir la columna con norma más grande
        _, max_col_idx = findmax(col_norms[k:end])
        j = k - 1 + max_col_idx

        # Intercambiar columnas en R y Z
        R[:, [k, j]] = R[:, [j, k]]
        Z[:, [k, j]] = Z[:, [j, k]]
        col_norms[[k, j]] = col_norms[[j, k]]

        # Vector de Householder
        x = R[k:end, k]
        v = copy(x)
        v[1] += sign(x[1]) * norm(x)
        v /= norm(v)

        # Aplicar la reflexión a R
        R[k:end, k:end] .-= 2 * v * (v' * R[k:end, k:end])

        # Aplicar la reflexión a Q
        Q[:, k:end] .-= 2 * (Q[:, k:end] * v) * v'

        # Actualizar normas restantes (opcional: mejora rendimiento)
        for j in k+1:n
            col_norms[j] = norm(R[k+1:end, j])
        end
    end

    return Q, R, Z
end


# ╔═╡ 1f1fa94d-cd90-42bf-8480-88cc87e936ac
md"
### Ejemplo
"


# ╔═╡ 4e3fa428-1811-41d4-8e0c-4e47985536f7
begin
	y1 = randn(5)    # response vector
	X1 = randn(5, 3) # predictor matrix
	display(X1)
end

# ╔═╡ a2067dcf-ce34-4efb-91eb-fdf9e9a259fe
function solve_with_qr(A, b)
    Q, R, Z = qr_householder_with_pivot_results(A)
    Qtb = Q' * b
    x_hat = R \ Qtb     # Solución en coordenadas permutadas
    return Z * x_hat    # Deshacer la permutación de columnas
end

# ╔═╡ 4e529288-d039-49ca-b298-295d60b6e99e
Q1,R1,Z1 = qr_householder_with_pivot_results(X1)

# ╔═╡ c92f4eab-1462-49ff-8ef5-a5a5db663209
reconstructed = Q1 * R1 * Z1

# ╔═╡ 9bb2ebde-f43d-4111-b1f0-706e4be8c50f
norm(reconstructed - X1)

# ╔═╡ 6ffc7d62-400d-441f-9905-c79ebb2eb38d
md"""
### Evaluación del algoritmo en presencia de columnas linealmente dependientes o casidependientes
Evaluemos los cambios en el algoritmo al modificar la casi dependencia de las columnas.

El pivoteo del algoritmo aprovecha la casi dependencia de columnas, al procesar las columnas casi dependientes primero y más rápido. Por lo cual, para evaluar la diferencia entre la presencia de columnas casi dependientes o no, es importante que no sean las primeras columnas las dependientes.

Para la implementación de esto, primero generamos matrices cuya primera y última columnas son dependientes con `gen_dependent_matrix`.
Luego, con la función `mod_dependent_matrix` modificamos el primer valor de la primera columna, tal que las columnas mencionadas sean tan "casi dependientes" como se quiera.
"""

# ╔═╡ 0619059d-68dd-4e83-b725-551b5f8a4d0c
function gen_dependent_matrix(m::Int, n::Int; ratio::Float64 = 2.0)
    @assert n ≥ 2 "La matriz debe tener al menos dos columnas"
    
    # Generar una columna aleatoria
    first_col = randn(m)
    
    # Hacer la segunda columna una combinación lineal de la primera
    last_col = ratio * first_col
    
    # Generar el resto de columnas (aleatorias)
    other_cols = randn(m, n - 2)
    
    # Construir la matriz concatenando las columnas
    A = hcat(first_col, other_cols, last_col)
    return A
end

# ╔═╡ 33904f82-3141-4263-b5d6-1c9f71c91c13
function mod_dependent_matrix(A::Matrix, diff::Float64)
	A[1] += diff
    return A
end

# ╔═╡ 9d8bda6d-69bd-485b-8c21-f012e42884d7
function max_absolute_value_of(r::StepRangeLen)
	return max(
		abs(first(r)),
		abs(last(r))
	)
end	

# ╔═╡ 07d35c59-b314-4034-92ee-89d69fe286f8
md"
#### Evaluación del error del algoritmo
En esta sección analizaremos si el error del algoritmo cambia si en la matriz hay columnas dependientes o casi dependientes.

Para evaluar el error utilizaremos `error_norm_qr_householder_pivoting`, la cual halla la diferencia entre la matriz original y la matriz reconstruida a partir de la descomposición QR generada por `qr_householder_pivoting`.
"

# ╔═╡ 355cab0f-012f-4dff-9cb6-9f7135319487
function error_norm_qr_householder_pivoting(A)
    Q, R, Z = qr_householder_with_pivot_results(A)
    A_reconstructed = Q * R * Z'
	return norm(A - A_reconstructed)
end

# ╔═╡ 9ce7b2f3-3d53-49b6-a815-6491e0c8c2f0
function print_error_norm_qr_householder_pivoting(A)
    println("Norma del error ||A - QRZᵗ|| = ", error_norm_qr_householder_pivoting(A))
end

# ╔═╡ 5ad2df44-15af-44e3-b915-0aff64a54310
md"
#### En presencia de columnas dependientes:
"

# ╔═╡ cbb04c19-ced2-4f83-9559-0a77437c1f70
print_error_norm_qr_householder_pivoting([
	1.0  2.0  3.0;
	4.0  8.0  6.0;
	7.0  14.0  9.0
])

# ╔═╡ 72ecf964-878f-4aa0-9296-944a204e50d2
md"
#### En presencia de columnas casi dependientes:
"

# ╔═╡ a985046b-3b89-4b81-8626-615ff1efb852
print_error_norm_qr_householder_pivoting([
	1.0  2.0  3.0;
	4.0  8.0  6.0;
	7.0  14.0001  9.0
])

# ╔═╡ cb654ecc-8917-4777-9f7e-c051f045d96b
print_error_norm_qr_householder_pivoting([
	1.0  2.0  3.0;
	4.0  8.0  6.0;
	7.0  15.0  9.0
])

# ╔═╡ a52dc169-d858-484a-8ece-b6c03ba7933f
md"El error no es significativamente diferente si las columnas son casi dependientes o no. Podemos hacer un mejor análisis si evaluamos un mayor número de posibilidades."

# ╔═╡ 4f3b57d3-26a7-47d2-a3e9-abb67ec92a7e
md"
#### Variando la casi dependencia de las columnas: 
Evaluemos los cambios en el error al modificar la casi dependencia de las columnas.

Para esto, graficamos el error de la reconstrucción QR con `error_norm_qr_householder_pivoting` para diferentes tres matrices con columnas casi dependientes.
"

# ╔═╡ df51f2eb-b090-4084-9fc5-6454e0f0a5df
begin
	AExample1 = gen_dependent_matrix(4,3)
	AExample2 = gen_dependent_matrix(4,3)
	AExample3 = gen_dependent_matrix(4,3)
	display(AExample1)
end

# ╔═╡ 3086fcf9-a97b-4b80-8dcf-ea069c7adccb
function graph_householder_error_for_almost_dependent_matrices(range)
	title = "Error para diferencias de hasta " * string(last(range))
	plot(
		range, 
		[error_norm_qr_householder_pivoting(mod_dependent_matrix(AExample1,xi)) for xi in range], 
		xlabel="Diferencia `x` entre columnas casi dependientes", 
		ylabel="Error ||A - QRZᵗ||", 
		title=title, 
		legend=true,
		label="AExample1"
	)
	plot!(
		range, 
		[error_norm_qr_householder_pivoting(mod_dependent_matrix(AExample2,xi)) for xi in range],
		label="AExample2"
	)
	plot!(
		range, 
		[error_norm_qr_householder_pivoting(mod_dependent_matrix(AExample3,xi)) for xi in range],
		label="AExample3"
	)
end

# ╔═╡ a0ac0068-0638-4a71-af95-8f78b2c14e32
graph_householder_error_for_almost_dependent_matrices(range(-5, 5, length=500))

# ╔═╡ f9f4bf3f-c73f-4eac-b430-8cad5e072b94
graph_householder_error_for_almost_dependent_matrices(range(-.5, .5, length=500))

# ╔═╡ a8c14878-ee23-45f4-a775-f9dd30545d3f
graph_householder_error_for_almost_dependent_matrices(range(-.001, .001, length=500))

# ╔═╡ 8898e3d0-3bf4-42c7-bb38-cbec5c2a0692
graph_householder_error_for_almost_dependent_matrices(range(-.00001, .00001, length=500))

# ╔═╡ e58f7432-b071-48c2-8683-76d7abbad4f1
md"
#### Resultados
Observamos que:

| Máxima diferencia | Máximo error (aprox) |
|-----|----|
|5.0  |$1.2*10^{-12}$|
|0.5  |$1.2*10^{-12}$|
|$1*10^{-3}$  |$6*10^{-14}$|
|$1*10^{-5}$  |$8*10^{-15}$|

Al comparar las gráficas de los diferentes rangos, podemos ver que el valor máximo disminuye a medida que disminuye el rango.
"

# ╔═╡ dc690a4f-0914-43f6-a6c5-ac3d66f2c32c
md"
### Evaluación del tiempo del algoritmo
En esta sección analizaremos si el tiempo del algoritmo cambia si en la matriz hay columnas dependientes o casi dependientes.
"

# ╔═╡ f0d694a0-12ca-4438-ac7c-6f10f6855b10
function benchmark_householder_pivot_qr_for_almost_dependent(ns, A)
    times = Float64[]

    for n in ns
        # Medición de tiempo
        t  = @belapsed qr_householder_with_pivot($mod_dependent_matrix($A,$n))
        push!(times, t)
    end
	return times
	
end

# ╔═╡ ccf32bed-ed91-4867-82b7-ebde7fe22ccc
md"Dado que las gráficas no tienden a tener cambios bruscos, no necesitamos graficar tantos valores. Por lo cual, vamos a generar las gráficas en un solo paso con ayuda de una función"

# ╔═╡ 1f6fac45-2cd5-4123-bbbc-2a033a09ddd2
function graph_benchmark_householder_pivot_qr_for_almost_dependent_columns(range, size, size_y=size)
	
	times = benchmark_householder_pivot_qr_for_almost_dependent(
		range,
		gen_dependent_matrix(size,size_y)
	)
	plot(
		range, 
		times, 
		xlabel="Diferencia `x` entre columnas casi dependientes", 
		ylabel="Tiempo (s)", 
		title= "Tiempo para diferencias de hasta " * string(max_absolute_value_of(range)), 
		legend=true,
		label="AExample1"
	)
end

# ╔═╡ fe359240-bcc5-4644-b4bc-306e2a70ffca
graph_benchmark_householder_pivot_qr_for_almost_dependent_columns(range(-1, 1, length=50), 50)

# ╔═╡ f81c8526-ee98-48cd-99f0-ffaad54aa289
graph_benchmark_householder_pivot_qr_for_almost_dependent_columns(range(-.1, .1, length=10), 50)

# ╔═╡ 6aabdc0e-c854-48f1-af38-98d2376bfaea
md"""
#### Resultados
"""

# ╔═╡ 735b730a-f3a5-4f60-96df-39388326b05c
md"
## Preguntas de análisis
Para matrices grandes y dispersas (sparse), compare conceptualmente y computacionalmente los métodos de:

*    Householder
*    Rotaciones de Givens

💡 Puede usar matrices dispersas generadas con sprand() o matrices reales simples de ejemplo, y puede visualizar el llenado con spy() del paquete SparseArrays o Plots.

"

# ╔═╡ cf2db2a0-0d20-4e5d-95bf-9ef3287cfc97
md"
### Implementación de Givens QR
Implementamos el algoritmo de descomposición de Givens siguiendo el algoritmo Givens QR del libro Matrix Computations (Golub & Van Loan).
"

# ╔═╡ d4367062-65de-49a7-9bd4-14851c9b9853
function givens(a, b)
    if b == 0
        return (1.0, 0.0)
    else
        r = sqrt(a^2 + b^2)
        return (a / r, b / r)
    end
end

# ╔═╡ a7f97197-6a32-4d06-b35e-25de66c0aab5
function givens_qr(A)
    A = copy(Matrix(A))  # Trabajamos con una copia densa para modificarla en sitio
    m, n = size(A)
    Q = Matrix(1.0I, m, m)  # Acumulador ortogonal

    for j in 1:n
        for i in m:-1:(j + 1)
            a = A[i-1, j]
            b = A[i, j]
            c, s = givens(a, b)

            # Construir Givens implícitamente y aplicar a filas i-1 e i de A (desde columna j en adelante)
            for k in j:n
                temp1 = c * A[i-1, k] + s * A[i, k]
                temp2 = -s * A[i-1, k] + c * A[i, k]
                A[i-1, k] = temp1
                A[i, k]   = temp2
            end

            # Acumular Givens en Q
            for k in 1:m
                temp1 = c * Q[k, i-1] + s * Q[k, i]
                temp2 = -s * Q[k, i-1] + c * Q[k, i]
                Q[k, i-1] = temp1
                Q[k, i]   = temp2
            end
        end
    end

    R = triu(A)
    return Q, R
end

# ╔═╡ 2e5b592c-2e88-4552-9788-a31d7be04ad1
md"
### Comparación conceptual
Recordemos que:

La *transformación de Householder* implica una reflexión ortogonal $H = I - 2 \frac{vv^\top}{v^\top v}$ que anula simultáneamente todos los elementos debajo de una entrada de la columna. 

La *rotación de Givens* implica una rotación ortogonal en el plano $(i, j)$ que anula un único elemento mediante una transformación local.

| **Criterio**                                                    | **Transformaciones de Householder**                                                                                                         | **Rotaciones de Givens**                                                                                           | **Ganador (para matrices dispersas)**                                                      |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| **Estructura de los operadores ortogonales**                    | Se requiere almacenar vectores reflejados ($v$), usualmente densos. Las matrices $H_k$ no son dispersas.                                    | Las rotaciones son representadas por ángulos $(c, s)$; no se almacenan matrices completas.                         | **Givens** (mejor almacenamiento)                                                          |
| **Alcance de cada transformación**                              | Global: afecta toda la submatriz inferior derecha.                                                                                          | Local: modifica únicamente dos filas (i, j) de la matriz.                                                          | **Givens** (mejor preservación local)                                                      |
| **Preservación de la estructura dispersa**                      | Mala: la aplicación de una reflexión global introduce nuevos elementos no nulos (fill-in) en muchas posiciones.                             | Buena: al modificar solo dos filas, el patrón de dispersión se mantiene en gran medida.                            | **Givens** (preserva mejor la dispersión)                                                  |
| **Fill-in (llenado de ceros)**                                  | Alto: cada paso puede transformar columnas y filas escasamente pobladas en densas.                                                          | Bajo: el llenado se restringe al soporte conjunto de las filas involucradas.                                       | **Givens** (mínimo fill-in)                                                                |                                             |
| **Costo computacional por transformación**                      | $\mathcal{O}(mn)$ por reflexión (en el caso general), pues afecta múltiples columnas y filas.                                               | $\mathcal{O}(n)$ por rotación (en promedio), afectando solo dos filas.                                             | **Givens** (mejor costo local)                                                             |
| **Número total de transformaciones**                            | $\min(m, n)$: una reflexión por columna.                                                                                                    | Hasta $\frac{n(n-1)}{2}$ en el peor caso (aunque muchas operaciones pueden omitirse si los elementos ya son cero). | **Empate** (Householder usa menos transformaciones, Givens puede optimizarse en dispersos) |
"

# ╔═╡ 3df82014-b96d-4b01-bd1e-57d76831a118
md"
### Comparación computacional
Vamos a realizar la comparación de ambos algoritmos para matrices densas y dispersas de diferentes tamaños.
#### De tiempo

"

# ╔═╡ e03a43e2-78da-4e19-99f8-aa694bf33675
ns = 10:100:1010  # Tamaños de matrices

# ╔═╡ 37ffd6f2-07f6-411f-b7c1-78dec8c19ff6
function benchmark_qr_methods(ns, density=nothing)
    times_givens = Float64[]
    times_house = Float64[]

    for n in ns
		if density==nothing
	        A = randn(n, n)
		else
			A = sprand(n, n, density)
		end
        # Medición de tiempo
        t_givens = @belapsed givens_qr($A)
        t_house  = @belapsed qr_householder_with_pivot($A)

        push!(times_givens, t_givens)
        push!(times_house, t_house)
    end

    return times_givens, times_house
end


# ╔═╡ 8dfc99fb-ab20-46f0-baef-464111c78ee0
md" ##### En matrices densas"

# ╔═╡ bf43a9d8-62de-47e6-bf60-2111f9a01b03
times_givens_dense, times_house_dense = benchmark_qr_methods(ns)

# ╔═╡ cd9e41cc-eca0-4db3-a675-80a20397149b
begin
	plot(ns, times_givens_dense, label="Givens QR", lw=2, marker=:o)
	plot!(ns, times_house_dense, label="Householder QR with Pivoting", lw=2, marker=:square)
	xlabel!("Matrix Size (n × n)")
	ylabel!("Execution Time (seconds)")
	title!("QR Decomposition Time for dense matrices")
	
end

# ╔═╡ 36b4511c-6531-46e4-bc66-a4cc75fbb87f
md" ##### En matrices dispersas"

# ╔═╡ 41e3ed42-9af4-4e00-b0be-ea4565f9bc67
times_givens_sparse, times_house_sparse = benchmark_qr_methods(ns, .1)

# ╔═╡ 20d9b30e-b85f-4087-a74f-d452a13c325f
begin
	plot(ns, times_givens_sparse, label="Givens QR", lw=2, marker=:o)
	plot!(ns, times_house_sparse, label="Householder QR with Pivoting", lw=2, marker=:square)
	xlabel!("Matrix Size (n × n)")
	ylabel!("Execution Time (seconds)")
	title!("QR Decomposition Time for sparse matrices")
	
end

# ╔═╡ 65e755c0-3910-4427-b9d1-663e4539a893
md" ##### Resultados
En ambos experimento, el algoritmo de Givens QR es más rápido que el de Householder QR con pivoteo.

Sin embargo, para las matrices densas, la diferencia es ligera. Mientras que, para las matrices dispersas, la diferencia es importante.
"

# ╔═╡ cbf02f31-d55e-46f4-85fb-800ec902e54c
md"
## De llenado de ceros
Evaluemos la conservación de la matriz dispersa de ambos métodos.
"

# ╔═╡ 0e88256e-3dd1-46fe-a64c-ba5d2cbe200b
function density(A) 
	num_nonzeros = isa(A, SparseArrays.AbstractSparseMatrix) ? nnz(A) : count(!iszero, A) 
	return num_nonzeros / length(A)
end

# ╔═╡ e743312f-1c13-4e3c-ba1c-e71d0b9109b1
md"
### Ejemplo
"

# ╔═╡ 8001cf5a-123d-411d-bee5-d218745823dc
begin
	A = sprand(150, 150, 0.01)
	A_fact, piv, rank_est = qr_householder_with_pivot(A)
	Q_householder, R_householder, Z_householder = qr_householder_with_pivot_results(A)
	Q_givens, R_givens = givens_qr(A)
end

# ╔═╡ 028e9a85-8a6b-41af-970b-72d337375083
begin
	display(density(A))
	display(density(R_householder))
	display(density(R_givens))
end

# ╔═╡ 2a00132b-4d2d-45e6-b2fb-3ab46f09f41c
spy(A, title="Matriz original")

# ╔═╡ fe1a52e8-c9bb-485e-86eb-fd0728eee79b
spy(R_householder,title="Matriz R con Householder QR con pivoteo")

# ╔═╡ c4e572dd-5873-4cd7-aa4b-80f622c2aeb7
spy(R_givens,title="Matriz R con Givens QR")

# ╔═╡ a317e5d7-e1a0-4701-8392-aac44a303331
md"### Experimento"

# ╔═╡ 8dd73bd7-35a5-4f28-9f60-2b5e254358cc
function density_experiment(n_trials::Int, n::Int, m::Int)
    results = NamedTuple[]

    for i in 1:n_trials
        A = sprand(n, m, 0.01)

        Q_householder, R_householder, Z_householder = qr_householder_with_pivot_results(A)

        Q_givens, R_givens = givens_qr(A)

        push!(results, (
            trial = i,
            density_A = density(A),
            density_R_householder = density(R_householder),
            density_R_givens = density(R_givens),
        ))
    end
	return results
end

# ╔═╡ ecd6d15b-c4fe-4100-aef9-82b7b493f6ec
densities = density_experiment(10,20,30)

# ╔═╡ 3fd4b826-d38f-4339-ac9b-f1d1e905e496
pretty_table(
	densities; 
	header=["Trial", "A", "R_Householder", "R_Givens"]
)

# ╔═╡ ce1ecfa4-7bc7-485a-9da2-de28a22aa04e
md"
### Resultado
Tanto en el ejemplo como en el experimento, podemos ver que el algoritmo de Givens conserva mejor la dispersión de la matriz que el de Householder con pivoteo.
"

# ╔═╡ 9cce2e64-c3ee-461d-98f0-b7a3460046e8
md"
### ¿Cuál de los dos métodos considera más adecuado para mantener la dispersión en una factorización QR de una matriz dispersa? 

Justifique su respuesta en términos de:
* la estructura de la matriz, 
* las operaciones necesarias y 
* el patrón de llenado (fill-in).

Para **matrices dispersas**, las **rotaciones de Givens** son más adecuadas si el objetivo es **preservar la estructura dispersa** y **minimizar el llenado (fill-in)** durante la factorización QR.

**Justificación:**

1. **Estructura local:** Givens modifica únicamente dos filas por vez. Esto evita que un cero en una fila se transforme en un valor no nulo por la propagación de una transformación global, como ocurre en Householder.

2. **Preservación del patrón disperso:** Las transformaciones de Givens pueden ser diseñadas para aprovechar los ceros ya presentes, omitiendo operaciones innecesarias. Esto es especialmente útil en aplicaciones como resolución de sistemas dispersos o álgebra lineal estructurada.

3. **Control de llenado:** El llenado que se introduce por rotaciones de Givens está restringido al soporte conjunto de las filas que se modifican. En contraste, una sola reflexión de Householder puede densificar completamente una matriz dispersa, generando costos de almacenamiento y computación más altos. Esta superioridad fue evaluada empíricamente en este cuaderno. 

4. **Rapidez**

"

# ╔═╡ 8ceb7928-fb68-4ecf-a9ca-4e6fbc4195c6
md"Hay uno que sí es mejor"

# ╔═╡ ea8df5dd-15a8-4acd-bfba-ebf2d5fe2f31
md"""
 ## Reflexión
Durante el desarrollo de esta tarea enfrenté diversas dificultades, principalmente relacionadas con el desconocimiento inicial de los algoritmos de factorización QR, en particular las transformaciones de Householder con pivoteo y las rotaciones de Givens. No obstante, el enfoque analítico requerido por el proyecto me permitió comprender su funcionamiento de forma progresiva, con el objetivo claro de poder compararlos tanto conceptual como computacionalmente.

En relación con la tarea anterior, tuve menos dificultades en el manejo de Julia. Me sentí más cómodo experimentando, y los análisis que realicé estuvieron mejor fundamentados. Utilicé herramientas como @belapsed del paquete BenchmarkTools para medir tiempos de ejecución de forma precisa, y realicé visualizaciones con hasta 500 puntos de prueba para que mis conclusiones se basaran en tendencias observables y no en casos aislados.

## 🤖 Declaración de IA y fuentes externas

Se utilizó inteligencia artificial —en particular **ChatGPT de OpenAI**— como herramienta de apoyo para:

* entender los algoritmos utilizados,
* generar código en Julia conforme a algoritmos académicos,
* organizar comparaciones conceptuales y computacionales,
* corregir problemas de implementación de código en Julia,
* y redactar secciones del informe bajo supervisión y revisión crítica del estudiante.

Los principales *prompts* utilizados fueron:

* "¿Por qué es diferente el resultado de `X \\ y` del que se obtiene con `qr_householder_pivoting()`?"
* "Por favor, guíame en cómo hacer esto: comparar computacionalmente Householder y Givens para matrices dispersas, usando `sprand()` y `spy()`."
* "Por favor, escribe este algoritmo de Givens en Julia: Algorithm 5.2.4 (Givens QR) dado A ∈ ℝ^{m×n}..."
* Cuando ejecuto el código, la función givens_qr() genera el siguiente error: InexactError: Bool(-0.9346593678274439)
* "¿Cómo podría hacer la comparación práctica entre los dos métodos?"
* "Me gustaría graficar los `compare_qr` para matrices de diferente tamaño."
* "Por favor, escribe dos secciones: la contribución del autor y reflexión, y la declaración de IA y fuentes externas."

La IA fue utilizada de manera responsable como herramienta de apoyo técnico y pedagógico.

### 📚 Recursos usados

* **ChatGPT**
* [*Householder Method for QR decomposition* playlist by Adam Sperry](https://www.youtube.com/playlist?list=PLxKgD50sMRvBHxvNPnGQ1kEHlO5y7mSnh)
* [*Computational tools for research in (bio)statistics* by Dr. Hua Zhou](https://hua-zhou.github.io/teaching/biostatm280-2019spring/slides/11-qr/qr.html#Householder-QR-with-column-pivoting)
* [*Householder reflections versus Givens rotations in sparse orthogonal decomposition*](https://www.sciencedirect.com/science/article/pii/002437958790111X)
* *Matrix Computations*, Golub & Van Loan
"""

# ╔═╡ b4bd7c1c-1ca3-4470-867e-e73cbbc128a7
md"
## TO DO
- [ ] time householder
- [ ] spy fill-in
- [ ] Rewrite
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
BenchmarkTools = "~1.3.2"
Plots = "~1.40.12"
PrettyTables = "~2.4.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "62f6c44ef9b32ab1aec122b8a9166d8b50e49e9e"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2ac646d71d0d24b44f3f8c84da8c9f4d70fb67df"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.4+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "e7b7e6f178525d17c720ab9c081e4ef04429f860"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.4"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "0ff136326605f8e06e9bcf085a356ab312eef18a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.13"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "9cb62849057df859575fc1dda1e91b82f8609709"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.13+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "f93655dc73d7a0b4a368e3c0bce296ae035ad76e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.16"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "1d4015b1eb6dc3be7e6c400fbd8042fe825a6bac"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.10"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cd10d2cc78d34c0e2a3a36420ab607b611debfbb"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.7"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a9697f1d06cc3eb3fb3ad49cc67f2cfabaac31ea"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.16+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "44f6c1f38f77cafef9450ff93946c53bd9ca16ff"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "41c9a70abc1ff7296873adc5d768bff33a481652"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.12"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.URIs]]
git-tree-sha1 = "cbbebadbcc76c5ca1cc4b4f3b0614b3e603b5000"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "068dfe202b0a05b8332f1e8e6b4080684b9c7700"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.47+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╠═4f16bf43-ed9a-4e2b-8e41-600b8a4b0ea4
# ╟─0d4e1dfc-3232-11f0-1549-17244c3a3ae6
# ╠═c6cbde37-2796-4867-b5f2-a918672749ad
# ╠═4fb5a702-eea7-4d84-b5cf-48f5f98c3a0d
# ╟─dce5fe61-66f5-4e43-a460-4299b8ce21a9
# ╠═c269a7b8-93fc-4cdb-bfb9-9ab3ed3e1c1c
# ╠═e9d5027a-719a-460c-8529-7e2a5fbd73d3
# ╠═fdd67542-eb97-4159-ac52-55235541cd89
# ╠═44c341fa-b702-48b7-afe4-5e329b3a8317
# ╠═7aef6c03-114b-48b6-9fe7-3f52375b37ab
# ╠═ad100da0-c243-4335-9a59-947989b46da7
# ╠═1f1fa94d-cd90-42bf-8480-88cc87e936ac
# ╠═4e3fa428-1811-41d4-8e0c-4e47985536f7
# ╠═a2067dcf-ce34-4efb-91eb-fdf9e9a259fe
# ╠═4e529288-d039-49ca-b298-295d60b6e99e
# ╠═c92f4eab-1462-49ff-8ef5-a5a5db663209
# ╠═9bb2ebde-f43d-4111-b1f0-706e4be8c50f
# ╠═6ffc7d62-400d-441f-9905-c79ebb2eb38d
# ╠═0619059d-68dd-4e83-b725-551b5f8a4d0c
# ╠═33904f82-3141-4263-b5d6-1c9f71c91c13
# ╠═9d8bda6d-69bd-485b-8c21-f012e42884d7
# ╟─07d35c59-b314-4034-92ee-89d69fe286f8
# ╠═355cab0f-012f-4dff-9cb6-9f7135319487
# ╠═9ce7b2f3-3d53-49b6-a815-6491e0c8c2f0
# ╟─5ad2df44-15af-44e3-b915-0aff64a54310
# ╠═cbb04c19-ced2-4f83-9559-0a77437c1f70
# ╟─72ecf964-878f-4aa0-9296-944a204e50d2
# ╠═a985046b-3b89-4b81-8626-615ff1efb852
# ╠═cb654ecc-8917-4777-9f7e-c051f045d96b
# ╟─a52dc169-d858-484a-8ece-b6c03ba7933f
# ╠═4f3b57d3-26a7-47d2-a3e9-abb67ec92a7e
# ╠═df51f2eb-b090-4084-9fc5-6454e0f0a5df
# ╠═3086fcf9-a97b-4b80-8dcf-ea069c7adccb
# ╠═a0ac0068-0638-4a71-af95-8f78b2c14e32
# ╠═f9f4bf3f-c73f-4eac-b430-8cad5e072b94
# ╠═a8c14878-ee23-45f4-a775-f9dd30545d3f
# ╠═8898e3d0-3bf4-42c7-bb38-cbec5c2a0692
# ╟─e58f7432-b071-48c2-8683-76d7abbad4f1
# ╟─dc690a4f-0914-43f6-a6c5-ac3d66f2c32c
# ╠═f0d694a0-12ca-4438-ac7c-6f10f6855b10
# ╟─ccf32bed-ed91-4867-82b7-ebde7fe22ccc
# ╠═1f6fac45-2cd5-4123-bbbc-2a033a09ddd2
# ╠═fe359240-bcc5-4644-b4bc-306e2a70ffca
# ╠═f81c8526-ee98-48cd-99f0-ffaad54aa289
# ╠═6aabdc0e-c854-48f1-af38-98d2376bfaea
# ╟─735b730a-f3a5-4f60-96df-39388326b05c
# ╟─cf2db2a0-0d20-4e5d-95bf-9ef3287cfc97
# ╠═d4367062-65de-49a7-9bd4-14851c9b9853
# ╠═a7f97197-6a32-4d06-b35e-25de66c0aab5
# ╠═2e5b592c-2e88-4552-9788-a31d7be04ad1
# ╠═3df82014-b96d-4b01-bd1e-57d76831a118
# ╠═e03a43e2-78da-4e19-99f8-aa694bf33675
# ╠═37ffd6f2-07f6-411f-b7c1-78dec8c19ff6
# ╠═8dfc99fb-ab20-46f0-baef-464111c78ee0
# ╠═bf43a9d8-62de-47e6-bf60-2111f9a01b03
# ╠═cd9e41cc-eca0-4db3-a675-80a20397149b
# ╠═36b4511c-6531-46e4-bc66-a4cc75fbb87f
# ╠═41e3ed42-9af4-4e00-b0be-ea4565f9bc67
# ╠═20d9b30e-b85f-4087-a74f-d452a13c325f
# ╠═65e755c0-3910-4427-b9d1-663e4539a893
# ╟─cbf02f31-d55e-46f4-85fb-800ec902e54c
# ╠═0e88256e-3dd1-46fe-a64c-ba5d2cbe200b
# ╟─e743312f-1c13-4e3c-ba1c-e71d0b9109b1
# ╠═8001cf5a-123d-411d-bee5-d218745823dc
# ╠═028e9a85-8a6b-41af-970b-72d337375083
# ╠═2a00132b-4d2d-45e6-b2fb-3ab46f09f41c
# ╠═fe1a52e8-c9bb-485e-86eb-fd0728eee79b
# ╠═c4e572dd-5873-4cd7-aa4b-80f622c2aeb7
# ╠═a317e5d7-e1a0-4701-8392-aac44a303331
# ╠═8dd73bd7-35a5-4f28-9f60-2b5e254358cc
# ╠═ecd6d15b-c4fe-4100-aef9-82b7b493f6ec
# ╠═3fd4b826-d38f-4339-ac9b-f1d1e905e496
# ╠═ce1ecfa4-7bc7-485a-9da2-de28a22aa04e
# ╠═9cce2e64-c3ee-461d-98f0-b7a3460046e8
# ╠═8ceb7928-fb68-4ecf-a9ca-4e6fbc4195c6
# ╠═ea8df5dd-15a8-4acd-bfba-ebf2d5fe2f31
# ╠═b4bd7c1c-1ca3-4470-867e-e73cbbc128a7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
