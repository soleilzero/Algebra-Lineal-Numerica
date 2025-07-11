### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 701241c2-322f-45ca-b17d-437cf32a7ff3
using LinearAlgebra

# ╔═╡ a58e9c22-51a1-11f0-0fc0-914efb421b18
md"""
# Cálculo Numérico de la SVD: Algoritmo de Golub–Kahan

## Índice
1. Objetivos
1. Introducción
1. Fundamentos teóricos
3. Algoritmos
"""

# ╔═╡ 11b2ebb7-626e-49c1-a294-c9b834c73bdb
md"
## Objetivos

Explicar y analizar detalladamente el algoritmo numérico para calcular la descomposición en valores singulares (SVD), con énfasis en el enfoque de Golub–Kahan basado en la reducción bidiagonal y la iteración QR implícita.

* Exponer las conexiones teóricas entre la SVD y los problemas de autovalores simétricos,
* analizar el comportamiento numérico del algoritmo clásico frente a métodos mal condicionados,
* implementar y visualizar los pasos clave del algoritmo de Golub–Kahan,
* discutir la eficiencia computacional y estrategias prácticas.

En otro trabajo se puede aplicar la SVD computada numéricamente a un problema práctico.

## Introducción 

La descomposición en valores singulares (SVD) es una de las herramientas más poderosas y versátiles del álgebra lineal numérica.
No solo existe para cualquier matriz y proporciona información estructural sobre ella —como su rango, condición y modos principales de acción—, sino que también constituye la base de aplicaciones fundamentales en análisis de datos, compresión, métodos de mínimos cuadrados, y problemas inversos.

Desde el punto de vista computacional, calcular la SVD de una matriz general $\in \mathbb{R}^{m \times n}$ de forma precisa y eficiente requiere mucho más que aplicar definiciones algebraicas. De hecho, formar matrices como $A^T A$ puede llevar a errores numéricos significativos. Para abordar este desafío, Golub y Kahan propusieron en 1965 un algoritmo robusto y eficiente basado en dos pasos clave:

1. **Reducción bidiagonal de $A$** mediante transformaciones de Householder.
2. **Diagonalización de la matriz bidiagonal** resultante usando una iteración QR adaptada, que opera de forma implícita sobre $B^T B$, sin construirla explícitamente.

Este notebook está dedicado al estudio detallado de este algoritmo. Nuestro objetivo no es solo entender cómo se implementa, sino por qué funciona, cómo se conecta con los problemas simétricos de autovalores, y qué garantías numéricas ofrece.

A lo largo de este recorrido, desarrollaremos teoría, visualizaremos ejemplos pequeños y construiremos bloques de código que ilustran cada etapa del proceso. Este análisis está inspirado en la sección 8.6 del libro *Matrix Computations*, de Golub y Van Loan, una referencia clásica en el área.

> **Requisitos previos**: se asume familiaridad con transformaciones ortogonales (Householder, Givens), descomposición QR, y problemas de autovalores simétricos.

---
"

# ╔═╡ 6b68d0ac-8118-44aa-940e-5a807217e562
md"
## Fundamentos teóricos de la SVD
Definición y existencia

Propiedades básicas

SVD y matrices simétricas asociadas ?

Matriz aumentada simétrica

Implicaciones algorítmicas

### 🟢 Definición y existencia de la SVD

Sea $A \in \mathbb{R}^{m \times n}$. La **descomposición en valores singulares (SVD)** de $A$ es una factorización de la forma:

$A = U \Sigma V^T,$

donde
- las matrices $U \in \mathbb{R}^{m \times m}$ y $V \in \mathbb{R}^{n \times n}$ son ortogonales,
- la matriz $\Sigma \in \mathbb{R}^{m \times n}$ es diagonal rectangular con entradas no negativas en la diagonal:
  
$\Sigma = \begin{bmatrix}
\sigma_1 & & & \\
& & \ddots & \\
& & & \sigma_r \\
& & & & 0 \\
\end{bmatrix}, 
\quad \text{con } \sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0,$

y $r = \operatorname{rank}(A)$.

Las cantidades $\sigma_i$ se llaman **valores singulares** de $A$, y son únicos. Los vectores columna de $U$ y $V$ se llaman **vectores singulares izquierdos** y **derechos**, respectivamente.

#### ❗ Teorema de existencia

**Toda** matriz real \( A \in \mathbb{R}^{m \times n} \), sea cuadrada, rectangular, singular o no, **admite una descomposición SVD** como la descrita arriba.

Esto contrasta con la factorización espectral, que solo existe para matrices simétricas.

#### 🧠 Intuición geométrica

La SVD describe la acción de la matriz $A$ como una transformación que:

1. **Rota** el espacio fuente (a través de $V^T$),
2. **Escala** en direcciones ortogonales (a través de $\Sigma$),
3. **Rota** el espacio imagen (a través de $U$).

En otras palabras, toda matriz real lineal puede descomponerse como una secuencia de rotaciones/reflexiones y escalamientos. Esta intuición es presentada de manera gráfica por Visual Kernel en el Capítulo 1 Visualize Different Matrices de su colección SEE Matrix.


"

# ╔═╡ 56ef11c7-1b71-45f7-b286-38a932ac3dc4
md"""
### 🟢 Propiedades básicas de la SVD

Dada la descomposición $A = U \Sigma V^T$ de una matriz $A \in \mathbb{R}^{m \times n}$, se cumplen las siguientes propiedades fundamentales:

#### 🔹 Ortogonalidad de los vectores singulares

- Las **columnas de $V$** (vectores singulares derechos) son autovectores de $A^T A$ y forman una base ortonormal de $\mathbb{R}^m$.
- Las **columnas de $U$** (vectores singulares izquierdos) son autovectores de $A A^T$ y forman una base ortonormal de $\mathbb{R}^n$.


#### 🔹 Interpretación de los valores singulares

- Los **autovalores** de $A^T A$ y $A A^T$ son exactamente $\sigma_i^2$.
- Los valores singulares $\sigma_i$ son los **autovalores positivos** de $A^T A$ o $A A^T$:
  $A^T A = V \Sigma^2 V^T, \quad A A^T = U \Sigma^2 U^T$
- Siempre se ordenan de forma no creciente:
  $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$
- Se cumple que $\text{rank}(A) = r = \# \{ \sigma_i > 0 \}$.

#### 🔹 Normas y condición

- Norma 2:
  $\| A \|_2 = \sigma_1$
- Norma de Frobenius:
  $\| A \|_F = \left( \sum_{i=1}^r \sigma_i^2 \right)^{1/2}$
- Número de condición (si $A$ es cuadrada y de rango completo):
  $\kappa_2(A) = \frac{\sigma_1}{\sigma_r}$

#### 🔹 Descomposición como suma de rangos 1

La descomposición SVD permite representar cualquier matriz $A \in \mathbb{R}^{m \times n}$ como una **suma de matrices de rango 1**:

$A = \sum_{i=1}^r \sigma_i u_i v_i^T,$

donde $r = \operatorname{rank}(A)$, $u_i$ y $v_i$ son los vectores singulares izquierdo y derecho correspondientes al valor singular $\sigma_i$, cada término $\sigma_i u_i v_i^T$ es una matriz de **rango 1**.

Esta forma revela que la SVD descompone $A$ como la suma de **modos básicos** ordenados por importancia (es decir, energía o norma).

##### 🔸 Aproximación de rango bajo óptima

Una de las propiedades más poderosas de la SVD es que permite construir la **mejor aproximación de rango $k$** a $A$ en sentido de mínima distancia, tanto en norma 2 como en norma de Frobenius.

Para cualquier $k < r$, definimos:

$A_k = \sum_{i=1}^k \sigma_i u_i v_i^T,$

entonces se cumple $\| A - A_k \|_2 = \sigma_{k+1}$ y $\| A - A_k \|_F^2 = \sum_{i = k+1}^r \sigma_i^2$.

Este resultado se conoce como el **teorema de Eckart–Young**, y afirma que $A_k$ es la mejor aproximación de $A$ por una matriz de rango $k$, en el sentido de ser la más cercana según dichas normas.


##### 📌 Consecuencias prácticas

- Los primeros términos de la SVD capturan la **estructura principal** de $A$.
- Los términos restantes pueden considerarse **ruido o detalles finos**.
- Este principio es la base de muchas técnicas de **compresión**, **reducción de dimensionalidad** y **filtrado**.

Ejemplos típicos incluyen:
- Representar imágenes usando solo los primeros 10 valores singulares.
- Aproximar grandes matrices de datos con bajo rango efectivo.

"""

# ╔═╡ dbb9e53d-bb68-4220-a32d-6c3ce32e274a
md"""
## Algoritmo de cálculo de la SVD

* **Opción 1**: $A^T A$
* * Definición
* * Por qué no se usa
* **Opción 2**: Desde el punto de vista numérico, **¿cómo se calcula esta descomposición sin formar $A^T A$?**
* * Intro
* * Fundamentos teóricos
* * Método de Golub–Kahan

## Comparación / Análisis de los algoritmos
En la sección anterior vimos que la SVD de una matriz $A \in \mathbb{R}^{m \times n}$ (con $m \geq n$) tiene la forma:

$A = U \Sigma V^T,$

donde $U$ y $V$ son ortogonales, y $\Sigma$ es una matriz diagonal rectangular con entradas no negativas.

"""

# ╔═╡ 85c6df45-20ef-4db2-8703-ddf76166832e
md"""
## Algoritmos de cálculo de la SVD

En esta sección abordamos el problema de cómo calcular numéricamente la descomposición en valores singulares de una matriz $A \in \mathbb{R}^{m \times n}$. Aunque su existencia está garantizada teóricamente, el cálculo efectivo de las matrices $U$, $\Sigma$ y $V$ requiere algoritmos cuidadosamente diseñados para asegurar estabilidad y eficiencia, especialmente cuando $A$ está mal condicionada o es de gran tamaño.

Comenzamos con el algoritmo clásico, que se basa en la relación $A^T A = V \Sigma^2 V^T$ y permite obtener los valores y vectores singulares a partir de la descomposición espectral. Este método es útil para fines pedagógicos, pero es numéricamente inadecuado debido a su sensibilidad a errores de redondeo.

Luego presentaremos el método de Golub–Kahan, que constituye la base de las implementaciones modernas de la SVD. Este enfoque evita formar $A^T A$ y combina una reducción bidiagonal mediante transformaciones ortogonales con una versión adaptada del algoritmo QR para matrices simétricas. Analizaremos este método en tres partes: una introducción conceptual, los fundamentos teóricos que lo justifican, y una descripción detallada del algoritmo.

Por último, exploraremos otras variantes relevantes como el método de Jacobi, los algoritmos divide-and-conquer, y algunas estrategias paralelas. Estas alternativas ofrecen ventajas particulares en términos de precisión o rendimiento, y permiten comparar diferentes enfoques bajo distintas condiciones computacionales.


### Algoritmo clásico

#### 🔹 Definición

Dada una matriz $A \in \mathbb{R}^{m \times n}$, el producto $A^T A$ es una matriz simétrica y semidefinida positiva de tamaño $n \times n$. Sus autovalores son no negativos, y sus autovectores son precisamente los vectores singulares derechos de $A$.

Además, se cumple:

$A^T A = V \Sigma^2 V^T,$

lo que sugiere que una forma de obtener los valores singulares sería:

1. Formar $C = A^T A$,
2. Calcular sus autovalores $\lambda_i$,
3. Obtener $\sigma_i = \sqrt{\lambda_i}$,
4. Recuperar $U$ a partir de $AV = U \Sigma$.

#### 🔸 ¿Por qué no se usa este enfoque?

Aunque conceptualmente sencilla, esta estrategia es **numéricamente inestable**:

- **Pérdida de precisión:** el cálculo de $A^T A$ **cuadra la condición** de $A$:
  $\kappa_2(A^T A) = \kappa_2(A)^2,$
  lo que magnifica errores cuando $A$ es mal condicionada.

- **Cancelaciones destructivas:** en aritmética de punto flotante, calcular $A^T A$ puede eliminar información valiosa contenida en $A$.

- **No preserva la estructura original:** operaciones como la bidiagonalización sí preservan mejor las propiedades numéricas de $A$.

Por estas razones, los algoritmos modernos evitan formar explícitamente $A^T A$, y utilizan en cambio métodos más estables, como el de **Golub–Kahan**.


"""

# ╔═╡ 74470dd1-eb43-4039-9d93-6bdf32768466
md"""
### Método de Golub–Kahan

El algoritmo de Golub–Kahan para calcular la SVD de una matriz $A \in \mathbb{R}^{m \times n}$ (con $m \geq n$) consta de dos fases principales:

#### Paso a paso
##### 🔹 Paso 1: Reducción bidiagonal

El primer paso transforma la matriz original $A$ en una matriz **bidiagonal** $B$, es decir, una matriz con ceros fuera de la diagonal y la superdiagonal:

$B = U_1^T A V_1$

Este paso se logra aplicando **transformaciones ortogonales de Householder** por la izquierda y la derecha, que eliminan progresivamente los elementos debajo de la diagonal y fuera de la banda bidiagonal.

La estructura resultante tiene la forma:

$B = \begin{bmatrix}
d_1 & f_1 &        &        & \\
    & d_2 & f_2    &        & \\
    &     & \ddots & \ddots & \\
    &     &        & d_{n-1} & f_{n-1} \\
    &     &        &        & d_n
\end{bmatrix}$

Donde $d_i$ son los elementos diagonales y $f_i$ los elementos de la superdiagonal.

Este paso convierte el problema general en uno estructurado, sin alterar los valores singulares, y además **preserva la estabilidad numérica**.

##### 🔹 Paso 2: Diagonalización bidiagonal (iteración QR simétrica)

Una vez reducida $A$ a bidiagonal, el siguiente objetivo es transformar $B$ en una matriz **diagonal**, es decir, eliminar su superdiagonal sin perturbar la estructura.

Para esto se emplea una versión adaptada del **algoritmo QR simétrico con desplazamiento**. En lugar de formar $B^T B$ (que sería tridiagonal), se trabaja directamente sobre $B$ aplicando secuencias de **rotaciones de Givens** que eliminan progresivamente los elementos $f_i$.

Cada iteración consta de:

1. Cálculo de un **desplazamiento (shift)** aproximado del valor singular, basado en el extremo de la banda bidiagonal.
2. Aplicación de rotaciones alternadas por la derecha e izquierda (tipo QR simétrico implícito) para "empujar" el elemento fuera de banda hacia la esquina inferior.
3. Eliminación (o reducción por deflación) cuando algún $f_i$ es suficientemente pequeño en magnitud.

Este proceso se repite hasta que toda la superdiagonal de $B$ se anula, y la matriz resultante se convierte en la matriz $\Sigma$ de la SVD. Las rotaciones aplicadas se acumulan en matrices ortogonales $U_2$ y $V_2$.


##### 🔹 Paso 3: Composición final

Finalmente, la SVD completa de $A$ se reconstruye a partir de las transformaciones aplicadas:

$A = U \Sigma V^T, \quad \text{donde } U = U_1 U_2,\quad V = V_1 V_2$

En esta descomposición:
- la matriz $\Sigma$ es una matriz diagonal con los valores singulares ordenados,
- las matrices $U$ y $V$ son ortogonales, y
- el proceso es **estable**, ya que todas las operaciones se realizan con transformaciones ortogonales.

"""

# ╔═╡ c0b961d8-3400-4d21-a6ba-3953f48accd8
md"
## 🤖 Implementaciones
"

# ╔═╡ 5584e9c6-ddff-4f96-a32e-fc0a9309617a
md"
### Funciones auxiliares
"

# ╔═╡ 4b7d264c-0790-40a9-bf34-f7d1e31fcc41
"""
Analize si la matriz es ortogonal o no-
"""
function is_orthogonal(Q; atol=1e-12)
    n = size(Q, 2)
    return isapprox(Q' * Q, I(n), atol=atol)
end


# ╔═╡ 6cd89edb-e55f-46bf-b6b1-39b2453cdf1e
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

# ╔═╡ ea4159af-a82d-40b2-ace2-c689aaeecd0d
"""
Valida una descomposición SVD comparando la matriz original A
con su reconstrucción desde la tupla o estructura SVD `F`.

Imprime la norma del error ‖A - UΣVᵀ‖₂.
"""
function validate_svd(A::Matrix{Float64}, F)
    A_hat = reconstruct_from_svd(F)
    error = norm(A - A_hat)
    println("Error de reconstrucción ‖A - UΣVᵀ‖₂ = ", error)
    return error
end


# ╔═╡ 8b67358e-800f-4bf9-8869-90b29612e51e
struct SVDReconstruction
    U::Matrix{Float64}
    S::Vector{Float64}
    V::Matrix{Float64}
end


# ╔═╡ beac433a-c1f4-4191-899f-6eb37d929889
example = randn(5, 3)

# ╔═╡ aab73b0c-4d6e-467e-aaf8-93d8456508bc
md"
### SVD nativo de Julia
Utilizamos la función `LinearAlgebra.svd`
"

# ╔═╡ bb27217f-4d64-4e47-946d-65a4590226ea
svd

# ╔═╡ e2d4f475-165a-47dc-a4ab-1e9b1f97be3f
md"#### Ejemplo"

# ╔═╡ f2d58441-03af-4efb-9ad5-f3628511d7fa
F1 = svd(example, full=true)

# ╔═╡ c7b2ba31-3e8a-4c66-8583-d7817bf53e1b
reconstruct_from_svd(F1)

# ╔═╡ d804ab81-03bb-4770-a508-d58dc204de2b
validate_svd(example, F1)

# ╔═╡ b6b0df75-1ffa-44bb-9b74-518bb918e8ab
md"### Método ingenuo"

# ╔═╡ 5655e762-fc65-48da-a4d8-9bbf0cf5e3fc
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

# ╔═╡ 59f9f7e8-f355-40a3-94e0-5cb3c6dec15d
md"#### Ejemplo "

# ╔═╡ 40a0849e-6ad5-4b70-b708-5cb1d92ed578
F2 = svd_via_ata(example)

# ╔═╡ 6363a2cf-a7bf-475b-8f9c-0de2c5c66697
reconstruct_from_svd(F2)

# ╔═╡ 08313203-f0ae-4603-be49-a85da9ffe178
validate_svd(example,F2)

# ╔═╡ 247c0f32-0141-424f-9e8d-6dda19a521db
md"""
### Método de Golub–Kahan
#### Funciones auxiliares
"""

# ╔═╡ 311a1fdd-e0cc-413a-81c5-262e29a1132b
md"
#### Paso de Golub-Kahan"

# ╔═╡ d5e98c9e-92ee-454d-9de8-7c5f38e89c68
"""
Calcula el shift de Wilkinson (mu) a partir de los tres elementos finales de B bidiagonal.

Inputs:
- dm: elemento d_{n-1} (penúltimo de la diagonal)
- fm: elemento f_{n-1} (último de la superdiagonal)
- dn: elemento d_n (último de la diagonal)

Output:
- mu: shift de Wilkinson
"""
function wilkinson_shift(dm::Float64, fm::Float64, dn::Float64)
    tmm = dm^2 + fm^2
    tmn = fm * dn
    tnn = dn^2

    δ = (tmm - tnn) / 2
    denom = abs(δ) + sqrt(δ^2 + tmn^2)

    if denom == 0
        return tnn
    else
        return tnn - sign(δ) * tmn^2 / denom
    end
end


# ╔═╡ 7902449e-e1ca-4dd7-aef1-96d32e051f40
"""
Devuelve (c, s, r) tal que [c s; -s c] * [y; z] = [r; 0]
"""
function givens_rotation(y::Float64, z::Float64)
    if z == 0
        return (1.0, 0.0, y)
    elseif y == 0
        return (0.0, 1.0, z)
    else
        r = hypot(y, z)
        return (y / r, z / r, r)
    end
end


# ╔═╡ adba5d98-0080-4d56-81ca-897d8a97eb39
"""
Aplica un paso QR bidiagonal de Golub–Kahan a una matriz bidiagonal superior B ∈ ℝ^{n×n}.

- B debe tener solo elementos en la diagonal y superdiagonal.
- La estructura bidiagonal se preserva.
- La transformación es ortogonal y mantiene aproximadamente los autovalores de BᵗB.

Modifica B in-place.
"""
function golub_kahan_svd_step_matrix!(B::Matrix{Float64})
    n = size(B, 1)
    @assert size(B, 2) == n

    # Calcular el shift de Wilkinson
    d = diag(B)
    f = [B[i, i+1] for i in 1:n-1]
    μ = wilkinson_shift(d[end-1], f[end], d[end])

    # Inicializar el primer vector para la rotación por la derecha
    y = d[1]^2 - μ
    z = d[1] * B[1, 2]

    for k in 1:n-1
        # -------- Rotación por la derecha (columnas k, k+1) --------
        c, s, _ = givens_rotation(y, z)

        # Solo columnas k y k+1 de filas k y siguientes (preservar estructura)
        for i in k:n
            t1 = B[i, k]
            t2 = B[i, k+1]
            B[i, k]   =  c * t1 + s * t2
            B[i, k+1] = -s * t1 + c * t2
        end

        # -------- Rotación por la izquierda (filas k, k+1) --------
        y = B[k, k]
        z = B[k+1, k]
        c, s, _ = givens_rotation(y, z)

        # Solo filas k y k+1 de columnas k y siguientes
        for j in k:n
            t1 = B[k, j]
            t2 = B[k+1, j]
            B[k, j]   =  c * t1 + s * t2
            B[k+1, j] = -s * t1 + c * t2
        end

        # Preparar vector (y, z) para la próxima rotación por la derecha
        if k < n - 1
            y = B[k, k+1]
            z = B[k, k+2]
        end
    end

    # Limpiar numéricamente la matriz fuera de la banda bidiagonal
    for i in 1:n
        for j in 1:n
            if j ≠ i && j ≠ i+1
                B[i, j] = 0.0
            end
        end
    end

    return B
end


# ╔═╡ 7f39475d-a3c9-4c5c-8a55-aa2ba34d6289
md"##### Validación"


# ╔═╡ c2105a86-9dda-4ccb-a9df-cba3c10b6161
"""
Construye una matriz bidiagonal superior B ∈ ℝ^{n×n} a partir de:

- d: diagonal de B, longitud n
- f: superdiagonal de B, longitud n - 1

Retorna: B :: Matrix{Float64}, una matriz n×n con estructura bidiagonal superior.
"""
function build_bidiagonal(d, f)
    n = length(d)
    @assert length(f) == n - 1 "La superdiagonal f debe tener longitud n - 1"

    B = zeros(Float64, n, n)

    for i in 1:n
        B[i, i] = d[i]
        if i < n
            B[i, i+1] = f[i]
        end
    end

    return B
end


# ╔═╡ 8ca30862-249a-444d-8a79-702ec8732935
"""
Compara `wilkinson_shift` con el verdadero autovalor de una matriz 2x2 para validar.
"""
function validate_shift(dm, fm, dn)
    T = [
        dm^2 + fm^2    fm * dn
        fm * dn        dn^2
    ]
    λ = eigen(T).values
    μ_ref = λ[argmin(abs.(λ .- dn^2))]  # autovalor más cercano a t_nn
    μ_custom = wilkinson_shift(dm, fm, dn)
    
    println("μ (eigen 2×2)     = $μ_ref")
    println("μ (Wilkinson calc)= $μ_custom")
    println("Error             = $(abs(μ_custom - μ_ref))")
end


# ╔═╡ a1c28c45-10de-460c-909f-ab724f2afd45
"""
Valida un paso de Golub–Kahan aplicado a una matriz bidiagonal B.

- step_func: función que modifica B in-place (como golub_kahan_svd_step_matrix!)
- B: matriz bidiagonal cuadrada (con solo diagonal y superdiagonal)

Opcional:
- verbose = true: imprime detalles
- atol: tolerancia para comparar autovalores

Retorna: true si todo pasa, false si falla alguna prueba.
"""
function validate_golub_kahan_step(step_func::Function, B::Matrix{Float64};
                                   atol=1e-10, verbose=true)
    n = size(B, 1)
    @assert size(B, 2) == n "B debe ser cuadrada"

    # --- Copiar datos originales ---
    B_before = copy(B)
    T_before = B_before' * B_before
    λ_before = sort(eigvals(Symmetric(T_before)))

    # --- Aplicar paso QR bidiagonal ---
    step_func(B)

    # --- Verificar estructura bidiagonal ---
    is_bidiagonal = all(i == j || j == i+1 || B[i, j] == 0.0 for i in 1:n, j in 1:n)

    # --- Comparar autovalores ---
    T_after = B' * B
    λ_after = sort(eigvals(Symmetric(T_after)))
    λ_diff = norm(λ_before - λ_after, Inf)

    # --- Imprimir si se desea ---
    if verbose
        println("✔ Estructura bidiagonal: ", is_bidiagonal)
        println("✔ Autovalores antes:  ", round.(λ_before, digits=8))
        println("✔ Autovalores después:", round.(λ_after, digits=8))
        println("Δλ ∞-norm: ", λ_diff)
    end

    return is_bidiagonal && λ_diff ≤ atol
end


# ╔═╡ 8ec72b94-d090-4c76-bde4-70787a67461a
validate_golub_kahan_step(
	golub_kahan_svd_step_matrix!, 
	build_bidiagonal(
		[1.0, 2.0, 3.0, 4.0], 
		[1.0, 1.0, 0.01]
	)
)

# ╔═╡ e344fe0e-b939-4163-84ac-01c402895e38
md"#### Diagonalización"

# ╔═╡ b2928cde-9e9b-4d28-8a0a-45bae8e8f4ef
"""
Applies the full Golub–Kahan SVD iteration (Algorithm 8.6.2) on a real upper bidiagonal matrix B.

This function repeatedly applies the Golub–Kahan SVD step with Wilkinson shift to deflate
the superdiagonal of B until it becomes numerically zero (i.e., diagonalizes B).

The bidiagonal structure is preserved, and the function modifies B in-place.

Inputs:
- B::Matrix{Float64}: square upper bidiagonal matrix (with only diagonal and superdiagonal)
- ϵ: relative tolerance for deflation (default: 100 × eps(Float64))

Output:
- B is modified in-place to become diagonal (approximate singular values on the diagonal)
"""
function golub_kahan_svd_matrix!(B::Matrix{Float64}; ϵ = 100 * eps(Float64))
    n = size(B, 1)
    @assert size(B, 2) == n "B must be square"
    
    while true
        # Step 1: deflation — set small superdiagonal entries to zero
        for i in 1:n-1
            d1 = abs(B[i, i])
            d2 = abs(B[i+1, i+1])
            tol = ϵ * (d1 + d2)
            if abs(B[i, i+1]) ≤ tol
                B[i, i+1] = 0.0
            end
        end

        # Step 2: find the largest q such that B[q-1, q] ≠ 0
        q = n
        while q > 1 && B[q-1, q] == 0.0
            q -= 1
        end

        if q == 1
            break  # Fully diagonalized
        end

        # Step 3: find the smallest p such that B[p-1, p] == 0
        p = q - 1
        while p > 1 && B[p-1, p] ≠ 0.0
            p -= 1
        end

        # Step 4: handle zeros on the diagonal
        zero_on_diag = false
        maxd = maximum(abs.(diag(B)))
        for i in p:q
            if abs(B[i, i]) ≤ ϵ * maxd
                zero_on_diag = true
                break
            end
        end

        if zero_on_diag
            for i in p:q-1
                if abs(B[i, i]) ≤ ϵ * maxd
                    B[i, i+1] = 0.0
                end
            end
        elseif q - p + 1 ≥ 2
            # Step 5: apply one Golub–Kahan step to the active block
            Bblock = B[p:q, p:q]
            golub_kahan_svd_step_matrix!(Bblock)
            B[p:q, p:q] .= Bblock
        end
    end

    return B
end


# ╔═╡ 502c0650-1925-4174-8c8e-dd963728c7b2
md"##### Validación"

# ╔═╡ b4276865-6a07-484a-9290-7d557af8ac88
"""
Verifica la corrección de golub_kahan_svd_matrix! sobre una matriz bidiagonal B.

- B: matriz bidiagonal cuadrada (modificada in-place)
- atol: tolerancia absoluta sobre errores espectrales y fuera de la diagonal
- verbose: si true, imprime diferencias y estructura

Retorna: true si todo está correcto, false si falla alguna validación.
"""
function validate_golub_kahan_svd_matrix(B::Matrix{Float64}; atol=1e-10, verbose=true)
    n = size(B, 1)
    @assert size(B, 2) == n "B debe ser cuadrada"

    # Copia para comparar
    B0 = copy(B)
    λ0 = sort(eigvals(Symmetric(B0' * B0)))

    # Ejecutar algoritmo
    golub_kahan_svd_matrix!(B)

    # Verificación 1: la matriz resultante debe ser diagonal (bidiagonal con superdiagonal ≈ 0)
    is_diagonal = all(abs(B[i, j]) ≤ atol for i in 1:n, j in 1:n if i ≠ j)

    # Verificación 2: los autovalores de BᵗB deben conservarse
    λf = sort(eigvals(Symmetric(B' * B)))
    λ_diff = norm(λ0 - λf, Inf)

    # Verificación 3: valores singulares ordenados (opcional)
    σ = sort(abs.(diag(B)), rev=true)

    if verbose
        println("✔ Diagonal final: ", is_diagonal)
        println("✔ Autovalores iniciales: ", round.(λ0, digits=8))
        println("✔ Autovalores finales:   ", round.(λf, digits=8))
        println("Δλ ∞-norm: ", λ_diff)
        println("✔ Valores singulares:    ", round.(σ, digits=8))
    end

    return is_diagonal && λ_diff ≤ atol
end


# ╔═╡ c756f929-8da9-41f6-aa58-6247d2a57acb
validate_golub_kahan_svd_matrix(
	build_bidiagonal(
		[1.0, 2.0, 3.0, 4.0],
		[1.0, 1.0, 0.01]
	)
)

# ╔═╡ 3069c786-0823-42a0-92c0-4df61174e5d1
md"
Podemos ver que los autovalores parecen conservarse en las validaciones de `golub_kahan_svd_step_matrix` y `golub_kahan_svd_matrix`. Sin embargo la norma del error es demasiado grande. Esto nos dice que no es un problema de la lógica del algoritmo, sino de su implementación.
Por esto, creemos otra implementación que utilice la matriz bidiagonal de manera implícita.
"

# ╔═╡ 04eb9c27-8a4f-462e-930a-4f643d424769
md"#### Algoritmo de Golub Kahan"

# ╔═╡ 4e69684e-b6bb-439a-a8c7-3a65ebb36f25
"""
Reduce la matriz A a forma bidiagonal utilizando reflexiones de Householder.
Devuelve la matriz bidiagonal B, y las matrices ortogonales U y V tal que A ≈ U * B * Vᵀ.
"""
function bidiagonalize_householder(A::Matrix{Float64})
    m, n = size(A)
    B = copy(A)
    U = Matrix{Float64}(I, m, m)
    V = Matrix{Float64}(I, n, n)

    for i in 1:min(m, n)
        # Reflexión de Householder desde la izquierda (columnas)
        x = B[i:end, i]
        v = copy(x)
        v[1] += sign(x[1]) * norm(x)
        v = v / norm(v)
        B[i:end, i:end] -= 2 * v * (v' * B[i:end, i:end])
        U[:, i:end] -= 2 * (U[:, i:end] * v) * v'

        if i < n
            # Reflexión de Householder desde la derecha (filas)
            x = B[i, i+1:end]'
            v = copy(x)
            v[1] += sign(x[1]) * norm(x)
            v = v / norm(v)
            B[i:end, i+1:end] -= 2 * (B[i:end, i+1:end] * v') * v
            V[:, i+1:end] -= 2 * (V[:, i+1:end] * v') * v
        end
    end

    return B, U, V
end


# ╔═╡ cfa62a92-828e-4510-81ae-985e84e2250e
"""
Fuerza a cero los elementos de la matriz B que están fuera de la banda bidiagonal,
si su magnitud es menor que un umbral dado (por defecto 1e-14).
Esto es útil para limpiar errores numéricos tras la bidiagonalización.

Modifica B in-place.
"""
function force_bidiagonal!(B::Matrix{Float64}; atol=1e-14)
    m, n = size(B)
    for i in 1:m
        for j in 1:n
            if j != i && j != i+1 && abs(B[i,j]) < atol
                B[i,j] = 0.0
            end
        end
    end
end


# ╔═╡ c1a1f0f7-39a9-4597-9c1e-3358b0ee232d
begin
	Y = randn(5, 3)
	B, U, V = bidiagonalize_householder(Y)
	A_hat = U * B * V'
	println("Error de reconstrucción: ", norm(Y - A_hat))
	
end

# ╔═╡ d8888165-0100-4c40-8bd1-0182f91e3565
begin
	
	function is_bidiagonal(B; atol=1e-12)
	    m, n = size(B)
	    for i in 1:m
	        for j in 1:n
	            if (j != i && j != i+1) && abs(B[i,j]) > atol
	                return false
	            end
	        end
	    end
	    return true
	end
	
	@show is_bidiagonal(B)
end

# ╔═╡ 65aa59c8-586f-4850-875c-7a19ed968882
"""
Calcula la SVD de una matriz A ∈ ℝ^{m×n} utilizando el algoritmo de Golub-Kahan.
Paso 1: Bidiagonalización de A usando reflexiones de Householder.
Paso 2: Diagonalización iterativa de la matriz bidiagonal mediante rotaciones de Givens.
Devuelve U, Σ, Vᵀ
"""
function svd_golub_kahan(A::Matrix{Float64}; tol=1e-12, maxiter=1000)
    m, n = size(A)
    B, U, V = bidiagonalize_householder(A)
	force_bidiagonal!(B)
	B = B[1:n, 1:n]
    golub_kahan_svd_matrix!(B)
	display(B)
    Σ = zeros(m, n)
    for i in 1:min(m,n)
        Σ[i,i] = abs(B[i,i])
    end

    return SVDReconstruction(U, diag(Σ), V')
end


# ╔═╡ 6c502074-cfa6-4d7d-8754-b4114743aaeb
md"#### Ejemplo y validación"

# ╔═╡ 537968ea-5d8c-4adc-9f36-d229e21d4085
begin
	F3 = svd_golub_kahan(example)
	validate_svd(example,F3)
end

# ╔═╡ 4f81f155-f5c7-42f2-8e10-af8ae8ad1dae
A_reconstructed = reconstruct_from_svd(F3)

# ╔═╡ ab104798-39bf-44cb-ad07-9d5592524730
md" ### Funciones auxiliares extra"

# ╔═╡ d79631a6-4494-49bd-a9e5-8b100de0272d
"""
Aplica rotación de Givens por la derecha sobre d[k] y f[k]
"""
function apply_right_rotation!(d, f, k::Int, c::Float64, s::Float64)
    d_k  = d[k]
    f_k  = f[k]
    d[k] =  c * d_k + s * f_k
    f[k] = -s * d_k + c * f_k
end


# ╔═╡ 2edc23a7-b981-4416-8516-54084054bd74
"""
Aplica rotación de Givens por la izquierda sobre d[k] y d[k+1]
"""
function apply_left_rotation!(d, k::Int, c::Float64, s::Float64)
    d_k  = d[k]
    d_k1 = d[k+1]
    d[k]   =  c * d_k + s * d_k1
    d[k+1] = -s * d_k + c * d_k1
end


# ╔═╡ a48dea5f-3e17-45d2-9a07-7eb0228ca516
"""
Diagonaliza la matriz bidiagonal B utilizando rotaciones de Givens
y acumula las transformaciones en U y V.
Modifica B, U y V in-place.
"""
function diagonalize_bidiagonal!(B::Matrix{Float64}, U::Matrix{Float64}, V::Matrix{Float64};
                                  tol=1e-12, maxiter=1000)

    m, n = size(B)
    for iter = 1:maxiter
        converged = true

        for i in 1:n-1
            # Verifica si el elemento fuera de la diagonal es significativo
            if abs(B[i, i+1]) > tol * (abs(B[i,i]) + abs(B[i+1,i+1]))
                converged = false

                # Paso 1: rotación a la derecha (columna i e i+1 de B y V)
                x = B[i,i]^2 - B[i+1,i+1]^2
                y = B[i,i] * B[i,i+1]
                c, s = givens_rotation(x, y)

                apply_right_rotation!(B, i, i+1, c, s)
                apply_right_rotation!(V, i, i+1, c, s)

                # Paso 2: rotación a la izquierda (fila i e i+1 de B y U)
                c, s = givens_rotation(B[i,i], B[i+1,i])
                apply_left_rotation!(B, i, i+1, c, s)
                apply_right_rotation!(U, i, i+1, c, s)
            end
        end

        if converged
            break
        end
    end
end


# ╔═╡ a0b20c8f-64dd-4966-87c3-ba4156ebdbf2
md"##### Ejemplo"

# ╔═╡ 78ad67b1-5ead-4d3e-b5bb-062907f524f5
md"---"

# ╔═╡ 277c10ee-bd03-4efe-b54b-0d835a8781a0
md"### Old: 👾 Método implícito"

# ╔═╡ 9a929fc1-a2b2-474a-9deb-4d8a6bc6d7d1
"""
Computa los valores singulares de una matriz A ∈ ℝ^{m×n} (con m ≥ n)
usando la descomposición bidiagonal seguida de la iteración QR implícita
(Golub–Kahan, Algoritmo 8.6.2 de Golub y Van Loan).

No acumula U ni V explícitamente (solo valores singulares).
"""
function svd_golub_kahan_values(A::Matrix{Float64}; ϵ = 100 * eps(Float64))
    m, n = size(A)
    @assert m ≥ n "Se requiere m ≥ n"

    # Paso 1: Reducción bidiagonal A = U₁ * B * V₁ᵗ
    B_full = copy(A)
    U1 = Matrix{Float64}(I, m, m)
    V1 = Matrix{Float64}(I, n, n)
    LinearAlgebra.bidiagonalize!(B_full, U1, V1)

    # Extraer la bidiagonal B
    d = diag(B_full[1:n, 1:n])           # diagonal
    f = [B_full[i, i+1] for i in 1:n-1]  # superdiagonal

    # Paso 2: Iteración QR implícita sobre (d, f)
    function is_converged(f, d)
        all(abs(f[i]) ≤ ϵ * (abs(d[i]) + abs(d[i+1])) for i in 1:length(f))
    end

    while !is_converged(f, d)
        golub_kahan_svd_step!(d, f)
    end

    # Retornar valores singulares ordenados
    return sort(abs.(d); rev=true)
end


# ╔═╡ cb27deb4-cefa-4d34-b29f-be8fd219035d
md"""begin
	# Matriz de prueba
	A = randn(6, 4)
	
	# SVD por Golub–Kahan implícito (solo valores singulares)
	σ = svd_golub_kahan_values(A)
	
	# SVD de referencia usando la función estándar de Julia
	σ_ref = svd(A).S
	
	# Mostrar resultados
	println("σ (GK implícito) = ", round.(σ, digits=6))
	println("σ (referencia)   = ", round.(σ_ref, digits=6))
	println("Error absoluto    = ", round.(maximum(abs.(σ .- σ_ref)), digits=6))
	
end"""

# ╔═╡ 00ca8f64-6937-45ee-8970-d1c2bf49fd59
md"
## To Do:
- [X] Have a working Golub
* - [ ] Full algorithm (From initial Householder)
* - [x] Make a verification function
* - [ ] How to get the SVD from the result?
- [ ] Add or transform into a `Bidiagonal` version (does it count as implicit?)
- [X] Add other implementation (either classical or Jacobi)
     -> Comparar con versión INGENUA, no clásica.
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
# ╟─a58e9c22-51a1-11f0-0fc0-914efb421b18
# ╠═701241c2-322f-45ca-b17d-437cf32a7ff3
# ╟─11b2ebb7-626e-49c1-a294-c9b834c73bdb
# ╟─6b68d0ac-8118-44aa-940e-5a807217e562
# ╟─56ef11c7-1b71-45f7-b286-38a932ac3dc4
# ╟─dbb9e53d-bb68-4220-a32d-6c3ce32e274a
# ╟─85c6df45-20ef-4db2-8703-ddf76166832e
# ╟─74470dd1-eb43-4039-9d93-6bdf32768466
# ╟─c0b961d8-3400-4d21-a6ba-3953f48accd8
# ╟─5584e9c6-ddff-4f96-a32e-fc0a9309617a
# ╟─4b7d264c-0790-40a9-bf34-f7d1e31fcc41
# ╟─6cd89edb-e55f-46bf-b6b1-39b2453cdf1e
# ╟─ea4159af-a82d-40b2-ace2-c689aaeecd0d
# ╠═8b67358e-800f-4bf9-8869-90b29612e51e
# ╠═beac433a-c1f4-4191-899f-6eb37d929889
# ╟─aab73b0c-4d6e-467e-aaf8-93d8456508bc
# ╟─bb27217f-4d64-4e47-946d-65a4590226ea
# ╟─e2d4f475-165a-47dc-a4ab-1e9b1f97be3f
# ╟─f2d58441-03af-4efb-9ad5-f3628511d7fa
# ╟─c7b2ba31-3e8a-4c66-8583-d7817bf53e1b
# ╟─d804ab81-03bb-4770-a508-d58dc204de2b
# ╟─b6b0df75-1ffa-44bb-9b74-518bb918e8ab
# ╟─5655e762-fc65-48da-a4d8-9bbf0cf5e3fc
# ╟─59f9f7e8-f355-40a3-94e0-5cb3c6dec15d
# ╟─40a0849e-6ad5-4b70-b708-5cb1d92ed578
# ╟─6363a2cf-a7bf-475b-8f9c-0de2c5c66697
# ╟─08313203-f0ae-4603-be49-a85da9ffe178
# ╟─247c0f32-0141-424f-9e8d-6dda19a521db
# ╟─d5e98c9e-92ee-454d-9de8-7c5f38e89c68
# ╟─7902449e-e1ca-4dd7-aef1-96d32e051f40
# ╟─c2105a86-9dda-4ccb-a9df-cba3c10b6161
# ╟─8ca30862-249a-444d-8a79-702ec8732935
# ╟─311a1fdd-e0cc-413a-81c5-262e29a1132b
# ╠═adba5d98-0080-4d56-81ca-897d8a97eb39
# ╟─7f39475d-a3c9-4c5c-8a55-aa2ba34d6289
# ╟─a1c28c45-10de-460c-909f-ab724f2afd45
# ╠═8ec72b94-d090-4c76-bde4-70787a67461a
# ╟─e344fe0e-b939-4163-84ac-01c402895e38
# ╟─b2928cde-9e9b-4d28-8a0a-45bae8e8f4ef
# ╟─502c0650-1925-4174-8c8e-dd963728c7b2
# ╟─b4276865-6a07-484a-9290-7d557af8ac88
# ╠═c756f929-8da9-41f6-aa58-6247d2a57acb
# ╟─3069c786-0823-42a0-92c0-4df61174e5d1
# ╟─04eb9c27-8a4f-462e-930a-4f643d424769
# ╟─4e69684e-b6bb-439a-a8c7-3a65ebb36f25
# ╟─cfa62a92-828e-4510-81ae-985e84e2250e
# ╟─c1a1f0f7-39a9-4597-9c1e-3358b0ee232d
# ╟─d8888165-0100-4c40-8bd1-0182f91e3565
# ╟─a48dea5f-3e17-45d2-9a07-7eb0228ca516
# ╟─65aa59c8-586f-4850-875c-7a19ed968882
# ╟─6c502074-cfa6-4d7d-8754-b4114743aaeb
# ╠═4f81f155-f5c7-42f2-8e10-af8ae8ad1dae
# ╠═537968ea-5d8c-4adc-9f36-d229e21d4085
# ╟─ab104798-39bf-44cb-ad07-9d5592524730
# ╟─d79631a6-4494-49bd-a9e5-8b100de0272d
# ╟─2edc23a7-b981-4416-8516-54084054bd74
# ╟─a0b20c8f-64dd-4966-87c3-ba4156ebdbf2
# ╟─78ad67b1-5ead-4d3e-b5bb-062907f524f5
# ╟─277c10ee-bd03-4efe-b54b-0d835a8781a0
# ╟─9a929fc1-a2b2-474a-9deb-4d8a6bc6d7d1
# ╟─cb27deb4-cefa-4d34-b29f-be8fd219035d
# ╟─00ca8f64-6937-45ee-8970-d1c2bf49fd59
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
