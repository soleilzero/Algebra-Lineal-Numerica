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

# ╔═╡ 247c0f32-0141-424f-9e8d-6dda19a521db
md"""
#### 🤖 Implementación

Supuestos:

* La matriz $B$ es **bidiagonal superior**: solo tiene elementos en la diagonal $d$ y superdiagonal $f$.
* No se acumulan rotaciones (esto puede añadirse).
* El paso trabaja **in-place** sobre los vectores `d` y `f`.

Notas:

* Esta implementación **modifica `d` y `f` in-place**.
* No acumula las transformaciones $U$ y $V$, pero eso puede añadirse si se desea formar explícitamente las matrices singulares.
* El desplazamiento de Wilkinson garantiza una buena convergencia, especialmente cerca del valor singular más pequeño.

"""

# ╔═╡ 3d76d8e3-fa4a-4e88-81cd-7489cdb146d7
md"##### 👾 Desplazamiento de Wilkinson"

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


# ╔═╡ e9b5e9ee-83ae-41e0-b5cd-d7772d95d27b
validate_shift(3.0, 0.01, 4.0)

# ╔═╡ 906732db-6348-495d-909c-017bbc036659
validate_shift(1.0, 0.5, 1.2)

# ╔═╡ 1740f53a-0e59-4fa7-b63e-e74c0b8bb484
md"##### 👾 Rotaciones de Givens"

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


# ╔═╡ c0d5d51b-09c8-407a-8169-bd8937545954
md"##### 👾 Paso de Golub-Kahan"

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


# ╔═╡ fd6ebedb-0a06-4675-adb2-7076f96fe25b
md"###### 👾 Versión implícita"

# ╔═╡ 9a929fc1-a2b2-474a-9deb-4d8a6bc6d7d1
function golub_kahan_svd_step!(d::Vector{Float64}, f::Vector{Float64})
    n = length(d)
    @assert length(f) == n - 1

    if n < 2
        return d, f
    end

    # --- Paso 1: calcular el shift de Wilkinson ---
    μ = wilkinson_shift(d[end-1], f[end], d[end])

    # --- Paso 2: inicializar el bulge ---
    x = d[1]^2 - μ
    z = d[1] * f[1]

    for k in 1:n-1
        # --- Rotación por la derecha: columnas k, k+1 ---
        c, s, _ = givens_rotation(x, z)

        d_k   = d[k]
        f_k   = f[k]
        d_kp1 = d[k+1]

        # Actualizar d[k] y f[k]
        d[k] =  c * d_k + s * f_k
        f[k] = -s * d_k + c * f_k

        # Crear bulge en d[k+1]
        x = d[k]
        z = d_kp1

        # --- Rotación por la izquierda: filas k, k+1 ---
        c, s, _ = givens_rotation(x, z)

        d[k]     =  c * x + s * z
        d[k+1]   = -s * x + c * z

        # Actualizar f[k] y f[k+1] si no es el último paso
        if k < n - 1
            f_kp1 = f[k+1]
            x = f[k]
            z = f_kp1
            c, s, _ = givens_rotation(x, z)
            f[k]   =  c * x + s * z
            f[k+1] = 0.0  # bulge eliminado
        end
    end

    return d, f
end


# ╔═╡ 9cd5e17f-721a-4b99-bd80-db95a1ef99f0
begin
	golub_kahan_svd_step!([4.0, 3.0, 2.0] , [1.0, 0.5])
	golub_kahan_svd_step!([1.0, 2.0, 3.0, 4.0] , [1.0, 1.0, 0.01])
end

# ╔═╡ 8065ba2c-c0ce-4cc8-87d2-e98fad2868a3
begin
	d = [3.0, 2.0, 1.0]
	f = [0.5, 0.3]
	
	validate_golub_kahan_step(d, f)
	
end

# ╔═╡ 004da5d0-d039-4f0c-890b-16cdf36d21f9
md"###### 👾 Versión explícita"

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


# ╔═╡ 2607c879-d632-4b80-8523-123221a01303
begin
	golub_kahan_svd_step_matrix!(build_bidiagonal([4.0, 3.0, 2.0] , [1.0, 0.5]))
	golub_kahan_svd_step_matrix!(build_bidiagonal([1.0, 2.0, 3.0, 4.0] , [1.0, 1.0, 0.01]))
end

# ╔═╡ 8ec72b94-d090-4c76-bde4-70787a67461a
begin
	B = build_bidiagonal([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 0.01])
	
	validate_golub_kahan_step(golub_kahan_svd_step_matrix!, B)
end

# ╔═╡ 5bd0c109-17c9-4e93-bdfb-e8faf660f708
md"##### 👾 Algoritmo de Golub-Kahan"

# ╔═╡ b2928cde-9e9b-4d28-8a0a-45bae8e8f4ef
"""
Aplica el algoritmo completo de Golub–Kahan para diagonalizar una matriz bidiagonal.

Input:
- d: Vector{Float64} con la diagonal de la matriz bidiagonal B
- f: Vector{Float64} con la superdiagonal de B
- ϵ: Tolerancia (por defecto, 100×eps(Float64))

Output:
- d: Diagonal actualizada (valores singulares)
- f: Superdiagonal (debe terminar cercana a cero)
"""
function golub_kahan_svd!(d::Vector{Float64}, f::Vector{Float64}; ϵ = 100 * eps(Float64))
    n = length(d)
    @assert length(f) == n - 1

    # Ciclo externo: repetir hasta que todos los f[i] sean pequeños
    while true
		display(f)
		display(d)
        # Paso 1: forzar ceros pequeños en f por deflación
        for i in 1:n-1
            tol = ϵ * (abs(d[i]) + abs(d[i+1]))
            if abs(f[i]) ≤ tol
                f[i] = 0.0
            end
        end

        # Paso 2: buscar submatrices no diagonales
        # Buscar el mayor índice q tal que f[q] ≠ 0
        q = n
        while q > 1 && f[q-1] == 0.0
            q -= 1
        end

        # Si ya no queda banda activa, terminar
        if q == 1
            break
        end

        # Buscar el menor p tal que f[p] ≠ 0
        p = q - 1
        while p > 1 && f[p-1] ≠ 0.0
            p -= 1
        end

        # Verificar si hay ceros en la diagonal de B_{p:q}
        zero_on_diag = false
        for i in p:q
            if abs(d[i]) ≤ ϵ * maximum(abs.(d))
                zero_on_diag = true
                break
            end
        end

        if zero_on_diag
            # Si hay ceros en la diagonal, hacer cero la fila correspondiente
            for i in p:q-1
                if abs(d[i]) ≤ ϵ * maximum(abs.(d))
                    f[i] = 0.0
                end
            end
        elseif q - p + 1 >= 2
            # Aplicar un paso QR bidiagonal al bloque activo d[p:q], f[p:q-1]
            dblock = view(d, p:q)
            fblock = view(f, p:q-1)
            golub_kahan_svd_step_matrix!(build_bidiagonal(dblock, fblock))
        end
    end

    return d
end


# ╔═╡ 7c84c4d6-27fc-4c78-9002-eeb6b8780272
#golub_kahan_svd!([4.0, 3.0, 2.0], [1.0, 0.5])
	
# Resultado: `d` contiene los valores singulares aproximados de la matriz bidiagonal

# ╔═╡ 00ca8f64-6937-45ee-8970-d1c2bf49fd59
md"
To Do:
- [ ] Add other implementation (either classical or Jacobi)
- [ ] Have a working Golub

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
# ╟─247c0f32-0141-424f-9e8d-6dda19a521db
# ╟─3d76d8e3-fa4a-4e88-81cd-7489cdb146d7
# ╟─d5e98c9e-92ee-454d-9de8-7c5f38e89c68
# ╟─8ca30862-249a-444d-8a79-702ec8732935
# ╠═e9b5e9ee-83ae-41e0-b5cd-d7772d95d27b
# ╠═906732db-6348-495d-909c-017bbc036659
# ╟─1740f53a-0e59-4fa7-b63e-e74c0b8bb484
# ╟─7902449e-e1ca-4dd7-aef1-96d32e051f40
# ╟─d79631a6-4494-49bd-a9e5-8b100de0272d
# ╟─2edc23a7-b981-4416-8516-54084054bd74
# ╟─c0d5d51b-09c8-407a-8169-bd8937545954
# ╟─c2105a86-9dda-4ccb-a9df-cba3c10b6161
# ╟─a1c28c45-10de-460c-909f-ab724f2afd45
# ╟─fd6ebedb-0a06-4675-adb2-7076f96fe25b
# ╟─9a929fc1-a2b2-474a-9deb-4d8a6bc6d7d1
# ╠═9cd5e17f-721a-4b99-bd80-db95a1ef99f0
# ╠═8065ba2c-c0ce-4cc8-87d2-e98fad2868a3
# ╟─004da5d0-d039-4f0c-890b-16cdf36d21f9
# ╠═adba5d98-0080-4d56-81ca-897d8a97eb39
# ╠═2607c879-d632-4b80-8523-123221a01303
# ╠═8ec72b94-d090-4c76-bde4-70787a67461a
# ╟─5bd0c109-17c9-4e93-bdfb-e8faf660f708
# ╠═b2928cde-9e9b-4d28-8a0a-45bae8e8f4ef
# ╠═7c84c4d6-27fc-4c78-9002-eeb6b8780272
# ╠═00ca8f64-6937-45ee-8970-d1c2bf49fd59
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
