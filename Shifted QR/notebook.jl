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
 * Objetivos
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

# ╔═╡ 5a04ad92-ed1f-4674-858a-f36de9a335a4
md"
## Introducción
### Objetivos
* Explicar claramente el método de descomposición QR con desplazamiento, haciendo énfasis en los aspectos teóricos y computacionales relevantes,

* implementar el algoritmo de Shifted QR en Julia,

* realizar experimentos numéricos bien diseñados, que permitan ilustrar el comportamiento del método bajo diferentes condiciones (tamaño de matrices, precisión, número de iteraciones, etc.),

* analizar y discutir los resultados, con observaciones fundamentadas tanto en la teoría como en los datos obtenidos.

* resumir los hallazgos más relevantes y, plantear preguntas o ideas para trabajo futuro.

### Motivación
### ?
Aceleración de convergencia
Eficiencia en algoritmos modernos (desplazamientos—explícitos o implícitos—incrementan el aislamiento de subbloques triangulares de la matriz, reduciendo rápidamente la parte activa del problema)
"

# ╔═╡ 06a284b8-ab2d-4b12-9153-5f8e88485e78
md"
## Teoría 

### Método QR básico y limitaciones

#### Esquema del algoritmo

El método QR es una iteración matricial basada en la descomposición QR:

Dado $A_0=A$, se realiza en cada paso:

* Descomposición QR de $A_k$: $A_k=Q_kR_k \text{}$

* Cálculo de $A_{k+1}$: $A_{k+1}=R_kQ_k$

Notemos que cada iteración produce una matriz similar a la anterior:
$A_{k+1}=Q_k^TA_kQ_k$


Por tanto, todas las $A_k$ son similares a la original $A$, y tienen los mismos autovalores. 

Bajo ciertas condiciones (e.g., $A$ simétrica), las $A_k$ convergen a una matriz diagonal cuya diagonal contiene los autovalores de $A$.

#### Convergencia

Este tema es tratado en la sección 7.3.3 del libro.

Sea $A \in \mathbb{C}^{n \times n}$ una matriz diagonalizable con valores propios ordenados por magnitud:

$|\lambda_1| > |\lambda_2| > \cdots > |\lambda_n|.$

Supongamos que $A$ tiene una descomposición de Schur:

$A = Q_0 T_0 Q_0^H,$

donde $Q_0$ es unitario y $T_0$ es triangular superior con los valores propios de $A$ en la diagonal.

El algoritmo QR sin shift genera una secuencia de matrices $\{A^{(k)}\}$ mediante:

$A^{(k)} = Q_k R_k, \quad A^{(k+1)} = R_k Q_k,$

con lo cual:

$A^{(k+1)} = Q_k^{-1} A^{(k)} Q_k.$

**Teorema (convergencia de subespacios):**

Sea $e_1 \in \mathbb{C}^n$ el primer vector canónico. Si $A$ tiene vectores propios 
$\{x_i\}$, con $A x_i = \lambda_i x_i$, entonces:

$\lim_{k \to \infty} A^{(k)} = T = \text{matriz triangular superior similar a } A,$

y además, la subespacio generado por $A^{(k)} e_1$ converge al **autovector dominante** $x_1$, con razón:

$\left| \frac{\lambda_2}{\lambda_1} \right|^k.$

Más específicamente, los elementos subdiagonales de $A^{(k)}$ se atenúan con razón lineal:

$|(A^{(k)})_{i+1, i}| = O\left(\left|\frac{\lambda_{i+1}}{\lambda_i}\right|^k\right).$

Es decir que el algoritmo QR sin shift **converge linealmente**, y su velocidad depende del cociente entre valores propios consecutivos. La convergencia puede ser muy lenta si los valores propios están cercanos en magnitud.

##### Estabilidad del método QR básico

El método QR se basa en transformaciones ortogonales (o unitarias, en el caso complejo), las cuales son numéricamente estables porque **preservan la norma 2** y no amplifican los errores de redondeo. En cada paso, se realiza una transformación de similaridad de la forma:

$A_{k+1} = Q_k^\top A_k Q_k$

Estas transformaciones no degradan la condición del problema ni introducen inestabilidad inherente. Por esta razón, el método QR es considerado numéricamente estable: los autovalores obtenidos son aproximaciones fiables de los autovalores exactos de la matriz original, dentro de los límites del error de redondeo.

---

#### Precisión del método QR básico

La precisión en los autovalores calculados depende principalmente de:

* La separación entre los autovalores (problemas con autovalores cercanos presentan mayor sensibilidad).
* La acumulación de errores en la descomposición QR (influenciada por la técnica empleada: Householder, Givens, MGS, etc.).

Según el análisis de Golub y Van Loan, el método QR aplicado a una matriz simétrica produce autovalores $\hat{\lambda}_i$ tales que:

$|\hat{\lambda}_i - \lambda_i| \approx u \|A\|_2$

donde $u$ es la unidad de redondeo (por ejemplo, $u \approx 10^{-16}$ en doble precisión), y $\|A\|_2$ es la norma espectral de la matriz.
"

# ╔═╡ 891fb287-b56b-46e1-bd7c-93eb7ca4c9ad
md"""
## Método QR con shift
### Motivación

Dado que la velocidad de convergencia del método QR básico es el cociente entre los dos primeros autovalores, este método puede presentar **convergencia lenta** cuando los autovalores de la matriz están cercanos entre sí. Esta lentitud se traduce en una mayor cantidad de iteraciones y, por tanto, un mayor costo computacional.

### Definición

Sea $A \in \mathbb{R}^{n \times n}$ una matriz real. El **método QR con desplazamiento** es una variante del algoritmo QR clásico que acelera la convergencia mediante la incorporación de un escalar $\mu_k \in \mathbb{R}$ en cada iteración.

##### Tipos
Estático vs dinámico

#### Esquema general de la iteración

Dado $A_0 := A$, para cada $k \geq 0$ realizamos:

1. **Aplicar el shift**:

   $A_k - \mu_k I = Q_k R_k$

   donde $Q_k$ es ortogonal y $R_k$ es triangular superior.

2. **Actualizar la matriz**:

   $A_{k+1} = R_k Q_k + \mu_k I$

##### Justificación algebraica del shift estático
Esta operación puede reescribirse como:
$A_{k+1} = Q_k^\top A_k Q_k$,
debido a que:

$Q_k^TA_kQ_k$

$=Q_k^T (Q_k R_k+ \mu I) Q_k$

$=Q_k^T Q_k R_k Q_k + \mu Q_k^T Q_k$

$=R_k Q_k +\mu I$

$= A_k$

Esto implica que $A_{k+1} \sim A_k \sim A$: la matriz resultante es similar a la anterior, y por tanto **preserva los autovalores**. En otras palabras, el método QR con shift genera una sucesión de matrices similares entre sí.

#### Forma de Hessenberg
Las matrices de Hessenberg
...
#### Elección del desplazamiento
#### Variando el desplazamiento
Escogiendo $h_{nn}$ cada vez.

#### Convergencia
La convergencia es lineal, y el cociente espectral afecta el ritmo de convergencia:
$\bigg|\frac{\lambda_{i+1}−u}{\lambda_i−u}\bigg|^k.$ (Analizado en la sección 7.5.2)


Un shift estático no induce deflación rápida, a diferencia del shift dinámico basado en Rayleigh o Wilkinson. 
"""

# ╔═╡ e4c54b68-e57e-4dc4-ba2e-bdc055ec67ba
md"
### Versión cambiando valor
#### Convergencia 
Con mucho gusto. A continuación presento una redacción formal para la subsección **Convergencia del método QR con desplazamiento**, basada en *Matrix Computations* (Golub & Van Loan, 4ta ed., sección 7.5.2), en un estilo acorde al notebook:

---

### 📈 Convergencia del método QR con desplazamiento

El método QR sin desplazamiento converge en general **lentamente**, especialmente cuando los autovalores están próximos entre sí o no bien separados en magnitud. La introducción de un desplazamiento $\mu_k$ en cada iteración no sólo acelera la convergencia, sino que **altera el comportamiento espectral de forma favorable**.

#### 🧠 Intuición espectral

El desplazamiento tiene el efecto de acercar un autovalor particular al origen del espectro de la matriz desplazada $A_k - \mu_k I$. Como el algoritmo QR aplicado a una matriz desplazada se comporta análogamente al **método de la potencia inversa**, este desplazamiento **atrae la convergencia hacia el autovalor más cercano a $\mu_k$**.

Después de reconstruir la matriz con $A_{k+1} = R_k Q_k + \mu_k I$, ese autovalor aparece más claramente en la forma (cuasi)triangular de $A_{k+1}$.

---

### 🧪 Resultados teóricos

#### 🔹 Caso simétrico (autovalores reales y ortogonal diagonalización)

Si $A \in \mathbb{R}^{n \times n}$ es simétrica, entonces el método QR con desplazamiento converge **cuadráticamente** hacia una matriz diagonal:

$$\lim_{k \to \infty} A_k = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$$

y los vectores $Q_0 Q_1 \cdots Q_k$ convergen a una matriz ortogonal cuyos vectores columna son autovectores de $A$.

> **Con desplazamiento**, la velocidad de convergencia se ve aumentada en comparación al caso sin shift. En particular, se muestra (ver GVL, §7.5.2) que:
>
> * Si $A$ es simétrica tridiagonal,
> * y se usa un desplazamiento de Wilkinson (definido más adelante),
>
> entonces los elementos subdiagonales convergen a cero a **una velocidad cuadrática**, i.e., $|a_{n,n-1}^{(k)}| = O(|a_{n,n-1}^{(k-1)}|^2)$.

Esta mejora se debe a que el shift de Wilkinson se elige estratégicamente para aproximar un autovalor dominante en la esquina inferior de la matriz, lo que acelera la separación espectral de ese valor.

---

### ⏱ Comparación entre QR sin shift y con shift

| Característica            | QR sin desplazamiento   | QR con desplazamiento            |
| ------------------------- | ----------------------- | -------------------------------- |
| Velocidad de convergencia | Lineal                  | Cuadrática (en casos favorables) |
| Estructura aprovechada    | Ninguna                 | Tridiagonal (ideal), simetría    |
| Posibilidad de deflación  | Difícil                 | Natural (cuando subdiagonal → 0) |
| Costo por iteración       | $O(n^2)$ con Hessenberg | Igual, pero menos iteraciones    |
| Práctico para uso real    | No                      | Sí                               |

---

### 📌 Comentario adicional: efecto en matrices no simétricas

En matrices generales (no simétricas), el método QR con shift no necesariamente converge a una forma diagonal, sino a la **forma de Schur** (triangular superior). No obstante, el desplazamiento sigue siendo útil: acelera la aparición de valores dominantes en la diagonal, y favorece la **deflación** local.


#### Complejidad




#### 💻 Eficiencia computacional

* Para matrices generales, el costo por iteración es $O(n^3)$.
* Sin embargo, si $A_k$ se mantiene en **forma de Hessenberg** (lo cual es usual en implementaciones prácticas), el costo se reduce a $O(n^2)$ por iteración, gracias a la estructura esparsa.

Este algoritmo constituye la base de la versión moderna del QR para cómputo de autovalores y es utilizado en bibliotecas numéricas como LAPACK.

"

# ╔═╡ 6db839bb-ea7b-486d-9ef0-19d476273b85
md"
# Notas

#### 📌 Observaciones importantes

* Si $\mu_k = 0$ para todo $k$, se recupera el método QR sin desplazamiento.
* A diferencia del QR clásico, el uso del shift permite orientar la iteración hacia una parte específica del espectro.
* Las iteraciones están diseñadas para hacer que ciertos elementos subdiagonales de $A_k$ se reduzcan rápidamente a cero, facilitando la **deflación**.

#### 💡 Justificación algebraica

Sea $A_k = Q_k R_k + \mu_k I$, con

$A_k - \mu_k I = Q_k R_k \quad \Rightarrow \quad A_k = Q_k R_k + \mu_k I$

Entonces:

$A_{k+1} = R_k Q_k + \mu_k I = Q_k^\top A_k Q_k$

Esto implica que cada paso es una transformación ortogonal de similaridad: la matriz $A_k$ es transformada por conjugación ortogonal con $Q_k$. Por lo tanto, todos los $A_k$ son ortogonalmente similares entre sí, y sus autovalores (en exactitud matemática) son idénticos.



#### 🧠 Interpretación espectral

En el contexto de autovalores, restar $\mu_k I$ a $A_k$ equivale a considerar los autovalores $\lambda_i - \mu_k$. El paso QR actúa como una iteración que **reduce la dispersión de los autovalores en torno a cero**, y la restauración con $+\mu_k I$ devuelve la escala original. De este modo, si $\mu_k$ es cercano a un autovalor real de $A_k$, esa componente será realzada en la iteración siguiente.

"

# ╔═╡ 16b57aca-a13d-4774-8f58-17b795e58c01
md"
## Análisis numérico
### Implementación
#### #1
#### #2
### Comparación de tiempo
#### Con matrices normales y de Hessenberg
#### Diferencias de tamaño
### Comparación de estabilidad numérica
"

# ╔═╡ Cell order:
# ╟─c478a7cc-42b0-11f0-1c45-919167ce835a
# ╟─17dbbafe-fe46-4001-a1a0-5636395e89d8
# ╟─5a04ad92-ed1f-4674-858a-f36de9a335a4
# ╠═06a284b8-ab2d-4b12-9153-5f8e88485e78
# ╠═891fb287-b56b-46e1-bd7c-93eb7ca4c9ad
# ╟─e4c54b68-e57e-4dc4-ba2e-bdc055ec67ba
# ╟─6db839bb-ea7b-486d-9ef0-19d476273b85
# ╠═16b57aca-a13d-4774-8f58-17b795e58c01
