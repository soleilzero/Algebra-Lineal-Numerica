# Proyectos de Álgebra Lineal Numérica
En este repositorio se encuentran las soluciones a los proyectos solicitados para una clase de Álgebra Lineal Numérica de nivel de pregrado.

Puede consultar rápidamente los contenidos y resultados en los reportes en formato PDF, o puede ejecutar los archivos .jl por su cuenta.

Los cuadernos y reportes pueden contener errores de análisis de resultados, pero pueden ser una buena guía para alguien que busque realizar experimentos parecidos. 
Para información más confiable, se recomienda revisar textos académicos como Matrix Computations, Golub & Van Loan. 

## Herramientas
Las herramientas usadas en la creación y ejecución de este repositorio fueron:
* Julia 1.11.1
* Pluto 0.20.3
* ChatGPT (utilizado en la redacción de los reportes)

Sin embargo, también deberían poder ser ejecutados con nuevas versiones. Los reportes en formato PDF y HTML fueron generados automáticamente desde Pluto.

## Ejecución 
Para visualizar y ejecutar los cuadernos .jl en Debian:

1. Asegúrese de tener julia instalado en su computador
2. Ejecute el programa julia
```
$ julia
```
3. Ejecute el programa Pluto dentro de Julia
```julia
> import Pluto; Pluto.run()
```
Esto debería abrir una nueva pestaña Pluto.jl en su navegador.
4. En esta nueva pestaña, ingrese la ubicación del cuaderno en la sección "Open a new notebook"
Aquí ya puedes visualizar cómodamente los contenidos del cuaderno.
5. Si quiere ejecutar el cuaderno, haga click en `Run notebook code`, esto puede demorar más o menos tiempo dependiendo de su contenido.

## Referencias
Además de las mencionadas en cada proyecto, las refencias más importantes fueron:
* Notas de clase,
* [Documentación oficial de Julia](https://julialang.org/),
* Matrix Computations, Golub & Van Loan.
