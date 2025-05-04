### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 12c18a31-344e-4514-b28e-59e3bfd0b592
begin
	using PlutoUI
	using HypertextLiteral
	using LinearAlgebra
end

# ╔═╡ 0bfb8466-89bd-410b-9176-54cc257a1c69
md"""
# Tarea 1
"""

# ╔═╡ b6f1a156-268d-11f0-02c5-939326678b23
md""" 
## Algoritmos GS y GSM
Utilizamos la implementación de las notas de clase:
"""

# ╔═╡ 8495c828-e82f-4330-aae7-b1fc51972486
#Gram - Schmidt clásico
function QRCGS(A)
    sizeA=size(A)
    Q = zeros(sizeA) #(m,n)
    R = zeros(sizeA[2],sizeA[2]) #(n,n)
    for i = 1:sizeA[2]
        for j = 1:i-1
            R[j,i] = Q[:,j]'A[:,i]
        end
        p = A[:,i] - Q[:,1:i-1]*R[1:i-1,i]
        R[i,i]=norm(p)
#        if abs(R[i,i])<0.00000000001
#            println("Rii cercano 0")
#        end
        Q[:,i] = p/R[i,i]
    end
    return Q,R
end

# ╔═╡ 54ba1080-8e48-4df9-8e22-be0235a1177a
# Gram - Schmidt modificado
function QRMGS(A)
    sizeA=size(A)
    Q = zeros(sizeA) #(m,n)
    R = zeros(sizeA[2],sizeA[2]) #(n,n)
    for i = 1:sizeA[2]
        R[i,i] = norm(A[:,i])
        Q[:,i] = A[:,i]/R[i,i]
        for j = i + 1: sizeA[2]
            R[i,j] = Q[:,i]'A[:,j]
            A[:,j] = A[:,j] - Q[:,i]R[i,j]
        end
    end
    return Q, R
end

# ╔═╡ f298da4c-c52c-4150-abee-03a55dae42dc
md"""
## Familia de matrices
Generar una familia de matrices aleatorias de tamaño creciente (por ejemplo, n=10,50,100,200, n = 10, 50, 100, 200, n=10,50,100,200) y aplicar ambos algoritmos a cada una. Puede usar también la familia de matrices de Hilbert o del mercado de matrices https://math.nist.gov/MatrixMarket/ 
"""

# ╔═╡ 52189c6b-4273-4223-8107-8b37d9c6418c
md"Generamos la familia de matrices aleatorias n=10,50,100,200"

# ╔═╡ 3c3f0f3e-6107-4445-b2da-713e20b64d92
md"Aplicamos ambos algoritmos a cada una"

# ╔═╡ 0954a5b2-d53e-4eee-b56d-e643208e28f0
begin
	A_10 = rand(10,10)
	A_50 = rand(50,50)
	A_100 = rand(100,100)
	A_200 = rand(200,200)
end

# ╔═╡ 7006cecd-68be-4acf-9bdf-5772abc46a1a
Q1_10, R1_10 = QRCGS(A_10);

# ╔═╡ 5f30fbe8-718e-47ef-93fa-1c7766b5ee2f
Q2_10, R2_10 = QRMGS(A_10); 

# ╔═╡ 3614f55e-062f-4275-96fa-ef004f40ce0c
sizes = [10,50,100,200]

# ╔═╡ 4c4ae68e-c4e0-46e9-8138-2a6b1da4e852
matrices = [rand(Float16,x,x) for x in sizes]

# ╔═╡ 6cc2206f-e495-4193-b130-c69c84929e2c
Q_1, R_1 = QRCGS(matrices[1])

# ╔═╡ 471f522d-8db7-45f4-bf55-cc29b28a3743
results = [QRCGS(x) for x in matrices]

# ╔═╡ d53b9bb5-7bbe-46f8-bdb8-64bbfc3e42b1
display(results[1][1])

# ╔═╡ b7310764-78cf-480a-b79d-65195da553fb
begin
	Q = []; R = [];
	for i in range(1,size(results)[1])
		display(results[i][1])
		
		push!(Q, results[i][1]);
		push!(R, results[i][2]);
	end
end

# ╔═╡ 79664c3b-ef11-4daa-88b3-09482eae61f3
display(Q[1])

# ╔═╡ 9c043bb5-269c-46a6-8600-e0e1cf42d640
display(Q_1)

# ╔═╡ 8ad3dcae-e142-4d0e-a290-f9a1a16eff96
display(R_1)

# ╔═╡ 07981b90-9c99-4bfe-b4c0-2888e4026f52
md" ## Mediciones"

# ╔═╡ c75fa7f2-158e-4ddd-ae29-5b070f595a0f
md"""
Para cada tipo de precisión (Float16, Float32, Float64), medir:
 * **Tiempo de ejecución** de cada algoritmo.
 * **Error de ortogonalidad:** ∥Q^TQ−I∥, \| Q^T Q - I \|,
 * **Residuo de la factorización QR:** ∥A−QR∥ o
"""

# ╔═╡ fcdc43e3-4948-4c6e-97c2-e3de62509dfe
md"Residuo relativo de la factorización y residuo de la ortogonalización"

# ╔═╡ ff0538ab-f3f3-440e-bf79-9a6c533e05c0
begin
	absError = []
	ortError = []
	for i in range(1,size(matrices)[1])
		append!(absError, opnorm(matrices[i]-Q[i]*R[i]))
		append!(ortError, opnorm(Q[i]'Q[i]-UniformScaling(1)))
	end
end

# ╔═╡ 7ac89bce-bfc6-4b6a-8ffb-ed28bd7bcc26
absError

# ╔═╡ 1e89d772-8621-4aef-af60-c0dcda4dbf16
ortError

# ╔═╡ da205278-06c3-4fed-9be6-0c5fee90091a
opnorm(Q1_10'Q1_10-UniformScaling(1))

# ╔═╡ 7a3a41f8-f751-457e-9eac-52d482c47e57
QRCGS(A_10)

# ╔═╡ 3d65fc8d-7d4e-4df3-98fd-79389f9f9098
md"""
## Hi
Para cada tipo de precisión (Float16, Float32, Float64), medir:


 * Tiempo de ejecución de cada algoritmo.

 * Error de ortogonalidad: ∥Q^TQ−I∥, \| Q^T Q - I \|, 

 * Residuo de la factorización QR: ∥A−QR∥ o

"""

# ╔═╡ 2214203b-5e22-464a-8c5f-d6db90448f93
md"""
## Hi
Representar gráficamente los resultados para cada métrica y discutir:


¿Cuál de los dos algoritmos es más estable numéricamente?


¿Cómo afecta la precisión (Float16, Float32, Float64) a cada algoritmo?


¿Cuál es más rápido? ¿A partir de qué tamaño?

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
HypertextLiteral = "~0.9.5"
PlutoUI = "~0.7.62"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "0d161adb749e5cfe6ab3e368a99f79866f3294a3"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

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

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "44f6c1f38f77cafef9450ff93946c53bd9ca16ff"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "d3de2694b52a01ce61a036f18ea9c0f61c4a9230"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.62"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

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

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═0bfb8466-89bd-410b-9176-54cc257a1c69
# ╠═12c18a31-344e-4514-b28e-59e3bfd0b592
# ╠═b6f1a156-268d-11f0-02c5-939326678b23
# ╠═8495c828-e82f-4330-aae7-b1fc51972486
# ╠═54ba1080-8e48-4df9-8e22-be0235a1177a
# ╟─f298da4c-c52c-4150-abee-03a55dae42dc
# ╟─52189c6b-4273-4223-8107-8b37d9c6418c
# ╟─3c3f0f3e-6107-4445-b2da-713e20b64d92
# ╠═0954a5b2-d53e-4eee-b56d-e643208e28f0
# ╠═7006cecd-68be-4acf-9bdf-5772abc46a1a
# ╠═5f30fbe8-718e-47ef-93fa-1c7766b5ee2f
# ╠═3614f55e-062f-4275-96fa-ef004f40ce0c
# ╠═4c4ae68e-c4e0-46e9-8138-2a6b1da4e852
# ╠═6cc2206f-e495-4193-b130-c69c84929e2c
# ╠═471f522d-8db7-45f4-bf55-cc29b28a3743
# ╠═d53b9bb5-7bbe-46f8-bdb8-64bbfc3e42b1
# ╠═b7310764-78cf-480a-b79d-65195da553fb
# ╠═79664c3b-ef11-4daa-88b3-09482eae61f3
# ╠═9c043bb5-269c-46a6-8600-e0e1cf42d640
# ╠═8ad3dcae-e142-4d0e-a290-f9a1a16eff96
# ╠═07981b90-9c99-4bfe-b4c0-2888e4026f52
# ╟─c75fa7f2-158e-4ddd-ae29-5b070f595a0f
# ╠═fcdc43e3-4948-4c6e-97c2-e3de62509dfe
# ╠═ff0538ab-f3f3-440e-bf79-9a6c533e05c0
# ╠═7ac89bce-bfc6-4b6a-8ffb-ed28bd7bcc26
# ╠═1e89d772-8621-4aef-af60-c0dcda4dbf16
# ╠═da205278-06c3-4fed-9be6-0c5fee90091a
# ╠═7a3a41f8-f751-457e-9eac-52d482c47e57
# ╠═3d65fc8d-7d4e-4df3-98fd-79389f9f9098
# ╠═2214203b-5e22-464a-8c5f-d6db90448f93
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
