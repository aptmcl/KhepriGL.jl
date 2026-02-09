using KhepriGL
using Documenter

DocMeta.setdocmeta!(KhepriGL, :DocTestSetup, :(using KhepriGL); recursive=true)

makedocs(;
    modules=[KhepriGL],
    authors="António Menezes Leitão <antonio.menezes.leitao@gmail.com>",
    sitename="KhepriGL.jl",
    format=Documenter.HTML(;
        canonical="https://aptmcl.github.io/KhepriGL.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/aptmcl/KhepriGL.jl",
    devbranch="master",
)
