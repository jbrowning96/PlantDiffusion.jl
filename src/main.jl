include("gray_scott.jl")

function main()
    u, v = init(10);
    show(stdout, "text/plain", u)
    print('\n')
    show(stdout, "text/plain", v)

end

main()