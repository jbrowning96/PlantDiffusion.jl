include("gray_scott.jl")

function main()
    f = 0.0545;
    k = 0.6200;
    Du = .1;
    Dv = .05;
    n = 10;

    U, V = init(n);

    
    print("Initialization Successful || ")
    log(U,V);

    for t in 1:40

        print("TIMESTEP: ", t, " || ")
        log(U,V);

        U, V = gray_scott(U, V, Du, Dv, f, k);
    end

end

main()