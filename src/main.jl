include("gray_scott.jl")

function main()

    #Constants:
    f = 0.0545;
    k = 0.6200;
    Du = .1;
    Dv = .05;
    n = 10;
    time_start = 1;
    time_end = 100;
    
    U, V = init(n);

    for time in time_start:time_end
        U, V = gray_scott(U, V, Du, Dv, f, k, n);

        print("TIMESTEP ", time, ": Array Values (U,V) || ")
        log(U,V)
    end

    println("END OF SIMULATION")
end

main()