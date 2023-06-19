function init(n)

    u = ones((n+2, n+2));
    v = zeros((n+2, n+2));

    X_Mesh = (0:(n+1))' .* ones(n+2)/11;
    Y_Mesh = (ones(n+2)/(n+1))' .* (0:(n+1));
    
    for i = 1:12, j in 1:12
        if X_Mesh[i,j] > 0.4 && X_Mesh[i,j] < 0.6 && Y_Mesh[i,j] > 0.4 && Y_Mesh[i,j] < 0.6
            u[i,j] = 0.5;
            v[i,j] = 0.25;
        end
    end

    return u, v;
end

function log(u, v)
    #Logging:
    println("Successful Operation")
    println("====================================================================================")
    show(stdout, "text/plain", u)
    print('\n')
    show(stdout, "text/plain", v)
    println()
    println("====================================================================================")
    println()
end

function periodic_bc(u)
    # Periodic Boundary Conditions
    u[1,:] = u[end-1,:]
    u[end,:] = u[2,:]
    u[:,1] = u[:,end-1]
    u[:,end] = u[:,2]
end

function laplacian(u)
    # Second Order Finite Difference 
    laplacian = zeros(10,10);

    for i in 2:11, j in 2:11
        laplacian[i-1,j-1] = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]
    end

    return laplacian;
end

function gray_scott(U, V, Du, Dv, f, k)
    #=
    U - 'Chemical A' array 
    V - 'Chemical B' array
    Du - Diffusion Rate of A
    Dv - Diffusion Rate of B
    f - Feed Rate of A
    k - Kill Rate of B

    The formulas are given as follows:

    ∂u/∂t = Du∇²u - uv² + f(1-u)
    ∂v/∂t = Dv∇²v + uv² - (f+k)v

    where ∇² is the Laplacian operator, numerically approximated using 
    the second finite difference.
    =# 

    u, v = U[2:end-1, 2:end-1], V[2:end-1, 2:end-1];
    print("(u, v): Import of U and V || ")
    log(u, v);

    Lu = laplacian(U);
    Lv = laplacian(V);

    print("(Lu, Lv): Laplacian of U and V || ")
    log(Lu, Lv);

    uv² = u.*v.^2;
    u += Du*Lu - uv² .+ f*(1-u);
    v += Dv*Lv + uv² .- (f+k)*v;

    print("(u, v): After operations are applied to u and v || ")
    log(u, v);
    
    print('\n')
    print("(U, V): After operations are applied to U and V || ")
    log(U, V);

    periodic_bc(U);
    periodic_bc(V);

    print("(U, V): After periodic_bc is applied to U and V || ")
    log(U, V);

    return U, V;
end

# VISUALIZATION

#TODO: Fix all of these functions, to work in Julia. Seems like init works as expected, but the rest don't.