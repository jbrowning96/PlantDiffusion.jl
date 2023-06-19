function init(n)

    u = ones((n+2, n+2));
    v = zeros((n+2, n+2));

    X_Mesh = (0:(n+1))' .* ones(n+2)/(n+1);
    Y_Mesh = (ones(n+2)/(n+1))' .* (0:(n+1));
    
    for i ∈ 1:n+2, j ∈ 1:n+2
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

function laplacian(u,n)

    # Second Order Finite Difference
    laplacian = zeros(n,n);

    for i ∈ 2:n+1, j ∈ 2:n+1
        laplacian[i-1,j-1] = u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - (4 * u[i,j])
    end

    return laplacian;
end

function gray_scott(U, V, Du, Dv, f, k, n)

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

    ∇²U = laplacian(U, n);
    ∇²V = laplacian(V, n);

    uv² = u .* v.^2;
    u += Du*∇²U .- uv² .+ f*(1 .- u);
    v += Dv*∇²V .+ uv² .- (f+k)*v;

    U[2:end-1,2:end-1] = u;
    V[2:end-1,2:end-1] = v;

    periodic_bc(U);
    periodic_bc(V);

    return U, V;
end

# VISUALIZATION

#TODO: Fix all of these functions, to work in Julia. Seems like init works as expected, but the rest don't.