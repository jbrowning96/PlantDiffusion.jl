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

function periodic_bc(u)
    u[0,:] = u[-2,:];
    u[-1,:] = u[1,:];
    u[:,0] = u[:,-2];
    u[:,-1] = u[:,1];
end

function laplacian(u)
    # Second Order Finite Difference 
    laplacian = u[]
    return laplacian;
end

function gray_scott(U, V, )
    return 0
end

# VISUALIZATION

#TODO: Fix all of these functions, to work in Julia. Seems like init works as expected, but the rest don't.