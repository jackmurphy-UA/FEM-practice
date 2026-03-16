
    % 2D FEM solver for heat transfer on rectangular domains
    % Using Q4 elements with uniform mesh
    % To do: compute analytical solutions and plot L2 error (as a function of element size or number of elements)

function fem2d_heat_transfer()
        
    close all

    % Problem parameters
    Lx = 10;         % Length in x-direction (m)
    Ly = 10;         % Length in y-direction (m)
    nx = 40;        % Number of elements in x-direction
    ny = 40;        % Number of elements in y-direction
    k  = 1;        % Thermal conductivity (W/mK)
    
    % Boundary conditions (1=Dirichlet, 2=Neumann) (try varying B.C.s or
    % Robin)
    % Order: left, right, bottom, top
    BC_types = [1, 1, 1, 1]; 
    BC_values = [0, 0, 0, 0];
    
    % Heat source function 
    % Q = @(x,y) 0;  % No heat source for simplicity
    
    % Or try something with fluctuations, etc. 
    Q = @(x,y) 5*sin(1/5 * x*y);

    % Q = @(x,y) exp(-1 / ((x-5) * (y-5)));

    % Generate mesh
    [nodes, elements] = generate_mesh(Lx, Ly, nx, ny);
    
    % Initialize global system
    num_nodes = size(nodes, 1);
    K = sparse(num_nodes, num_nodes);
    F = zeros(num_nodes, 1);
    
    % Assembly
    for e = 1:size(elements, 1)
        elem_nodes = elements(e, :);
        x = nodes(elem_nodes, 1);
        y = nodes(elem_nodes, 2);
        
        [Ke, Fe] = element_heat(x, y, k, Q);
        
        K(elem_nodes, elem_nodes) = K(elem_nodes, elem_nodes) + Ke;
        F(elem_nodes) = F(elem_nodes) + Fe;
    end
    
    % Apply boundary conditions
    [K, F] = apply_bc(K, F, nodes, BC_types, BC_values);
    
    % Solve the system

    T = K\F;  % if increasing system size, use iterative method instead
    
    % Plot results
    plot_results(nodes, elements, T, nx, ny);
end

function [nodes, elements] = generate_mesh(Lx, Ly, nx, ny)
    % Generate uniform mesh on rectangular domain [0,Lx] x [0,Ly]
    
    % Node coordinates
    x = linspace(0, Lx, nx+1);
    y = linspace(0, Ly, ny+1);
    [X, Y] = meshgrid(x, y);
    nodes = [X(:), Y(:)];
    
    % Element connectivity (Q4 elements)
    elements = zeros(nx*ny, 4);
    for j = 1:ny
        for i = 1:nx
            elem_id = (j-1)*nx + i;
            n1 = (j-1)*(nx+1) + i;
            n2 = n1 + 1;
            n4 = j*(nx+1) + i;
            n3 = n4 + 1;
            elements(elem_id, :) = [n1, n2, n3, n4];
        end
    end
end

function [Ke, Fe] = element_heat(x, y, k, Q)
    % Compute element stiffness matrix and force vector for Q4 element
    
    % Gauss points (2x2 integration)
    gp = [-1/sqrt(3), -1/sqrt(3);
            1/sqrt(3), -1/sqrt(3);
            1/sqrt(3),  1/sqrt(3);
           -1/sqrt(3),  1/sqrt(3)];
    w = [1, 1, 1, 1]; % weights
    
    Ke = zeros(4,4);
    Fe = zeros(4,1);
    
    for i = 1:4
        xi = gp(i,1);
        eta = gp(i,2);
        
        % Shape functions and derivatives
        N = 0.25 * [(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)];
        dNxi = 0.25 * [-(1-eta), (1-eta), (1+eta), -(1+eta)];
        dNeta = 0.25 * [-(1-xi), -(1+xi), (1+xi), (1-xi)];
        
        % Jacobian matrix
        J = [dNxi*x, dNxi*y; dNeta*x, dNeta*y];
        detJ = det(J);
        % invJ = inv(J);
        
        % Derivatives of shape functions w.r.t. x and y
        dN = [dNxi; dNeta];
        dNxy = J \ dN;
        dNx = dNxy(1,:);
        dNy = dNxy(2,:);
        
        % B matrix
        B = [dNx; dNy];
        
        % Element stiffness matrix contribution
        Ke = Ke + w(i) * (B' * B) * k * detJ;
        
        % Element force vector contribution
        x_phys = N * x;
        y_phys = N * y;
        Fe = Fe + w(i) * N' * Q(x_phys, y_phys) * detJ;
    end
end

function [K, F] = apply_bc(K, F, nodes, BC_types, BC_values)
    % Apply boundary conditions (Dirichlet and/or Neumann)
    
    tol = 1e-6;
    x = nodes(:,1);
    y = nodes(:,2);
    Lx = max(x);
    Ly = max(y);
    
    % Left boundary (x=0)
    left_nodes = find(abs(x) < tol);
    if BC_types(1) == 1 % Dirichlet
        K(left_nodes, :) = 0;
        K(left_nodes, left_nodes) = speye(length(left_nodes));
        F(left_nodes) = BC_values(1);
    end
    
    % Right boundary (x=Lx)
    right_nodes = find(abs(x - Lx) < tol);
    if BC_types(2) == 2 % Neumann
        q = BC_values(2);
        for i = 1:length(right_nodes)-1
            n1 = right_nodes(i);
            n2 = right_nodes(i+1);
            length_edge = norm(nodes(n2,:) - nodes(n1,:));
            F(n1) = F(n1) + q * length_edge/2;
            F(n2) = F(n2) + q * length_edge/2;
        end
    end
    
    % Bottom boundary (y=0)
    bottom_nodes = find(abs(y) < tol);
    if BC_types(3) == 1 % Dirichlet
        K(bottom_nodes, :) = 0;
        K(bottom_nodes, bottom_nodes) = speye(length(bottom_nodes));
        F(bottom_nodes) = BC_values(3);
    end
    
    % Top boundary (y=Ly)
    top_nodes = find(abs(y - Ly) < tol);
    if BC_types(4) == 2 % Neumann
        q = BC_values(4);
        for i = 1:length(top_nodes)-1
            n1 = top_nodes(i);
            n2 = top_nodes(i+1);
            length_edge = norm(nodes(n2,:) - nodes(n1,:));
            F(n1) = F(n1) + q * length_edge/2;
            F(n2) = F(n2) + q * length_edge/2;
        end
    end
end

function plot_results(nodes, ~ , T, nx, ny)
    % Plotting of results
    
    figure;
    
    % Contour plot
    X = reshape(nodes(:,1), ny+1, nx+1);
    Y = reshape(nodes(:,2), ny+1, nx+1);
    T_grid = reshape(T, ny+1, nx+1);
    
    subplot(1,2,1);
    contourf(X, Y, T_grid, 20);
    colorbar;
    title('Temperature Distribution (°C)');
    xlabel('x (m)');
    ylabel('y (m)');
    axis equal;
    
    % Surface plot
    subplot(1,2,2);
    surf(X, Y, T_grid);
     view(65,32);          % A nice view of the 3d plot for this problem
    title('Temperature Surface');
    xlabel('x (m)');
    ylabel('y (m)');
    zlabel('Temperature (°C)');
    
    % Print min and max temperature
    fprintf('Minimum temperature: %.2f°C\n', min(T));
    fprintf('Maximum temperature: %.2f°C\n', max(T));
end
