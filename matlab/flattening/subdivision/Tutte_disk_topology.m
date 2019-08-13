function u =  Tutte_disk_topology(T,V)
% take T,V with disc topology and flattens them
    % create triangulation object
    tri = triangulation(T,V);
    % get the boundary of the mesh
    b=tri.freeBoundary();
    b=b(:,1);
    % solve Tutte with adjacency matrix
    W = adjacency_matrix(T);
    %create laplacian
    l = linspace(1,length(W),length(W));
    W = -W;
    W = W + sparse(l,l,-sum(W(l,:)));
    %fix the boundary vertices to the unit disk
    W(b,:) =0;
    W(b,b) = 1;
    W = W + sparse(b,b,ones(length(b),1),length(W),length(W));
    boundary_values = points_on_polygon(length(b));
    boundary_targets = zeros(length(W),2);
    boundary_targets(b,:) = boundary_values(:,:);
    %solve the linear system
    u = zeros(length(W),2);
    u(:,1) = W\boundary_targets(:,1);
    u(:,2) = W\boundary_targets(:,2);
end