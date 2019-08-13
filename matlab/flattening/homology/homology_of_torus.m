function hb = homology_of_torus(T,V)
    % hb is a 2 by 1 cell array with each cell containig a cycle
    hb = cell(2,1);
    %E are the edges of the cut graph = maximal edges such that cutting
    %along them gives a disc
    E = cut_graph(V,T);
    % make an adjacency matrix from E
    Ea = sparse(E(:,1),E(:,2),ones(size(E,1),1),size(V,1),size(V,1));
    Ea = Ea + Ea';
    %find the first cycle on the cut graph
    hb{1} = find_longest_loop_in_graph(Ea);
    c = [hb{1}(1:end-1); hb{1}(2:end)]';
    hb{1}  = hb{1}';
    %cut along the  first cycle
    [G,I] = cut_edges(T,c);
    W = V(I,:);
    %pick a point on the boundary and compute the shortest path to it's
    %"twin"
    t = triangulation(G,W);
    f = freeBoundary(t);
    f= f(:,1);
    A = adjacency_matrix(G);
    %   for every boundary vertex v if there is a loop from  v to it's twin 
    %   in the boundary of cut cylinder, disjoint from the boundary
    %   that is our second loop
    for j  = 1: size(f,1)
        %v is f(j) in the uncut torus
        v= I(f(j));
        %S are the copies of v in W (the torus cut to a cylinder)
        S = find(I==v);
        % if there is a path not intersecting the boundary from f(j) to its
        % twin then we are done. otherwise keep searching.
        A_temp = A;
        f_temp = setdiff(f,S);
        A_temp(f_temp,:) = 0;
        A_temp(:,f_temp) = 0;
        [~,p]= graphshortestpath(A_temp,S(1),S(2));
        if ~isempty(p)
            break
        end
    end
    hb{2} = I(p);

end