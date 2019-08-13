function E = cut_graph(V,T)
%%= the output are edges E in the mesh such that (T,V)/E is a disc
    %G is an adjacency matrix of the dual graph with the i'th row of 
    %T being the i'th vertex of the dual graph
    tri  = triangulation(T,V);
    % create the dual graph
    E = tri.edges();
    dual_edges = tri.edgeAttachments(E);
    dual_edges = cell2mat(dual_edges);
    % create the dual graph adjacency matrix
    A = sparse(dual_edges(:,1),dual_edges(:,2),ones(length(dual_edges),1),length(T),length(T));
    A = A+ A';    
    gr = graph(A);    
    %Tree is a matrix of the tree in G
    Tree = gr.minspantree;   
    dual_tree_edges = Tree.Edges;
    %convert the edges of the co-spanningtree to a matrix
    dual_tree_edges = table2array(dual_tree_edges);
    dual_tree_edges = dual_tree_edges(:,[1,2]);
    dual_edges = [ dual_edges ; dual_edges(:,[2,1])];
    % convert the coedges to edges
    [~,indx]=ismember(dual_tree_edges(:,:),dual_edges,'rows');
    indx = mod(indx,length(dual_edges)/2);
    indx(indx==0) = length(dual_edges)/2;
    % TE are the edges in the original matrix corresponding to the edges of
    % the minimal spanning tree of the dual graph
    TE = E(indx,:);
    % the cut graph is the edges of the mesh\TE
    E = setdiff(E,TE,'rows');
end