function  E = find_edges_intersecting_with_line(T,V,start_point,end_point)
    tri = triangulation(T,V);
    e = tri.edges;
    XY1 = [start_point end_point];
    XY2 = [V(e(:,1),1) , V(e(:,1),2), V(e(:,2),1) , V(e(:,2),2)];
    var  = lineSegmentIntersect(XY1,XY2);
    adj_matrix = var.intAdjacencyMatrix;
    E_ind = (adj_matrix == 1);
    E  = e(E_ind,:); 
end