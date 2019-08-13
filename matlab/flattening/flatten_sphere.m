function [cutMesh, gluer, flattener] = flatten_sphere(V ,T, cones,min_agd_point, tuple)
   
    % the gluing object. Glues spheres to torus
    disp([datestr(datetime('now')),' Gluing to torus']);
    gluer = Gluer(V,T,cones,tuple,min_agd_point);
     % the flattening object. Flattens torus to the plain
    disp([datestr(datetime('now')),' Flattening torus to the plane']);
    flattener = Torus_Flattener(gluer.V_torus,gluer.T_torus);
    disp([datestr(datetime('now')),' Creating CutMesh object']);
    cutMesh = CutMesh(gluer, flattener);
end

