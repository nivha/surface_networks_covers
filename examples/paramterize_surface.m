function pushed_function =  paramterize_surface(V,T,f)
%%input: V =spherical mesh vertices
%T = triangle matrix of mesh
%f an |T|x|d| array of d features on the faces of the mesh
addpath(genpath('..\gptoolbox-master'))
addpath(genpath('..\matlab'))

tuple1 = {{[6, 7, 8, 9, 10],[1],[2],[3],[4],[5]},{[1, 7, 3, 4, 9],[2],[5],[6],[8],[10]...
    },{[1, 8, 4, 3, 7],[2],[5],[6],[9],[10]},{[2, 5, 4, 7, 6],[1],[3],[8],[9],[10]},{[2, 10, 9, 4, 5],...
    [1],[3],[6],[7],[8]}};



k = length(tuple1);

params.n = 5;
params.nFarthest = k;
params.doplot = 0;
[cones, AGD] = getPointTripletsByFarthestPointSampling(V, T, params);
[~, min_agd_point] = min(AGD);

[cutMesh, ~, ~] = flatten_sphere(V ,T, cones, min_agd_point, tuple1);

pushed_function = push_functions_to_flattening_AE(cutMesh, f);

