function [dataFunctions] = push_functions_to_flattening_AE(cutMesh, functions, params)

    params.null = [];
    sz = getoptions(params,'sz', 512);
    numFunctions = size(functions,2);

    for ii = 1:length(cutMesh.divided_edges)
        functions = [functions ; ...
            (functions(cutMesh.divided_edges{ii}(:,1),:) + functions(cutMesh.divided_edges{ii}(:,2),:)) / 2];
    end

    % process first function and set good k (how many tiles to put on each side)
    has_nan = 1;
    k = 1;
    while any(has_nan(:))
        k = k + 1;
        disp([datestr(datetime('now')) ' tiling with k=', num2str(k)]);
        [V_merged, T_merged, f, vals] = tile(cutMesh, functions, k);
        X = linspace(0, 1-1/sz, sz);
        Y = linspace(0, 1-1/sz, sz);    
        [out,tn,al2,al3] = mytri2grid(V_merged', T_merged', f(vals), X, Y);
        dataFunctions(:,:,1) = out;

        has_nan = isnan(dataFunctions);
    end

    for ii=2:numFunctions
        f=functions(:,ii);
        [out,tn,al2,al3] = mytri2grid(V_merged', T_merged', f(vals), tn, al2, al3);
        dataFunctions(:,:,ii) = out;
    end
end


function [V_merged, T_merged, f, vals] = tile(cutMesh, functions, k)
    tot_tiles = (k*2+1)^2;

    f = functions(:,1);
    [V_merged, T_merged] = make_tiling(cutMesh, k);
    vals = repmat(cutMesh.inds_plane_to_divided_inds_mesh, tot_tiles, 1);
%     seg = seg(cutMesh.dividedTs2Ts);
%     seg_merged = repmat(seg, (size(cutMesh.T, 1) / size(seg, 1)) * tot_tiles, 1);
end


