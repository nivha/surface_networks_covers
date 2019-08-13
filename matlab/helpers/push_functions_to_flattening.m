function [dataFunctions, dataSeg] = push_functions_to_flattening(cutMesh, seg, functions, params)

    params.null = [];
    doplot = getoptions(params,'doplot', 0);
    sz = getoptions(params,'sz', 512);
    method = getoptions(params,'method', 'e_interp');
    numFunctions = size(functions,2);

    if strcmp(method,'e_interp')
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
            [V_merged, T_merged, seg_merged, f, vals] = tile(cutMesh, functions, seg, k);
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

        % flatten segmentation
        segsOnFlatGrid = [];
        for ii=1:max(seg)
            f = zeros(size(V_merged, 1), 1);
            Tseg = T_merged(seg_merged==ii, :); 
            f(Tseg(:)) = 1;
            [out,tn,al2,al3] = mytri2grid(V_merged', T_merged', f, tn, al2, al3);
            segsOnFlatGrid(:,:,ii) = out;
        end
        [~, dataSeg] = max(segsOnFlatGrid,[],3);

    else
        for ii=1:numFunctions
            f=functions(:,ii);
            fOnFlatGrid = captureFunctionOnTriangles(cutMesh, f, struct('isseg',0));
            dataFunctions(:,:,ii) = fOnFlatGrid;
        end
        segsOnFlatGrid =  [];
        for ii=1:max(seg)
           disp([datestr(datetime('now')) ' patching flattened segmentation' num2str(ii)])
           f = double(seg==ii);
           fOnFlatGrid = captureFunctionOnTriangles(cutMesh, f, struct('isseg',1));
           segsOnFlatGrid(:,:,ii) = fOnFlatGrid;
        end
        [~, dataSeg] = max(segsOnFlatGrid,[],3);
    end

    % plot flattened images
    if doplot
        for ii=1:numFunctions
            figure, imagesc(dataFunctions(:,:,ii))
        end
    end


    % if doplot
    % %     for ii=1:size(functions,2)
    % %         figure, imagesc(dataFunctions(:,:,ii))
    % %     end
    % 
    %     % segmentation flat
    %     figure,imagesc(dataSeg)
    %     
    %     % segmentation on mesh
    %     c = colormap('jet');
    %     cc = c(ceil(size(c,1)*double(round(seg))/numel(unique(seg))),:);
    %     figure, patch('vertices',V,'faces',T,'FaceVertexCData',seg,'FaceColor','flat','EdgeColor','none','FaceAlpha',1);
    %     hold on
    %     scatter3(V(cones(1,:),1),V(cones(1,:),2),V(cones(1,:),3),'r','filled')
    % % %     patch('faces',face','vertices',vertex','facecolor','interp','FaceVertexCData',cc,'edgecolor','none','CDataMapping','direct');title('GT')  
    % end

end


function [V_merged, T_merged, seg_merged, f, vals] = tile(cutMesh, functions, seg, k)
    tot_tiles = (k*2+1)^2;

    f = functions(:,1);
    [V_merged, T_merged] = make_tiling(cutMesh, k);
    vals = repmat(cutMesh.inds_plane_to_divided_inds_mesh, tot_tiles, 1);
    seg = seg(cutMesh.dividedTs2Ts);
    seg_merged = repmat(seg, (size(cutMesh.T, 1) / size(seg, 1)) * tot_tiles, 1);

%     xs = V_merged(:, 1);
%     xmin = min(xs(T_merged), [], 2);
%     xmax = max(xs(T_merged), [], 2);
%     
%     ys = V_merged(:, 2);
%     ymin = min(ys(T_merged), [], 2);
%     ymax = max(ys(T_merged), [], 2);
%     
%     ind_xmin = xmin > -0.1;
%     ind_xmax = xmax < 1.1;
%     ind_ymin = ymin > -0.1;
%     ind_ymax = ymax < 1.1;
%     
%     ind = ind_xmin & ind_xmax & ind_ymin & ind_ymax;
%     T_merged = T_merged(ind, :);  
%     seg_merged = seg_merged(ind, :);  

end


