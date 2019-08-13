function [pred_img, pred_on_mesh] = lift_image_AE(cutMesh, params)
    aggtype = getoptions(params, 'aggtype', 'new');
    numClasses=8;

    % get flattening parameters    
    V = cutMesh.V_orig;
    T = cutMesh.T_orig;

    % read prediction
    data = load(fname, 'data');
    pred_img = data.data;    
    % pad pred_img as preparation for lift-sampling
    pred_img = permute(pred_img, [2,3,1]);
    pred_img = [pred_img, pred_img(:, 1, :) ; pred_img(1, :, :), pred_img(1, 1, :)];
    pred_img = permute(pred_img, [3,1,2]);

    if strcmp(aggtype, 'old')
        % get f_on_mesh for each class
        classes_on_flat = zeros(numClasses, size(T, 1));
        for ic=1:numClasses
            class_img = double(pred_img==ic);
            [f_on_mesh, ~] = cutMesh.liftImageFaces(class_img);
            classes_on_flat(ic, :) = f_on_mesh;
        end
        [~, pred] = max(classes_on_flat,[],1);
        pred_on_mesh = pred';

    elseif strcmp(aggtype, 'new')
        preds_on_flat = zeros(size(cutMesh.T, 1), numClasses);
        for i=1:numClasses
            class_IM = squeeze(pred_img(i, :, :));
            [~, IM_on_flat] = cutMesh.liftImageFaces(class_IM);
            preds_on_flat(:, i) = IM_on_flat;
        end        
        T_flat_scales = cutMesh.flattener.dets;
        nTmesh = length(cutMesh.T_orig);
        preds_on_mesh = zeros(nTmesh, numClasses);
        for i=1:nTmesh
            Tmeshi_Ts = cutMesh.uncutTsTocutTs{i};
            Tmeshi_Ts_preds = preds_on_flat(Tmeshi_Ts, :);
            Tmeshi_Ts_scales = T_flat_scales(Tmeshi_Ts);
            preds_on_mesh(i, :) = Tmeshi_Ts_scales' * Tmeshi_Ts_preds;
        end
        [~, pred_on_mesh] = max(preds_on_mesh, [], 2);
        
    else
        error('aggtype %s is not supported', aggtype);
    end
end


