function aggregate_predictions(gptoolbox_path, preds_dir, segmentation_path, flat_dir, ploteach)
    % INPUT: PROJ_PATH, SHREC_NUM, MAT_DIR
    
    % add paths
    disp(['pwd: ' pwd])
    addpath(genpath(pwd))
    addpath(genpath('../'))
    addpath(genpath(gptoolbox_path))

    % Parameters
    params = [];
    AGGS_DIR = fullfile(preds_dir, 'aggs');
    mkdir(AGGS_DIR);

    % Iterate over files
    disp([datestr(datetime('now')) ' loading predictions from directory: ', preds_dir]);
    files = rdir(fullfile(preds_dir, '*_pred.mat'));
    fnames = {files.name};
    
    % log arrays
    numClasses=8;
    res = zeros(numel(fnames),1);

    % read segmentation file
    disp([datestr(datetime('now')) ' segmentation file: ', segmentation_path]);
    GTsegmentation = textread(segmentation_path);
   
    % iterate through predictions
    preds_on_mesh = [];
    for ii=1:numel(fnames)
        fname = fnames{ii};
        disp('------------------------------------');
        disp([datestr(datetime('now')) ' lifting fname: ', fname]);

        % get paths
        [~, shortname, ~] = fileparts(fname);
        disp([datestr(datetime('now')) ' file: ', shortname]);
        shortname = shortname(1:end-5);
        flatinfo_path = fullfile(flat_dir, [shortname '.mat']);
        disp([datestr(datetime('now')) ' loading flat_info from: ', flatinfo_path]);
        flat_data = load(flatinfo_path, 'cutMesh');
        V = flat_data.cutMesh.V_orig;
        T = flat_data.cutMesh.T_orig;

        % get prediction on mesh
        [pred_img, pred_on_mesh] = get_prediction_for_img(fname, flat_data.cutMesh, params);
        preds_on_mesh(:, ii) = pred_on_mesh;

        % evaluate prediction vs. ground-truth
        res(ii) = evaluate_mesh(V, T, pred_on_mesh, GTsegmentation);
        disp([datestr(datetime('now')) ' res: ', num2str(res(ii))]);

        % plot results
        if ploteach == true
            figure;
            c = colormap('jet');
            suptitle(strrep(shortname, '_', ' '));

            cc = c(ceil(size(c,1)*double(pred_on_mesh)/numClasses),:);
            subplot(1,2,1); patch('vertices',V,'faces',T,'FaceVertexCData',cc,'FaceColor','flat','EdgeColor','none','FaceAlpha',1);
            title(sprintf('Our Prediction - %.4f', res(ii)), 'FontSize', 30);
            axis equal

            cc = c(ceil(size(c,1)*double(GTsegmentation)/numClasses),:);
            subplot(1,2,2); patch('vertices',V,'faces',T,'FaceVertexCData',cc,'FaceColor','flat','EdgeColor','none','FaceAlpha',1);
            title('Ground Truth', 'FontSize', 30);
            axis equal           
            figpath = fullfile(AGGS_DIR, num2str(ii));
            savefig(figpath);
            disp([datestr(datetime('now')) ' ' 'saved ' figpath]);
        end
    end

    % Agregate predictions on mesh
    pred = mode(preds_on_mesh, 2);     
    res_agg = evaluate_mesh(V, T, pred, GTsegmentation);
    disp([datestr(datetime('now')) ' ' 'aggregated res: ', num2str(res_agg), ' mean: ', num2str(mean(res))]);

    % plot results
    figure;
    c = colormap('jet');
    suptitle(strrep(shortname, '_', ' '));

    cc = c(ceil(size(c,1)*double(pred)/numClasses),:);
    subplot(1,2,1); patch('vertices',V,'faces',T,'FaceVertexCData',cc,'FaceColor','flat','EdgeColor','none','FaceAlpha',1);
    title(sprintf('Our Prediction - %.4f', res_agg), 'FontSize', 30);
    axis equal

    cc = c(ceil(size(c,1)*double(GTsegmentation)/numClasses),:);
    subplot(1,2,2); patch('vertices',V,'faces',T,'FaceVertexCData',cc,'FaceColor','flat','EdgeColor','none','FaceAlpha',1);
    title('Ground Truth', 'FontSize', 30);
    axis equal

    % save res and res_agg
    figpath = fullfile(AGGS_DIR, 'aggregated');
    savefig(figpath);
%     save(figpath, 'res', 'res_agg')
    disp([datestr(datetime('now')) ' ' 'saved ' figpath]); 
end