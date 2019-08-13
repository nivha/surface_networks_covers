function shape_ret_flatten_test(gptoolbox_path, batches_dir, dst_dir, n_augment, n_features)
    % add paths
    disp(['pwd: ' pwd])
    addpath(genpath(pwd))
    addpath(genpath('../'))
    addpath(genpath(gptoolbox_path))

    % general parameters
    params.sz = 128;

    % load cutMesh object
    CUTMESH_PATH = '../spherical/cutMeshs/sphere_cutMesh_12_6';
    data = load(CUTMESH_PATH, 'cutMesh');
    cutMesh = data.cutMesh;

    % create destination images folder if needed
    OUT_IMG_DIR = fullfile(dst_dir);
    if ~exist(OUT_IMG_DIR, 'dir')
       mkdir(OUT_IMG_DIR);
    end

    % process all batch files in folder
    files = rdir(fullfile(batches_dir, '*.mat'));
    fnames = {files.name};
    for batch_i=1:numel(fnames)

        % retreive batch data 
        batch_fname = fnames{batch_i};    
        data = load(batch_fname, 'data', 'file_names');
        batch_data = data.data;
        batch_file_names = data.file_names;

        % flatten batch
        sample_augs = zeros(n_augment, params.sz, params.sz, n_features);
        batch_size = size(batch_data, 1) / n_augment;
        for sample_i=1:batch_size
            for jj=1:n_augment
               ii = n_augment * (sample_i-1) + jj;
               sample = squeeze(batch_data(ii, :, :))';
               sample_img = push_functions_to_flattening_AE(cutMesh, sample, params);
               sample_augs(jj, :, :, :) = sample_img;
            end

           % save flattened img (separately?)
           [~, shortname, ~] = fileparts(batch_fname);
           img_path = fullfile(OUT_IMG_DIR, [shortname '_' num2str(sample_i)]);
           data = single(sample_augs);
           file_name = batch_file_names(sample_i, :);
           save(img_path, 'data', 'file_name', '-v6');
           disp([datestr(datetime('now')) ' ' 'saved image at: ', img_path])
        end
    end
end