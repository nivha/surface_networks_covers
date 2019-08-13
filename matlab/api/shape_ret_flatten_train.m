function shape_ret_flatten_train(gptoolbox_path, batches_dir, dst_dir)
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
        data = load(batch_fname, 'data', 'target');
        batch_data = data.data;
        batch_target = data.target;

        % flatten batch
        batch_size = size(batch_data, 1);
        % batch_imgs = zeros(batch_size, params.sz, params.sz, nfeatures);
        for sample_i=1:batch_size
           sample = squeeze(batch_data(sample_i, :, :))';
           sample_img = push_functions_to_flattening_AE(cutMesh, sample, params);
        %    batch_imgs(i, :, :, :) = sample_img;

           % save flattened img (separately?)
           [~, shortname, ~] = fileparts(batch_fname);
           img_path = fullfile(OUT_IMG_DIR, [shortname '_' num2str(sample_i)]);
           data = single(sample_img);
           target = batch_target(sample_i);
           save(img_path, 'data', 'target', '-v6');
           disp([datestr(datetime('now')) ' saved image at: ', img_path])
        end
    end
end