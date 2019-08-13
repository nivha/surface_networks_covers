disp('START process_batches_spherical_stuff')

gptoolbox_path = fullfile(PROJ_PATH, 'gptoolbox-master');
addpath(genpath(fullfile(pwd,'/../')))
addpath(genpath(gptoolbox_path))

% general parameters
params.sz = IMG_SIZE;

% load cutMesh object
data = load(CUTMESH_PATH, 'cutMesh');
cutMesh = data.cutMesh;

% create destination images folder if needed
[~, out_folder_num, ~] = fileparts(BATCHES_DIR);
OUT_IMG_DIR = fullfile(DST_DIR, out_folder_num);
disp([datestr(datetime('now')) ' ' 'OUT_IMG_DIR: ', OUT_IMG_DIR])
if ~ exist(OUT_IMG_DIR, 'dir')
   mkdir(OUT_IMG_DIR)
end

% process all batch files in folder
files = rdir(fullfile(BATCHES_DIR, '*.mat'));
fnames = {files.name};
for batch_i=1:numel(fnames)
   
    % retreive batch data 
    batch_fname = fnames{batch_i};
    disp([datestr(datetime('now')) ' ' 'processing: ', batch_fname])
    data = load(batch_fname, 'data', 'file_names');
    batch_data = data.data;
    batch_file_names = data.file_names;

    % flatten batch
    sample_augs = zeros(N_AUGMENT, params.sz, params.sz, N_FEATURES);
    batch_size = size(batch_data, 1) / N_AUGMENT;
    for sample_i=1:batch_size
        for jj=1:N_AUGMENT
           ii = N_AUGMENT * (sample_i-1) + jj;
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
