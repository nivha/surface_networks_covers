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
    data = load(batch_fname, 'data', 'target');
    batch_data = data.data;
    batch_target = data.target;

    % flatten batch
    batch_size = size(batch_data, 1);
    % batch_imgs = zeros(batch_size, params.sz, params.sz, nfeatures);
    for sample_i=1:batch_size
       sample = squeeze(batch_data(sample_i, :, :))';
       target = batch_target(sample_i);
       sample_img = push_functions_to_flattening_AE(cutMesh, sample, params);
%        if FLATTEN_TARGET
%            target = target';
%            target = push_functions_to_flattening_AE(cutMesh, target, params);
%        end
    
       % save flattened img (separately?)
       [~, shortname, ~] = fileparts(batch_fname);
       img_path = fullfile(OUT_IMG_DIR, [shortname '_' num2str(sample_i)]);
       data = single(sample_img);
       save(img_path, 'data', 'target', '-v6');
       disp([datestr(datetime('now')) ' ' 'saved image at: ', img_path])
    end

%     break
end
disp([datestr(datetime('now')) ' Done!']);

