function flatten_mesh(gptoolbox_path, mesh_path, segmentation_path, data_dir, split, n_augment)
    % add paths
    disp(['pwd: ' pwd])
    addpath(genpath(pwd))
    addpath(genpath('../'))
    addpath(genpath(gptoolbox_path))
    
    % get file name parts
    [~, shortname, ext] = fileparts(mesh_path);
    disp([datestr(datetime('now')) ' mesh file: ', mesh_path])

    % read mesh file
    ext = ext(2:end);
    if strcmp(ext, 'ply')
        [V, T] = read_ply(mesh_path);
    elseif strcmp(ext, 'off')
        [V, T] = readOFF(mesh_path);
    elseif strcmp(ext, 'obj')
        [V, T] = readOBJ(mesh_path);
    else
        error('Extension %s is not supported', ext)
    end

    % read segmentation file
    disp([datestr(datetime('now')) ' segmentation file: ', segmentation_path]);
    GTsegmentation = textread(segmentation_path);

    % normalize mesh
    V = V / sqrt(CORR_calculate_area(T,V));
    % centralize mesh
    V = V - mean(V, 1);

    % compute cones on the mesh
    % if you want a different ramification tructure - this where you need
    % to input a different tuple of permutations
    product_one_tuple = {{[6, 7, 8, 9, 10],[1],[2],[3],[4],[5]}, ...
                         {[1, 7, 3, 4, 9],[2],[5],[6],[8],[10]}, ...
                         {[1, 8, 4, 3, 7],[2],[5],[6],[9],[10]}, ...
                         {[2, 5, 4, 7, 6],[1],[3],[8],[9],[10]}, ...
                         {[2, 10, 9, 4, 5],[1],[3],[6],[7],[8]}};
    k = length(product_one_tuple);
    params.nFarthest = k;
    params.doplot = 0;
    [cones, AGD] = getPointTripletsByFarthestPointSampling(V, T, params);

    % sample cones permutations
    ps = perms(1:k);

    % for each cones compute parameterization and push functions
    for i=1:n_augment
        % seed random generator
        seed = str2double(datestr(now, 'HHMMSSFFF'));       
        disp([datestr(datetime('now')) ' random seed: ' num2str(seed)]);
        rng(seed)

        % permute cones
        pi = randi([1 length(ps)]);
        perm = ps(pi, :);
        disp([datestr(datetime('now')) ' cones permutation: ' sprintf('%d', perm)]);
        cones = cones(perm);

        % set filename
        filename = [shortname '_cones_' sprintf('%d', perm) '_' num2str(i)];
        disp([datestr(datetime('now')) ' working on ', filename]);

        % get random orthonormal (rotation+reflection) matrix and scale in 0.85-1.15
        [q,r] = qr(randn(3)); q = q * diag(sign(diag(r))) * ((randi([0 1]) * 2) - 1);
        rand_scale = rand(1) * 0.3 + 0.85;
        % rotate and scale V
        V = V * q;
        V = V * rand_scale;

        % flatten mesh to R2
        [~, min_agd_point] = min(AGD);
        [cutMesh, ~, ~] = flatten_sphere(V ,T, cones, min_agd_point, product_one_tuple);

        params.doplot = 0;
        functionsOnMesh = V;
        [dataFunctions, dataSeg] = push_functions_to_flattening(cutMesh, GTsegmentation, functionsOnMesh, params);
        disp([datestr(datetime('now')) ' finished pushing functions to grid']);

        % save image
        ext = '.mat';
        img_path = fullfile(data_dir, 'images', [filename ext]);
        data = single(dataFunctions);
        save(img_path, 'data', '-v6');
        disp([datestr(datetime('now')) ' saved ' img_path]);
        % save label
        label_path = fullfile(data_dir, 'labels', [filename ext]);
        data = uint8(dataSeg);
        save(label_path, 'data', '-v6');
        disp([datestr(datetime('now')) ' saved ' label_path]);

        % save flattening info
        flat_path = fullfile(data_dir, 'flat_info', filename);
        if strcmp(split, 'test')
            cutMesh.compute_scale()
            save(flat_path, 'cutMesh', 'cones', 'seed', 'q', 'rand_scale', 'min_agd_point');
        else
            save(flat_path, 'cones', 'seed', 'q','rand_scale', '-v6');
        end
        disp([datestr(datetime('now')) ' saved ' flat_path '.mat']);
    end

    disp('-------------------------');
    disp([datestr(datetime('now')) ' ' 'Done Processing mesh!']);
    
end