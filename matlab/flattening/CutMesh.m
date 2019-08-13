classdef CutMesh < handle
    %represents a sphere mesh which has been cut open, duplicated, glued 
    %to a torus and flattened to the plane
    
    properties
        %vertices locations of the flattened covering mesh on the plane
        V;
        %triangles of the flattenes covering mesh
        T;
        %the gluer object. Takes a mesh and creates a toric covering space
        gluer;
        %flattening object. Takes a mesh with toric topology and flattens
        %to to unit square
        flattener;
        %vertices locations on the original mesh 
        V_orig;
        %faces of the original mesh 
        T_orig;
        % the triangles of the original mesh, after subdivision
        T_divided; 
        % the locations of vertices in the original mesh, after subdivision
        V_divided; 
        %coordintes of vertex of fundamental domain parallelogram 
        bxy = [1 0];
        %coordintes of vertex of fundamental domain parallelogram 
        axy= [0 1];
        % array where A(i) is the index of the original on the divided mesh
        %of vertex i 
        inds_plane_to_divided_inds_mesh;
        % cellarry where A{i} contains an array of all 
        %duplicates on the plane, of the vertex i on the mesh
        inds_mesh_divided_to_inds_plane;
        % array where A(i) is the index of the original on the divided mesh
        %of triangle i on the plane
        Ts_plane_to_divided_Ts_mesh;
        % array where A(i) is the index of the original on the undivided
        % mesh of triangle i on the plane
        Ts_plane_to_Ts_orig_mesh;
        %array where A{i} are the indices of the triangles
        %in the cut mesh corresponding to triangle i in the divided original mesh
        divided_Ts_on_mesh_to_Ts_in_plane;
        %cell array where uncutTsTocutTs{i} is an array containig the
        %indices of triangles that were divided out of triangle i in the
        %original mesh
        uncutTsTocutTs;
        %an array where dividedTs2Ts(i) is the index of the triangles in
        %the original mesh that triangle i in the divided mesh was born
        %from
        dividedTs2Ts;
        % cell array where the i'th cell are the edges that were divided in
        % the i;th attemp to subdivide the mesh
        divided_edges={};
        %array of #triangles by degree_of_cover determinant of linaer
        %trans of each triangle from the sphere to the plane. The i'th row
        %is corresponds to the it'h triangle on the sphere.
        dets; 
        %frobenius norm of linear trans of each tri
        frobenius;
        %smaller singular value of linear trans of each tri
        smin; 
        %larger singular value of linear trans of each tri
        smax;
        %condition_numbers of linear trans of each tri
        condition_numbers 
        %return the conformal distortion per vertex, averaged over the
        %adjacent faces according to their area
        vertex_scale; 
        %distortion per face
        face_scale;
    end
    
    methods
        function obj=CutMesh(gluer,flattener)
            %set some attributes
            obj.gluer = gluer;
            obj.flattener = flattener;
            obj.V_orig = gluer.V_orig;
            obj.T_orig = gluer.T_orig;
            obj.V_divided = gluer.V_divided_mesh;
            obj.T_divided = gluer.T_divided_mesh;
            obj.V = flattener.V_plane;
            obj.T = flattener.T_cut_torus;
            obj.dividedTs2Ts = gluer.dividedTs2Ts;
            obj.divided_edges = gluer.divided_edges;
            % create map of triangles from the mesh to the flattened
            % covering mesh
            obj.divided_Ts_on_mesh_to_Ts_in_plane = cell(length(obj.T_divided),1);
            for i = 1:length(obj.T_divided)
                obj.divided_Ts_on_mesh_to_Ts_in_plane{i} = ...
                    (i:length(gluer.T_divided_mesh):gluer.d*length(gluer.T_divided_mesh))';
            end
            % create map of indices from flattened covering mesh to the
            % mesh
            obj.inds_plane_to_divided_inds_mesh = gluer.torus_to_sphere(flattener.I_cut_to_uncut);
            % create map of indices from the mesh to the flattened
            % covering mesh
            obj.inds_mesh_divided_to_inds_plane = cell(size(gluer.V_divided_mesh,1),1);
            for i = 1:length(obj.V)
                obj.inds_mesh_divided_to_inds_plane{obj.inds_plane_to_divided_inds_mesh(i)} = ...
                    [obj.inds_mesh_divided_to_inds_plane{obj.inds_plane_to_divided_inds_mesh(i)} ;i];
            end
            % create map of triangles from the flattened covering mesh 
            %to the origianl (divided) mesh
            obj.Ts_plane_to_divided_Ts_mesh = [];
            for i=1: gluer.d
                obj.Ts_plane_to_divided_Ts_mesh = [obj.Ts_plane_to_divided_Ts_mesh;...
                    (1:length(gluer.T_divided_mesh))'];
            end
            % create map of triangles from the flattened covering mesh 
            %to the origianl mesh
            obj.Ts_plane_to_Ts_orig_mesh = obj.dividedTs2Ts(obj.Ts_plane_to_divided_Ts_mesh);
            
            % create map of triangles from the original mesh to the
            % flattened covering mesh
            obj.uncutTsTocutTs = cell(length(obj.T_orig), 1);
            for i = 1:length(obj.T)
                obj.uncutTsTocutTs{obj.Ts_plane_to_Ts_orig_mesh(i)} = ...
                    [obj.uncutTsTocutTs{obj.Ts_plane_to_Ts_orig_mesh(i)}; i];
            end
            % compute area distortion, angles distortion of each face and vertex 
            obj.compute_scale;
       
        end
        
        function sort_uncut2cutInds_by_scale(obj)
            % sort preimages of each vertex by their scales
            for i=1:length(obj.inds_mesh_divided_to_inds_plane)
                distor = obj.vertex_scale(obj.inds_mesh_divided_to_inds_plane{i});
                x = sortrows([distor obj.inds_mesh_divided_to_inds_plane{i}], 'descend');
                obj.inds_mesh_divided_to_inds_plane{i} = x(:,2);
            end            
        end
        
        function sort_uncutTs2cutTs_by_scale(obj)
            % sort preimages of each triangle by their scales
            for i=1:length(obj.uncutTsTocutTs)
                distor = obj.face_scale(obj.uncutTsTocutTs{i});
                x = sortrows([distor obj.uncutTsTocutTs{i}], 'descend');
                obj.uncutTsTocutTs{i} = x(:,2);
            end
        end
        
        function compute_scale(obj)
            f = obj.flattener;
            g = obj.gluer;
            f.computeDistortion();
            obj.face_scale = f.dets;
            obj.dets = reshape(f.dets,length(f.dets)/g.d,g.d);
            obj.smin = reshape(f.smin,length(f.dets)/g.d,g.d);
            obj.smax = reshape(f.smax,length(f.dets)/g.d,g.d);
            obj.frobenius = reshape(f.frobenius,length(f.dets)/g.d,g.d);
            obj.condition_numbers = reshape(f.condition_numbers,length(f.dets)/g.d,g.d);
            obj.sort_uncutTs2cutTs_by_scale();
            obj.vertex_scale = obj.flattener.vertexScale();
            obj.sort_uncut2cutInds_by_scale();
        end
        
        function V_bary = get_centers_of_faces(~, V, T)
            nTs = size(T, 1);
            V_bary = zeros(nTs, 2);
            for i=1:nTs
                v1 = V(T(i, 1), :);
                v2 = V(T(i, 2), :);
                v3 = V(T(i, 3), :);
                V_bary(i, :) = (v1 + v2 + v3) / 3;
            end
        end
        
        function [f_on_mesh, f_on_flat] = liftImageVertex(obj, IM)
            % lifting the image (grid) back to the original mesh
            % inputs: IM - toric image (flattened torus)
            assert(size(IM, 1)==size(IM, 2),'image should be square!');
            sz = size(IM, 1);            
            V_flat = obj.V; 
            % shift vertexes to unit square
            V1 = V_flat(:, 1);
            V2 = V_flat(:, 2);
            V1 = V1 - floor(V1);
            V2 = V2 - floor(V2);
            sample_points = linspace(0,1,sz);
            [X, Y] = meshgrid(sample_points, sample_points);
            f_on_flat = interp2(X, Y, IM, V1, V2);
            % choose first vertex for each preimage (assuming they are
            % sorted by their scale..
            f_on_mesh = f_on_flat(cellfun(@(X) X(1), obj.inds_mesh_divided_to_inds_plane));
        end
        
        function [IM_on_mesh, IM_on_flat] = liftImageFaces(obj, IM)
            % lifting the image (grid) back to the original mesh
            % inputs: IM - toric image (flattened torus)
            assert(size(IM, 1)==size(IM, 2),'image should be square!');
            sz = size(IM, 1);
            % compute barycentric coordinates of faces
            V_bar = obj.get_centers_of_faces(obj.V, obj.T);
            % shift vertexes to unit sphere
            V1 = V_bar(:, 1) - floor(V_bar(:, 1));
            V2 = V_bar(:, 2) - floor(V_bar(:, 2));
            sample_points = linspace(0,1,sz);
            [X, Y] = meshgrid(sample_points, sample_points);
            IM_on_flat = interp2(X, Y, IM, V1, V2);
            % choose first vertex for each preimage (assuming they are
            % sorted by their scale..
            IM_on_mesh = IM_on_flat(cellfun(@(X) X(1), obj.uncutTsTocutTs));
        end
        function aggregate_fs_on_flat(obj, fs_on_flat, params)
            aggtype = getoptions(params, 'aggtype', 5);
            
            if strcmp(aggtype, 'old')
                IM_on_mesh = fs_on_flat(cellfun(@(X) X(1), obj.uncutTsTocutTs));
            elseif strcmp(aggtype, 'new')
                
            else
                error('aggtype %s is not supported', aggtype);
            end
        end
    end
end

