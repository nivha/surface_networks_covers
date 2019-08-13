classdef Gluer < handle 
    properties
        %the triangles of the mesh after subdivision
        T_divided_mesh;
        %the points of the mesh after subdivision
        V_divided_mesh;
        %the adjacency matrix of the graph on the mesh
        A;
        %the cones on the mesh;
        cones;
        % the degree of the cover
        d;
        % the triangles of the d disks to be glued
        T_cut_sphere;
        % the points of the d disks to be glued
        V_cut_sphere;
        % the number of vertices in one cut sphere
        l;
        % the triangles of the glued torus
        T_torus;
        % the points of the glued torus
        V_torus;
        % seams is a cell array where seams{i} is 2*d times LENGTH_OF_SEAM array of the vertices
        % in path i in all of the spheres
        % copies 2*i-1 and 2*i of a seam are both in the i'th sphere
        seams;
        % product_one_tuple is a cell array of k by 1 where every
        % cell is an array of the cycles 
        product_one_tuple;
        % a list of indices from the torus to the d cut spheres
        torus_to_disks;
        % a list of indices from the torus to the uncut sphere
        torus_to_sphere
        % the points of the original mesh
        V_orig;
        % the triangles if the original mesh
        T_orig;
        % cell array where the i'th cell are the edges that were divided in
        % the i;th attemp to subdivide the mesh
        divided_edges={};
        % a vector where dividedTs2Ts(i) is the index of the triangle in
        % the original mesh that gave bith to triangle i in the divided
        % mesh
        dividedTs2Ts;

    end   
    methods
        function obj = Gluer(V,T,cones,product_one_tuple,genertic_vertex)
            %% input
            % V,T are the vertices and triangles of the original spherical mesh
            % cones is an array of indices of the cone points.
            % product_one_tuple is a cell array of k by 1 where every
            % cell is an array of the cycles
            % genertic_vertex is the index of a point on the mesh
            %% desription
            % We cut the sphere in the 
            % shape of a star centered at the inputed generic point, whose endpoints
            % are the cones. We now have a disc. Next, we duplicate the 
            % disk d times. The boundary of the disk is now k loops in the
            % sphere whose product is contractible.
            % the point at the center of the star appears k times in the
            % disc and each path on the boundary of the disc between two of its copies contains
            % exactly one cone points. We glue such a path containing the
            % i'th cones according to permutation product_one_tuple{i)
            %% 
            
            %get the degree of the covering
            obj.d = length(cell2mat(product_one_tuple{1}));
            obj.V_orig = V;
            obj.T_orig = T;
            % create the adjacency matrix of the original mesh
            obj.A = adjacency_matrix(obj.T_orig);
            obj.cones = cones;
            obj.product_one_tuple = product_one_tuple;
            % we should have same number of cones and permutations
            assert(length(obj.cones) == length(obj.product_one_tuple),...
                'there are %d permutations but %d cones given',length(obj.product_one_tuple),length(obj.cones));
            %% cut
            %cut the mesh to a disc topology and subdivide if necessary
            [G1,W1,cut_sphere_to_sphere,ds_cone_to_generic_vertex] = subdivide_and_cut(obj,genertic_vertex);
            % find the indices of the cones on the cut mesh
            [~,IA] = unique(cut_sphere_to_sphere);
            cones_after_cutting = IA(cones)';
            % get the number of vertices in the cut mesh
            obj.l = size(W1,1);
            % create the triangulation object of the cut mesh
            t1 = triangulation(G1,W1);
            %% copy the cut mesh to get d copies 
            obj.T_cut_sphere = G1;
            obj.V_cut_sphere = W1;
            for i = 1:obj.d-1
                obj.T_cut_sphere = [obj.T_cut_sphere ; G1 + obj.l * i];
                obj.V_cut_sphere = [obj.V_cut_sphere; W1];
            end
            %% compute the seams
            %the two copies of a seam on the boundary of the cut mesh are
            %the vertices between a cone and the two nearest copies of the generic vertex 
            
            %get the boundary of the cut mesh
            f = t1.freeBoundary;
            f =f(:,1);
            % circle the boundary so that the first vertex is the first
            % cone
            f = circshift(f,1-find(f==cones_after_cutting(end)));
            %find where the cones apppear on the bounday of the cut mesh
            ind = mod(find(f==cones_after_cutting),length(f));
            ind(ind==0) = length(f);
            ind = ind+length(f);
            f=[f;f;f];
            % seams is a cell array where seams{i} is 2*d times LENGTH_OF_SEAM array of the vertices
            % in path i in all of the spheres
            % copies 2*i-1 and 2*i of a seam are both in the i'th sphere
            obj.seams = cell(length(cones),1);
            for i = 1:length(cones)
                %in the boundary of the first mesh we go ds_cone_to_generic_vertex
                %backwards and forwards to get the two copies of the seam
                obj.seams{i} = [f(ind(i):ind(i) + ds_cone_to_generic_vertex(i)),...
                    f(ind(i):-1:ind(i) - ds_cone_to_generic_vertex(i))];
                % duplicate the two copies d times for each of the copies
                % of the cut mesh
                for j =1 :obj.d -1
                    obj.seams{i} = [obj.seams{i}, obj.seams{i}(:,[1,2]) + obj.l*j];
                end
            end
            %% glue the spheres to a torus
            %glue the d copies of the cut mesh to torus according to the
            %product one tuple
            [obj.V_torus,obj.T_torus] = glue(obj);
            %take the map from the vertices of the torus to the d cut meshs. Take mode l
            %number of vertices in the cut mesh. get a map from the covering torus to one cut mesh 
            obj.torus_to_sphere = mod(obj.torus_to_disks,obj.l);
            obj.torus_to_sphere(obj.torus_to_sphere == 0) = obj.l;
            % compose with a map from cut mesh to uncut mesh to get a map
            % from the covering torus to the original (divided) mesh
            obj.torus_to_sphere = cut_sphere_to_sphere(obj.torus_to_sphere);
        end
        function [W,G]  = glue(obj)
        %% takes the d copies of the cut sphere and glue them to a torus 
        %%= according to the information encoded in the product one tuple
        G = obj.T_cut_sphere;
        for i =1:length(obj.seams)
            gluing_instructinos = convert_permutation_to_gluing_instructions(obj,i);
            G = glue_seam_according_to_configuration(obj,G,i,gluing_instructinos);
        end
        % after gluing there are redundent vertices. Remove them
        [W,I,obj.torus_to_disks] = remove_unreferenced(obj.V_cut_sphere,G(:));
        G = I(G);
        % consistently orient
        G = bfs_orient(G);
        % make sure everything worked out fine
        S = statistics(W,G,'Fast',true);
        assert(S.num_connected_components ==1,'the covering manifold is not connected');
        assert(S.num_boundary_edges ==0,'the covering manifold has a boundary');
        assert(S.num_conflictingly_oriented_edges==0, 'the covering manifold is not orientable');
%         assert(S.euler_characteristic ==0,'the covering manifold has bad euler char %d',S.euler_characteristic );          
        end
        function G = glue_seam_according_to_configuration(obj,G,seam_index,p)
        %%==== G is the triangles matrix
        % p is a d by 2 array such that seam(:,p(j,1)) is glued 
        % to seam(:,p(j,2))
            for j =1:obj.d
                E1 = obj.seams{seam_index}(:,p(j,1));
                E2 = obj.seams{seam_index}(:,p(j,2));
                for i = 1:length(E1)
                    G(G==E1(i)) = E2(i);
                end
            end
        end 
        function gluing_instructions = convert_permutation_to_gluing_instructions(obj,index_of_branch_points)
            % input: cell array where each cell is an array contining the cycles. e.g. {[1,2,3] [4]} 
            % output: gluing_instructions is a d by 2 array such that seam(:,gluing_instructions(j,1)) is glued 
            % to seam(:,gluing_instructions(j,2)) 
            s = obj.product_one_tuple{index_of_branch_points};
            gluing_instructions = zeros(obj.d,2);
            index = 1;
            for j= 1:numel(s)
                cycle = s{j};
                for k = 1:length(cycle)-1
                    gluing_instructions(index,1) = 2*cycle(k)-1;
                    gluing_instructions(index,2) = 2*cycle(k+1);
                    index = index+1;
                end
                gluing_instructions(index,1)= 2*cycle(end)-1;
                gluing_instructions(index,2)= 2*cycle(1);
                index = index+1;
            end
        end
        function [G,W,I,ds_cone_to_generic_vertex] = subdivide_and_cut(obj, genertic_vertex)
            %%
            %== takes the mesh and cuts it to a disc topology along a star whos center
            %== is a specified generic vertex and whos enpoints are the
            %== cones. If no such paths exists, locally subdivide the original mesh. 
            %== we cut in stages. At step i there will be i-1 copies of the
            %== generic vertex on the boundary of the cut mesh. We cut along a shortest path from the
            %== i'th cone to the first copy of the genric vertex in the
            %== boundary 
            
            % at first set the divided mesh to be equal to the original mesh
            obj.dividedTs2Ts = (1:length(obj.T_orig))';
            obj.V_divided_mesh = obj.V_orig;
            obj.T_divided_mesh = obj.T_orig;
            while true
                %initialize the array holding the distances between the
                %cones and the genric point
                ds_cone_to_generic_vertex = zeros(length(obj.cones),1);
                %create a local copy of cones indices
                cones = obj.cones;
                %update the adjacenct matrix of the divided mesh
                AA = obj.A;
                % find a path from the generic point to the first cone that
                % intersect none of the other cones
                cones_t = setdiff(cones,cones(1));
                AA(cones_t,:) = 0;
                AA(:,cones_t) = 0;
                [ds_cone_to_generic_vertex(1),p] = graphshortestpath(AA,cones(1),genertic_vertex);
                % get the edges in the path
                E = [p(1:end-1)' p(2:end)'];
                % cut along the first path
                [G,I] = cut_edges(obj.T_divided_mesh,E);
                I1=I;
                %get the indices of the cones and the generic vertex in the
                %mesh after cutting once
                [~,IA] = unique(I1);
                cones = IA(cones);
                none_cones = IA(genertic_vertex);
                W = obj.V_divided_mesh(I,:);
                %try cutting cones 2 throgh k
                for i = 2:length(obj.cones)
                    %locate the correct copy of the none cone point on the
                    %bounday. We always take the first copy
                    t = triangulation(G,W);
                    % get f, the boundary of the cut mesh
                    f = t.freeBoundary();
                    f = f(:,1);
                    % find where the first cone is on the buondary
                    ind1 = find(f == cones(1));
                    % circle f so that the first cone will be the first
                    % index on the boundary of the cut mesh
                    f = circshift(f,1-ind1);
                    %find the first copy of the generic point
                    non_cone = f(1+ds_cone_to_generic_vertex(1));
                    % try and find a path from the current cone to the
                    % first copy of the generic vertex that does not
                    % intersect the boundary of the cut mesh or any of the
                    % other cones
                    ff = setdiff(f,non_cone);
                    AA = adjacency_matrix(G);   
                    AA(ff,:) = 0;
                    AA(:,ff) = 0;
                    cones_t = setdiff(cones,cones(i));
                    AA(cones_t,:) = 0;
                    AA(:,cones_t) = 0;                     
                    [ds_cone_to_generic_vertex(i),p] = graphshortestpath(AA,cones(i),non_cone);
                    %if we could find no path then subdivide the mesh and
                    %try again
                    if ds_cone_to_generic_vertex(i) == inf
                        %subdivide the mesh
                        [obj.V_divided_mesh, obj.T_divided_mesh, cutE, J] = subdivide_mesh_along_line(obj,G,W,non_cone,cones(i),I);
                        % update the map from the divided triangles to the
                        % undivided triangles
                        obj.dividedTs2Ts = obj.dividedTs2Ts(J);
                        % add a cell containing the edges cut in the
                        % current cutting attempt
                        obj.divided_edges{end+1} = cutE;
                        %update the adjecency matrix
                        obj.A = adjacency_matrix(obj.T_divided_mesh);
                        break
                    end
                    %if we could find a path, get the edges and continue
                    %cutting
                    E = [p(1:end-1)' p(2:end)'];
                    [G,I1] = cut_edges(G,E);
                    [~,IA] = unique(I1);
                    cones = IA(cones);
                    %%find the indices of the copies of the generic vertex
                    %%in the boundary of the cut mesh
                    none_cones_new = [];
                    for j =1:length(none_cones)
                        fin = find(I1==none_cones(j));
                        none_cones_new = [none_cones_new; fin];
                    end
                    none_cones = none_cones_new;
                    %update I the map from cut vertices to uncut vertices
                    I = I(I1);
                    W = W(I1,:);
                end
                % if at the last cut there exists a path from the correct
                % copy of the generic point to the last point, then we are
                % done.
                if i==length(obj.cones) && ds_cone_to_generic_vertex(i)<inf
                    break;
                end
            end
        end
        function [V,T,cutE,J] =  subdivide_mesh_along_line(obj,G,W,P0_index,P1_index,I)
            %== takes the mesh and the indices of two vertices. flattens the
            %== mesh to a disc, finds the line between the two flattened
            %== vertices and divide all the faces the line passes 
            % flatten the mesh
            u = Tutte_disk_topology(G,W);
            %get the points from their indices
            P0 = u(P0_index,:);
            P1 = u(P1_index,:);
            %find all the edges the line between the points intersect
            E = find_edges_intersecting_with_line(G,u,P0,P1);
            %ignore the edges on the boundary
            tt  = triangulation(G,u);
            f = tt.freeBoundary();
            f = [f; f(:,[2 1])];
            E = setdiff(E,f,'rows');
            %move the edges to the edges in the original mesh
            E = I(E);
            %divide the faces
            [V, T, J, cutE] = my_split_edges(obj.V_divided_mesh, obj.T_divided_mesh,E);
        end
    end
end
