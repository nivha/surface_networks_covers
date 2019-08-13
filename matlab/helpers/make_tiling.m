function [V_merged, T_merged] = make_tiling(cutMesh, k)


% position of the flattened parallelogram
ax = cutMesh.axy(1);
ay = cutMesh.axy(2);

V = cutMesh.V;
T = cutMesh.T;

V_merged = [];
T_merged = [];
T_shift = size(V, 1);
count = 0;
for x_trans=-k:1:k
    for y_trans=-k:1:k
        
        V_trans = V + x_trans*[1 0] + y_trans*[ax, ay];
       
        V_merged = [V_merged; V_trans];
        T_merged = [T_merged ; T + count * T_shift];
        count = count + 1;

    end
end
    
end