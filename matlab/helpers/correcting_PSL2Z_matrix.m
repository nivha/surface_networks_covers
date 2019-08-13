function best = correcting_PSL2Z_matrix(u,v)
    %% input: two Z linearly independent vectors u,v in R^2. 
    %% output: g in PSL2Z such that g[u;v] is closer to being a square
    % we scan for matrices d in PSL2Z in a range and take the one
    % with lowest cost
    assert(all(size(u)==[1,2]),' u should be a 1 by 2 row vector');
    assert(all(size(v)==[1,2]),' v should be a 1 by 2 row vector');
    % make u,v column vectors
    epsilon = 0.01;
    u = u';
    v = v';
    G = [(norm(u,2)^2 - norm(v,2)^2)/2, u'*v;v'*u, (norm(v,2)^2 - norm(u,2)^2)/2];
    min_cost = norm(G,'fro');
    best = [1 0; 0 1];
    for a = -5:5
        for b = -5:5
            for c = -5:5
                if mod(1 + b*c, a) == 0
                    d = (1 + b*c)/a;
                    uuvv = [a b; c d]*[u';v'];
                    uu = uuvv(1,:);
                    vv = uuvv(2,:);
                    G = [(norm(uu,2)^2 - norm(vv,2)^2)/2,uu*vv';...
                        vv*uu', (norm(vv,2)^2 - norm(uu,2)^2)/2];
                    cost = norm(G,'fro');
                    if cost < epsilon
                        best = [a b; c d];
                        return;
                    end
                    if cost<min_cost
                        min_cost = cost;
                        best = [a b; c d];
                    end
                end
            end
        end
    end
end