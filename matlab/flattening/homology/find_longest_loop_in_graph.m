function c = find_longest_loop_in_graph(A)
%takes an adjacency matrix A and returns the longest loop in the graph
gr = graph(A);
max_dist = -inf;
max_path = [];
% we use dfs to find all edges  in the graph that close a loop
d = gr.dfsearch(1,'edgetofinished');
% go over all of them and find the longest one
for i = 1:length(d)
    e = d(i,:);
    A_T = A;
    A_T(e(1),e(2)) = 0;
    A_T(e(2),e(1)) = 0;
    g = graph(A_T);
    [current_path,current_distance] = g.shortestpath(e(1),e(2));
    if current_distance >  max_dist
        max_dist = current_distance;
        max_path = current_path;
    end
end
c = [max_path, max_path(1)];
end