function idx = chooseFarthestPoints(dist,n, v_init)

[~, idx]  = max(dist(v_init,:));
for ii = 2:n
  [~, idx(ii)]  = max(min(dist(idx,:),[],1));
end

end