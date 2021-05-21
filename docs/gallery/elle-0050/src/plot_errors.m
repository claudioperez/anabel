fname = 'circle_iso/mesh3';

[node, tri, n, ne, nvar, np] = get_mesh(fname);

x = node(1,:);
y = node(2,:);
e = zeros(1,ne);

for i=1:ne
  e(i) = rand(1,1); % your job: compute errors on each triangle
end


if np==3
  tri1 = tri;
else
  tri1 = tri([1 4 2 5 3 6],:); % re-order nodes ccw
end

clf;
patch(x(tri1),y(tri1),e);
colorbar;
%shading flat  % <-- for finer meshes
