[node,tri,n,ne,nvar,np] = get_mesh('circle_iso/mesh3');
[node1,tri1,n1,ne1,nvar1,np1] = get_mesh('circle_iso/mesh3lin');

x = node(1,:)';
y = node(2,:)';

r = sqrt(x.^2 + y.^2);
u = cos(0.5*pi*r);  % exact solution. can replace with FE solution


% plot using the finer mesh with each triangle broken up into 4 subtrianlges
trisurf(tri1',x,y,0*x,u);
view(2);
shading interp;
h = colorbar;
axis equal;

% use the vertices of the original quadratic elements to draw the mesh
hold on;
trimesh(tri(1:3,:)',x,y,1+0*x,0*x,'EdgeColor','black','FaceColor','none');
hold off;
