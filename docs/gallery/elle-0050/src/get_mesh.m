function [node,tri,n,ne,nvar,np] = get_mesh(filename)
  fid = fopen(filename);
  n = fscanf(fid, '%d', 1);
  ne = fscanf(fid, '%d', 1);
  nvar = 2;
  if n>=1
    fgetl(fid); % read the newline
    s = fgetl(fid);  % read first line for size
    tmp = sscanf(s, '%f');
    nvar = size(tmp,1);
  end
  node = fscanf(fid, '%f', [nvar, n-1]);
  node = [tmp,node];

  np = 3;
  if ne>=1
    fgetl(fid); % read the newline
    s = fgetl(fid);  % read first line for size
    tmp = sscanf(s, '%d');
    np = size(tmp,1);
  end

  tri  = fscanf(fid, '%f', [np, ne-1]);
  tri = [tmp,tri];
  fclose(fid);
