%% Matlab script for Problem 2 of Hw Set 2

% Department of Civil and Environmental Engineering
% University of California, Berkeley
% CE 220:     Structural Analysis
% Instructor: Professor Filip C. Filippou
%
%% Clear workspace memory and initialize global variables
CleanStart

%% Create model
% select number of support points
ns = 3;
% specify node coordinates for support points
% angle for supports
phi = (0:1:ns-1)./ns.*2.*pi;
% outer support radius
Ro = 20;
% coordinates for support points
X  = cos(phi).*Ro;
Y  = sin(phi).*Ro;
% generate support points with height Z of 0
XYZ( 1:ns,:) = [X' Y' zeros(ns,1)];
% angles for upper ring (offset by pi/ns degrees from supports)
phi = phi+pi/ns;
% inner upper ring radius
Ri  = 10;
% coordinates for upper ring
X   = cos(phi).*Ri;
Y   = sin(phi).*Ri;
% generate coordinates for upper ring with height H
H  = 5;
XYZ( ns+1:2*ns,:) = [X' Y' H.*ones(ns,1)];

% connectivity array
CON (1:ns,:)          = [ (1: ns)'     (ns+1:2*ns)' ];
CON (ns+1,:)          = [ 1 2*ns];
CON (ns+2:2*ns,:)     = [ (2:ns)'      (ns+1:2*ns-1)'];
CON (2*ns+1:3*ns-1,:) = [ (ns+1:2*ns-1)' (ns+2:2*ns)'];
CON (3*ns,:)          = [ ns+1 2*ns];

% connectivity array
(1:ns)          %= [ (1: ns)'     (ns+1:2*ns)' ];
(ns+1)          %= [ 1 2*ns];
(ns+2:2*ns)     %= [ (2:ns)'      (ns+1:2*ns-1)'];
(2*ns+1:3*ns-1) %= [ (ns+1:2*ns-1)' (ns+2:2*ns)'];
(3*ns)          %= [ ns+1 2*ns];



% boundary conditions (1 = restrained,  0 = free) (specify only restrained dof's)
BOUN( 1:ns,:) = repmat([1 1 1],ns,1);

% specify element type
ne = length(CON);
[ElemName{1:ne}] = deal('Truss');    % linear truss element

% create model
Model = Create_SimpleModel(XYZ,CON,BOUN,ElemName);
 
% plot parameters
% -------------------------------------------------------------------------
WinXr = 0.40;  % X-ratio of plot window to screen size 
WinYr = 0.80;  % Y-ratio of plot window to screen size
% -------------------------------------------------------------------------
NodSF = 0.60;  % relative size for node symbol
% -------------------------------------------------------------------------
% plot and label model for checking (optional)
Create_Window (WinXr,WinYr);       % open figure window
PlotOpt.PlNod = 'yes';
PlotOpt.PlBnd = 'yes';
PlotOpt.NodSF = NodSF;
Plot_Model  (Model,[],PlotOpt);    % plot model
PlotOpt.AxsSF = 0.6;
Label_Model (Model,PlotOpt);       % label model

Create_Window (WinXr,WinYr);
Plot_Model  (Model,[],PlotOpt);    % plot model
Label_Model (Model);
% plan view rotated 90 deg
view([90 90])

Create_Window (WinXr,WinYr);
Plot_Model  (Model,[],PlotOpt);    % plot model
Label_Model (Model);
% elevation rotated 90 deg
view([90 0])