%Create nodes
XYZ(1,:) = [  0  0];
XYZ(2,:) = [  0  2];
XYZ(3,:) = [  3  2];
XYZ(4,:) = [  3  4];
XYZ(5,:) = [  6  4];
XYZ(6,:) = [  9  4];
XYZ(7,:) = [  9  2];
XYZ(8,:) = [ 12  2];
XYZ(9,:) = [ 12  0];

%Create elems
ne = 1
CON(ne, :) = [1, 2];
ne = ne+1;
CON(ne, :) = [1, 3];
ne = ne+1;
CON(ne, :) = [2, 3];
ne = ne+1;
CON(ne, :) = [2, 4];
ne = ne+1;
CON(ne, :) = [3, 4];
ne = ne+1;
CON(ne, :) = [3, 5];
ne = ne+1;
CON(ne, :) = [4, 5];
ne = ne+1;
CON(ne, :) = [5, 6];
ne = ne+1;
CON(ne, :) = [5, 7];
ne = ne+1;
CON(ne, :) = [6, 7];
ne = ne+1;
CON(ne, :) = [6, 8];
ne = ne+1;
CON(ne, :) = [7, 8];
ne = ne+1;
CON(ne, :) = [7, 9];
ne = ne+1;
CON(ne, :) = [8, 9];


BOUN(1, :)  = [1 1];
BOUN(9, :)  = [1 1];

[ElemName{1:14}] = deal('Truss');     % truss element

Model = Create_SimpleModel(XYZ,CON,BOUN,ElemName);


% plot parameters
% -------------------------------------------------------------------------
WinXr = 0.40;  % X-ratio of plot window to screen size 
WinYr = 0.80;  % Y-ratio of plot window to screen size
% -------------------------------------------------------------------------
%% Post-processing functions on Model (optional)
Create_Window (WinXr,WinYr);       % open figure window
PlotOpt.PlNod = 'yes';
PlotOpt.PlBnd = 'yes';
Plot_Model  (Model,[],PlotOpt);    % plot model
% PlotOpt.AxsSF = 2;
Label_Model (Model,PlotOpt);       % label model

%% specify element properties
ne = Model.ne;   % number of elements in structural model
ElemData = cell(ne,1);
for el=1:ne
   ElemData{el}.E = 1000;       % elastic modulus
   ElemData{el}.A = 10;         % area for EA = 10000
   ElemData{el}.I = 5;          % moment of inertia for EI = 5000
end


%% Loading
% nodal forces for w
Pe(4,2) = -12;
Pe(8,1)  = -18;
Pf = Create_NodalForces(Model,Pe);



Create_Window (WinXr,WinYr);       % open figure window
Plot_Model  (Model,[],PlotOpt);    % plot model
PlotOpt.FrcSF = 10*ne/4;
PlotOpt.TipSF = 2;
PlotOpt.PlNod = 'no';
PlotOpt.PlBnd = 'yes';
Plot_NodalForces (Model,Pe,PlotOpt);

S_DisplMethod

% display axial force distribution
Create_Window (WinXr,WinYr);
Plot_Model (Model)
Plot_AxialForces(Model,Q)
Label_AxialForces(Model,Q,1:2:ne,2)

% display bending moment distribution
Create_Window (WinXr,WinYr);
Plot_Model (Model)
Plot_2dMomntDistr(Model,[],Q,[],2)
Label_2dMoments(Model,Q,1:2:ne,2)

Uf(Model.DOF(ne/2+1,2),1)
Q(1)