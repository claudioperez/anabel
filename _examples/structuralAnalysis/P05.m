CleanStart
% Node Definitions
XYZ(1,:) = [ 0.0 0.0];
XYZ(2,:) = [ 8.0 0.0];
XYZ(3,:) = [ 8.0 6.0];
XYZ(4,:) = [16.0 6.0];

% Connections
CON(1,:) = [1 2];
CON(2,:) = [2 3];
CON(3,:) = [3 4];
CON(4,:) = [2 4];

% Boundary Conditions
BOUN(1,:) = [0 1 0];
BOUN(4,:) = [1 1 1];

% Specify element type
ElemName{1} = 'Lin2dFrm';
ElemName{2} = 'Lin2dFrm';
ElemName{3} = 'Lin2dFrm';
ElemName{4} = 'LinTruss';

% Create model
Model = Create_SimpleModel (XYZ,CON,BOUN,ElemName);

% Element properties

 ElemData = cell(Model.ne,1);

% Element: a
ElemData{1}.A = 100000000.0;
ElemData{1}.E = 1.0;
ElemData{1}.I = 50000;
ElemData{1}.Release = [0;0;0];

% Element: b
ElemData{2}.A = 100000000.0;
ElemData{2}.E = 1.0;
ElemData{2}.I = 50000;
ElemData{2}.Release = [0;0;0];

% Element: c
ElemData{3}.A = 100000000.0;
ElemData{3}.E = 1.0;
ElemData{3}.I = 50000;
ElemData{3}.Release = [0;0;0];

% Element: d
ElemData{4}.A = 30000.0;
ElemData{4}.E = 1.0;

%% Element loads

% Element: a
ElemData{1}.e0 = [0; 2e-3];

% Element: b
ElemData{2}.e0 = [0; -2e-3];

% Element: c
ElemData{3}.e0 = [0; 0e-3];

% Element: d
ElemData{4}.q0 = -76.3;

% plot parameters
% -------------------------------------------------------------------------
WinXr = 0.40;   % X-ratio of plot window to screen size 
WinYr = 0.80;   % Y-ratio of plot window to screen size
% -------------------------------------------------------------------------
NodSF = 3/4;    % relative size for node symbol
AxsSF = 3/4;    % relative size of arrows
HngSF = 3/4;    % relative size of releases
HOfSF = 1;      % relative size of hinge offset from end
% -------------------------------------------------------------------------
Create_Window (WinXr,WinYr);       % open figure window
PlotOpt.PlNod = 'yes';
PlotOpt.PlBnd = 'yes';
PlotOpt.NodSF = NodSF;
Plot_Model (Model,[],PlotOpt);
PlotOpt.AxsSF = AxsSF;
PlotOpt.LOfSF = 1.6.*NodSF;
Label_Model (Model,PlotOpt);

S_DisplMethod
%%
% plot curvature
Create_Window (WinXr,WinYr);       % open figure window
PlotOpt.PlNod = 'no';
Plot_Model  (Model,[],PlotOpt);    % plot model

Plot_2dCurvDistr (Model,ElemData,Q, [], 0.6);
Plot_Model (Model,[],PlotOpt);
% plot moments
Create_Window (WinXr,WinYr);       % open figure window
PlotOpt.PlNod = 'no';
Plot_Model  (Model,[],PlotOpt);    % plot model

Plot_2dMomntDistr (Model,ElemData,Q, [], 0.6);
Label_2dMoments(Model, Q)
Plot_Model (Model,[],PlotOpt);