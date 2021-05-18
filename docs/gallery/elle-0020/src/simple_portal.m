%function State = elle_0020%(Ph, ColumnArea, GirderArea, Geom)
% Claudio Perez

%% -------------------------
% Variables
%---------------------------
ft = 12.0;
B = 30.*ft; % Base
H = 13.*ft; % Height

Ph = 2e3;
Pv = -2e3;
ColumnE = ConcMod;
GirderE = ConcMod;

GirderArea = 684.0;
GirderHomMOI = 34383.8; % Homogeneous section MOI
ColumnArea = 576.0;
ColumnHomMOI = (24^4)/12.0;

% Structural geometry option
GirderGeom = Geom;
ColumnGeom = Geom; % linear, PDelta, or corotational
%% -------------------------
% Element Definitions
%---------------------------
nel=4;
for el=1:nel
	ElemName{el} = 'LE2dFrm';
end

for el = [1,4]
  ElemData{el}.E = ColumnE;
  ElemData{el}.A = ColumnArea;
  ElemData{el}.I = ColumnHomMOI;
  ElemData{el}.Geom = ColumnGeom;
end
for el = [2,3]
  ElemData{el}.E = GirderE;
  ElemData{el}.A = GirderArea;
  ElemData{el}.I = GirderHomMOI;
  ElemData{el}.Geom = GirderGeom;
end

%% -------------------------
% Model Definition
%---------------------------
% Set up nodes
XYZ(1,:) = [  0.,  0.];
XYZ(2,:) = [  0.,  H ];
XYZ(3,:) = [ B/2,  H ];
XYZ(4,:) = [  B ,  H ];
XYZ(5,:) = [  B ,  0.];

CON{1} = [ 1, 2];
CON{2} = [ 2, 3];
CON{3} = [ 3, 4];
CON{4} = [ 4, 5];

BOUN(1,:)= [1,1,1];
BOUN(5,:)= [1,1,1];

% generate Model data structure
Model = Create_Model(XYZ,CON,BOUN,ElemName);
ElemData = Structure ('chec',Model,ElemData);

%% -------------------------
% Loading
%---------------------------
Pe(2,1) = Ph;
Pe(2,2) = Pv;
Pe(4,2) = Pv;
Loading = Create_Loading(Model,Pe,[]);

%% -------------------------
% Solution
%---------------------------
SolStrat = Initialize_SolStrat;
State = Initialize_State(Model,ElemData);
[State,SolStrat] = Initialize(Model,ElemData,Loading,State,SolStrat);
[State,SolStrat] = Increment(Model,ElemData,Loading,State,SolStrat);
[State,SolStrat] = Iterate(Model,ElemData,Loading,State,SolStrat);


