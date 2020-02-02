

`E`
`kg`: consistent geometric stiffness matrix
`L` : undeformed element length

### `k` - Local elastic stiffness matrix

    k = BasicLE2dFrm (L,[ElemData],v)

- stiffness matrix in basic system

        k  = [ EA/L       0       0;
                0    4*EI/L  2*EI/L;
                0    2*EI/L  4*EI/L];

- compatibility matrix in the presence of axial and/or moment releases

        ah = [1-RI(1)            0                    0;
                0            1-RI(2)        -0.5*(1-RI(3))*RI(2);
                0      -0.5*(1-RI(2))*RI(3)       1-RI(3)        ];

- transform stiffness matrix for the presence of releases

        k  = ah'*k*ah;

### `kg` - consistent geometric stiffness matrix

    kg = kg_2dFrm (ElemData.Geom,xyz,u,q)

### `ke` - transform stiffness matrix to global coordinates and add geometric stiffness

    ke = ag' * k * ag + kg;

ElemState.ke = ke;


 %% state determination
    % undeformed element length
    L   = ElmLenOr(xyz+GeomData.jntoff);
    % extract displacements from ElemState and reshape to array
    nen = size(xyz,2);
    u   = ExtrReshu(ElemState,ndf,nen);
    % transform end displacements from global reference to basic system
    [ag,bg,ab,v] = GeomTran_2dFrm (ElemData.Geom,xyz,GeomData,u);
    %% basic force-deformation
    [q,k] = BasicLE2dFrm (L,ElemData,v);

    %% transform stiffness and forces of basic system to global coordinates

    % determine equilibrium forces of basic system under element loads
    pbw = [-w(1)*L; -w(2)*L/2;  0;  0; -w(2)*L/2;  0];

    % transform basic forces to global coordinates and add end forces due to w
    p = bg * q + ab' * pbw;



    end
    ElemState.p = p;
    ElemState.ConvFlag = true;       % element does not involve iterations
    ElemResp = ElemState;
% ==========================================================================================
  case 'mass'
    %% lumped mass vector and consistent mass matrix
    ElemMass = Mass4Prism2dFrm (xyz,ElemData);
    ElemResp = ElemMass;
% ==========================================================================================
  case 'post'
    %% post-processing information - coordinates of deformed shape
    % undeformed element length
    L   = ElmLenOr(xyz+GeomData.jntoff);
    % extract displacements from ElemState and reshape to array
    nen = size(xyz,2);
    u   = reshape(ElemState.u,ndf,nen);
    % transform end displacements from global reference to basic system
    [~,~,~,v] = GeomTran_2dFrm (ElemData.Geom,xyz,GeomData,u);
    % basic force-deformation
    q = BasicLE2dFrm (L,ElemData,v);
    % determine deformations ve in the presence of releases, if any

    f = blkdiag(L/(E*A),L/(6*E*I).*[2 -1; -1 2]);  % flexibility matrix

    % initial deformations under uniformly distributed element loads and uniform initial deformations
    v0 = [w(1)*L^2/(2*E*A);
          w(2)*L^3/(24*E*I);
         -w(2)*L^3/(24*E*I)] + [e0(1)*L ; e0(2)*L/2.*[-1;1] ];
    % element deformations ve
    ve = f*q + v0;
    % post-processing information
    ElemPost.v  = v;
    ElemPost.q  = q;
    ElemPost.ve = ve;
    ElemResp = ElemPost;
% ==========================================================================================    
  case 'defo'
    ElemResp = str2func('DeformShape2dFrm');    
% ==========================================================================================  

%% ---- function BasicLE2dFrm --------------------------------------------------------------
function [q,k] = BasicLE2dFrm (L,ElemData,v)
state determination of basic 2d frame element


### initial element deformations due to element loading and non-mechanical effects

v0(1)   = w(1)*L*L/(2*EA) + e0(1)*L;
v0(2:3) = w(2)*L^3/(24*EI).*[1;-1] + e0(2)*L/2.*[-1;1];

###   basic force-deformation relation

q = k*(v-v0);