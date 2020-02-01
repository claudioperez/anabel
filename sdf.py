import numpy as np

class GroundMotion:
    """Seismic parameters of an acceleration time history

     param=OpenSeismoMatlab(time,xgtt,sw,baselineSw,dti,ksi,T,mu,AlgID)

     Description
     ------------------------------------------
         This function calculates the seismic parameters from an acceleration
         time history. More specifically, it calculates the following:
         1) Velocity vs time
         2) Displacement vs time
         3) Resampled acceleration time history (i.e. the input acceleration
         time history with modified time step size)
         4) Peak ground acceleration
         5) Peak ground velocity
         6) Peak ground displacement
         7) Total cumulative energy and normalized cumulative energy vs time
         8) Significant duration according to Trifunac & Brady (1975)
         9) Total Arias intensity (Ia)
         10) Linear elastic pseudo-acceleration response spectrum
         11) Linear elastic pseudo-velocity response spectrum
         12) Linear elastic displacement response spectrum
         13) Linear elastic velocity response spectrum
         14) Linear elastic acceleration response spectrum
         15) Constant ductility displacement response spectrum
         16) Constant ductility velocity response spectrum
         17) Constant ductility acceleration response spectrum
         18) Fourier amplitude spectrum
         19) Mean period (Tm)

     Input parameters
     ---------------------------------
         #dt# (scalar) is the size of the time step of the input acceleration
             time history #xgtt#.
         #xgtt# ([#numsteps# x 1]) is the input acceleration time history.
         #sw# (string) is a string which determines which parameters of the
             input acceleration time history will be calculated. #sw# can take
             one of the following values (strings are case insensitive):
             'TIMEHIST': the displacement, velocity and acceleration time
                 histories are calculated.
             'RESAMPLE': the acceleration time history with modified time step
                 size is calculated.
             'PGA': The peak ground acceleration is calculated.
             'PGV': The peak ground velocity is calculated.
             'PGD': The peak ground displacement is calculated.
             'ARIAS': The total cumulative energy, significant duration
                 according to Trifunac & Brady (1975) and Arias intensity are
                 calculated.
             'ES': The linear elastic response spectra and pseudospectra are
                 calculated.
             'CDS': The constant ductility response spectra are calculated.
             'FAS': The Fourier amplitude spectrum and the mean period are
                 calculated.
         #baselineSw# (boolean) determines if baseline correction will be
             applied for the calculation of the various output quantities.
         #dti# (scalar) is the new time step size for resampling of the input
             acceleration time history.
         #ksi# (scalar) is the fraction of critical viscous damping.
         #T# ([#numSDOFs# x 1]) contains the values of eigenperiods for which
             the response spectra are requested. #numSDOFs# is the number of
             SDOF oscillators being analysed to produce the spectra.
         #mu# (scalar) is the specified ductility for which the response
             spectra are calculated.
         #AlgID# (string as follows) is the algorithm to be used for the time
             integration, if applicable. It can be one of the following
             strings for superior optimally designed algorithms (strings are
             case sensitive):
                 'generalized a-method': The generalized a-method (Chung &
                 Hulbert, 1993)
                 'HHT a-method': The Hilber-Hughes-Taylor method (Hilber,
                 Hughes & Taylor, 1977)
                 'WBZ': The Wood烹ossak忙ienkiewicz method (Wood, Bossak &
                 Zienkiewicz, 1980)
                 'U0-V0-Opt': Optimal numerical dissipation and dispersion
                 zero order displacement zero order velocity algorithm
                 'U0-V0-CA': Continuous acceleration (zero spurious root at
                 the low frequency limit) zero order displacement zero order
                 velocity algorithm
                 'U0-V0-DA': Discontinuous acceleration (zero spurious root at
                 the high frequency limit) zero order displacement zero order
                 velocity algorithm
                 'U0-V1-Opt': Optimal numerical dissipation and dispersion
                 zero order displacement first order velocity algorithm
                 'U0-V1-CA': Continuous acceleration (zero spurious root at
                 the low frequency limit) zero order displacement first order
                 velocity algorithm
                 'U0-V1-DA': Discontinuous acceleration (zero spurious root at
                 the high frequency limit) zero order displacement first order
                 velocity algorithm
                 'U1-V0-Opt': Optimal numerical dissipation and dispersion
                 first order displacement zero order velocity algorithm
                 'U1-V0-CA': Continuous acceleration (zero spurious root at
                 the low frequency limit) first order displacement zero order
                 velocity algorithm
                 'U1-V0-DA': Discontinuous acceleration (zero spurious root at
                 the high frequency limit) first order displacement zero order
                 velocity algorithm
                 'Newmark ACA': Newmark Average Constant Acceleration method
                 'Newmark LA': Newmark Linear Acceleration method
                 'Newmark BA': Newmark Backward Acceleration method
                 'Fox-Goodwin': Fox-Goodwin formula

     Output parameters
     --------------------------------------------------
         #param# (structure) has the following fields:
             param.vel ([#numsteps# x 1]) Velocity vs time
             param.disp ([#numsteps# x 1]) Displacement vs time
             param.PGA (scalar) Peak ground acceleration
             param.PGV (scalar) Peak ground velocity
             param.PGD (scalar) Peak ground displacement
             param.Ecum (scalar) Total cumulative energy
             param.EcumTH ([#numsteps# x 1]) normalized cumulative energy vs
             time
             param.t_5_95 ([1 x 2]) Time instants at which 5# and 95# of
             cumulative energy have occurred
             param.Td (scalar) Time between when 5# and 95# of cumulative
             energy has occurred (significant duration according to
             Trifunac-Brady (1975))
             param.arias (scalar) Total Arias intensity (Ia)
             param.PSa ([#n# x 1]) Linear elastic pseudo-acceleration response
             spectrum
             param.PSv ([#n# x 1]) Linear elastic pseudo-velocity response
             spectrum
             param.Sd ([#n# x 1]) Linear elastic displacement response
             spectrum
             param.Sv ([#n# x 1]) Linear elastic velocity response spectrum
             param.Sa ([#n# x 1]) Linear elastic acceleration response
             spectrum
             param.SievABS ([#n# x 1]) Linear elastic absolute input energy
             equivalent velocity spectrum
             param.SievREL ([#n# x 1]) Linear elastic relative input energy
             equivalent velocity spectrum
             param.PredPSa (scalar) Predominant acceleration of the PSa
             spectrum
             param.PredPeriod (scalar) Predominant period of the PSa spectrum
             param.CDPSa ([#n# x 1]) Constant ductility pseudo-acceleration
             response spectrum
             param.CDPSv ([#n# x 1]) Constant ductility pseudo-velocity
             response spectrum
             param.CDSd ([#n# x 1]) Constant ductility displacement response
             spectrum
             param.CDSv ([#n# x 1]) Constant ductility velocity response
             spectrum
             param.CDSa ([#n# x 1]) Constant ductility acceleration response
             spectrum
             param.fyK ([#n# x 1]) yield limit that each SDOF must have in
             order to attain ductility equal to param.muK.
             param.muK ([#n# x 1]) achieved ductility for each period (each
             SDOF).
             param.iterK ([#n# x 1]) number of iterations needed for
             convergence for each period (each SDOF).
             param.FAS ([#2**(nextpow2(len(#xgtt#))-1)# x 1]) Fourier
             amplitude spectrum
             param.Tm (scalar) Mean period (Tm)
             param.Fm (scalar) Mean frequency (Fm)

    __________________________________________________________________________
     Copyright (c) 2018-2019
         George Papazafeiropoulos
         Captain, Infrastructure Engineer, Hellenic Air Force
         Civil Engineer, M.Sc., Ph.D. candidate, NTUA
         Email: gpapazafeiropoulos@yahoo.gr
    __________________________________________________________________________"""

    def __init__(self,dt,xgtt,
        sw='ES',baselineSw=True,dti=0.01,ksi=0.05,T=None,mu=2,AlgID='U0-V0-Opt'):
        if T is None:
            T = np.logspace(np.log10(0.02),np.log10(50),1000).T

        if ~ischar(AlgID):
            if not all(np.size(AlgID)==[1,14]):
                error('AlgID must be a 1x14 vector or string')
            
        ## Calculation
        xgtt = xgtt(:)
        nxgtt=len(xgtt)
        time =np.arange(nxgtt-1).T*dt
        sw=lower(sw)
        switch sw
            # TIME SERIES
            if sw ==  'timehist':
                if baselineSw:
                    [cor_xg,cor_xgt,cor_xgtt] = baselineCorr(time,xgtt)
                    param.time=time
                    param.acc=cor_xgtt
                    param.vel=cor_xgt
                    param.disp=cor_xg
                else:
                    param.time=time
                    # Acceleration time history
                    param.acc = xgtt
                    # Velocity time history
                    param.vel = np.cumtrapz(time,xgtt)
                    # Displacement time history
                    param.disp = np.cumtrapz(time,param.vel)
                
            if sw ==  'resample':
                # RESAMPLE THE ACCELERATION TIME HISTORY TO INCREASE OR DECREASE
                # THE SIZE OF THE TIME STEP
                d1,d2 = rat(dt/dti)
                # Resample the acceleration time history
                xgtt,_ = resample(xgtt,d1,d2)
                NANxgtt=find(isnan(xgtt))
                errxgtt=find(diff(NANxgtt)>1)
                if any(errxgtt):
                    error('Non consecutive NaNs in resampled acceleration time history')
                
                if any(NANxgtt):
                    xgtt = xgtt(1:NANxgtt(1)-1)
                
                param.acc = xgtt
                # Time scale
                param.time=(0:numel(param.acc)-1)'*dti;
                
                # PEAK RESPONSES
            if sw ==  'pga':
                # Peak ground acceleration
                param.PGA = max(abs(xgtt))
            if sw ==  'pgv':
                # Peak ground velocity
                param.PGV = max(abs(cumtrapz(time,xgtt)*dt))
            if sw ==  'pgd':
                # Peak ground displacement
                param.PGD = max(abs(cumtrapz(time,cumtrapz(time,xgtt)*dt)*dt))
                
            if sw ==  'arias':
                param.time=time
                # CUMULATIVE ENERGY
                # time history of cumulative energy
                EcumTH = cumsum(xgtt**2)*dt;
                # Total cumulative energy at the end of the ground motion
                Ecum = EcumTH(-1);
                param.Ecum = Ecum;
                # time history of the normalized cumulative energy
                param.EcumTH = EcumTH/Ecum;
                # SIGNIFICANT DURATION
                # elements of the time vector which are within the significant
                # duration
                timed = time(EcumTH>=0.05*Ecum & EcumTH<=0.95*Ecum);
                # starting and ending points of the significant duration
                param.t_5_95 = [timed(1),timed(end)];
                # significant duration
                param.Td = timed(end)-timed(1)+dt;
                
                # ARIAS INTENSITY
                # time history of Arias Intensity
                ariasTH = 1/9.81*cumsum(xgtt(time<=param.Td)**2)*np.pi*dt/2;
                # Total Arias Intensity at the end of the ground motion
                arias = ariasTH(end);
                param.arias = arias;
                
            if sw ==  'es':
                """LINEAR ELASTIC RESPONSE SPECTRA"""
                T=T(:)
                param.Period = T(:)
                dtTol=0.02
                rinf=1
                PSa,PSv,Sd,Sv,Sa,SievABS,SievREL = LEReSp(dt,xgtt,T,ksi,dtTol,AlgID,rinf)
                param.PSa=PSa(:);
                param.PSv=PSv(:);
                param.Sd=Sd(:);
                param.Sv=Sv(:);
                param.Sa=Sa(:);
                param.SievABS=SievABS(:);
                param.SievREL=SievREL(:);
                a1,a2=max(PSa(:));
                param.PredPSa=a1;
                param.PredPeriod=T(a2);
                
            if sw ==  'cds':
                """CONSTANT DUCTILITY RESPONSE SPECTRA"""
                T=T(:);
                param.Period = T(:);
                # #n# (scalar) is the maximum number of iterations.
                n = 65;
                tol = 0.02;
                dtTol = 0.02;
                rinf=1;
                CDPSa,CDPSv,CDSd,CDSv,CDSa,fyK,muK,iterK = CDReSp(dt,xgtt,T,ksi,mu,n,tol,dtTol,AlgID,rinf)
                param.CDPSa=CDPSa(:);
                param.CDPSv=CDPSv(:);
                param.CDSd=CDSd(:);
                param.CDSv=CDSv(:);
                param.CDSa=CDSa(:);
                param.fyK=fyK(:);
                param.muK=muK(:);
                param.iterK=iterK(:);
                
            if sw ==  'fas':
                """FOURIER AMPLITUDE SPECTRUM"""
                [f,U]=FASp(dt,xgtt)
                param.freq = f
                param.FAS = U
                
                # MEAN PERIOD AND FREQUENCY
                fi = f(f>0.25 & f<20)
                Ci = U(f>0.25 & f<20)
                Tm = ((Ci(:).T**2)*(1/fi(:)))/(Ci(:).T*Ci(:))
                param.Tm = Tm
                Fm = ((Ci(:).T**2)*(fi(:)))/(Ci(:).T*Ci(:))
                param.Fm = Fm

def LEReSp(dt,xgtt,T,ksi=0.05,dtTol=0.01,AlgID='U0-V0-CA',rinf=0):
    """Fast calculation of Linear Elastic Response Spectra (LEReSp) and
     pseudospectra.
    
     [#PSa#,#PSv#,#Sd#,#Sv#,#Sa#,#SievABS#,#SievREL#]=LEReSp(#dt#,#xgtt#,#T#,#ksi#)
    
     Description
         The linear elastic response spectra for a given time-history of
         constant time step, a given eigenperiod range and a given viscous
         damping ratio are computed.
    
     Input parameters
         #dt# (scalar) is the time step of the input acceleration time history
             #xgtt#.
         #xgtt# ([#numsteps# x 1]) is the input acceleration time history.
             #numsteps# is the length of the input acceleration time history.
         #T# ([#numSDOFs# x 1]) contains the values of eigenperiods for which
             the response spectra are requested. #numSDOFs# is the number of
             SDOF oscillators being analysed to produce the spectra.
         #ksi# (scalar) is the fraction of critical viscous damping.
         #dtTol# (scalar) is the maximum ratio of the integration time step to
             the eigenperiod. Default value 0.01.
         #AlgID# (string as follows) is the algorithm to be used for the time
             integration. It can be one of the following strings for superior
             optimally designed algorithms:


         #rinf# (scalar) is the minimum absolute value of the eigenvalues of
             the amplification matrix. For the amplification matrix see
             eq.(61) in Zhou & Tamma (2004).
    
     Output parameters
         #PSa# ([#numSDOFs# x 1]) is the Pseudo Acceleration Spectrum.
         #PSv# ([#numSDOFs# x 1]) is the Pseudo Velocity Spectrum.
         #Sd# ([#numSDOFs# x 1]) is the Spectral Displacement.
         #Sv# ([#numSDOFs# x 1]) is the Spectral Velocity.
         #Sa# ([#numSDOFs# x 1]) is the Spectral Acceleration.
         #SievABS# ([#numSDOFs# x 1]) is the equivalent absolute input energy
             velocity.
         #SievREL# ([#numSDOFs# x 1]) is the equivalent relative input energy
             velocity.
    
     Example
         dt=0.02;
         xgtt=rand(1000,1);
         Tspectra=logspace(log10(0.02),log10(50),1000)';
         ksi=0.05;
         [PSa,PSv,Sd,Sv,Sa,SievABS,SievREL]=LEReSp(dt,xgtt,T,ksi);
    
    __________________________________________________________________________
     Copyright (c) 2018
         George Papazafeiropoulos
         Captain, Infrastructure Engineer, Hellenic Air Force
         Civil Engineer, M.Sc., Ph.D. candidate, NTUA
         Email: gpapazafeiropoulos@yahoo.gr
     _________________________________________________________________________"""

    # required inputs
     # if ~isscalar(dt)
     #     error('dt is not scalar')
     # end
     # if dt<=0
     #     error('dt is zero or negative')
     # end
     # if ~isvector(xgtt)
     #     error('xgtt is not vector')
     # end
     # if ~isvector(T)
     #     error('T is not vector')
     # end
     # if any(T<=0)
     #     error('T must be positive')
     # end
     # # optional inputs
     # if ~isscalar(ksi)
     #     error('ksi is not scalar')
     # end
     # if ksi<0
     #     error('ksi is negative')
     # end

    ## Calculation
    # Set integration constants
    if all(size(AlgID)==[1,14])
        # define integration constants explicitly
        w1=AlgID(1);
        w2=AlgID(2);
        w3=AlgID(3);
        W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
        # W2=(1/3+w1/4+w2/5+w3/6)/(1+w1/2+w2/3+w3/4); # definition
        # W3=(1/4+w1/5+w2/6+w3/7)/(1+w1/2+w2/3+w3/4); # definition
        W1L1=AlgID(4);
        W2L2=AlgID(5);
        W3L3=AlgID(6);
        W1L4=AlgID(7);
        W2L5=AlgID(8);
        W1L6=AlgID(9);
        l1=AlgID(10);
        l2=AlgID(11);
        l3=AlgID(12);
        l4=AlgID(13);
        l5=AlgID(14);

    # initialize
    NumSDOF = len(T)
    Sd=np.zeros((NumSDOF,1))
    Sv=np.zeros((NumSDOF,1))
    Sa=np.zeros((NumSDOF,1))
    SievABS=np.zeros((NumSDOF,1))
    SievREL=np.zeros((NumSDOF,1))
    # Set the eigenfrequencies of the SDOF population
    omega=2*np.pi/T;
    # Flip eigenfrequency vector in order for the half-stepping algorithm
    # (HalfStep function) to work from large to small eigenperiods
    omega=omega(end:-1:1);
    # set initial conditions
    u0=0
    ut0=0
    for j=1:len(T)
        omegaj=omega[j];
        # Check if dt/T>dtTol. If yes, then reproduce the time history with the
        # half step
        if dt*omegaj/(2*np.pi)>dtTol:
            xgtt = HalfStep(xgtt)
            dt = dt/2

        u,ut,utt = LIDA(dt,xgtt,omegaj,ksi,u0,ut0,AlgID,rinf)
        # output
        Sd[j]=max(abs(u))
        Sv[j]=max(abs(ut))
        Sa[j]=max(abs(utt))
        SievABS[j]= np.sqrt(2*dt**2*sum((utt+xgtt)*cumsum(xgtt)))
        SievREL[j]= np.sqrt(-2*dt**2*sum((xgtt)*cumsum(utt)))
    
    # Flip output quantities to be compatible with omega
    omega = omega(end:-1:1);
    Sd = Sd[end:-1:1]
    Sv = Sv[end:-1:1]
    Sa = Sa[end:-1:1]
    # Calculate pseudovelocity and pseudoacceleration
    PSv=Sd*omega;
    PSa=Sd*omega.^2;

def CDReSp(dt,xgtt,T,varargin):
    """Constant Ductility Response Spectra (CDReSp)

    [PSa,PSv,Sd,Sv,Sa,fyK,muK,iterK]=CDReSp(dt,xgtt,T,ksi,mu,n,tol)

    Description
        The constant ductility response spectra for a given time-history of
        constant time step, a given eigenperiod range, a given viscous
        damping ratio and a given ductility are computed. See section 7.5 in
        Chopra (2012).

    Input parameters
    -------------------------------------------
        #dt# (scalar) is the time step of the input acceleration time history
            #xgtt#.
        #xgtt# ([#numsteps# x 1]) is the input acceleration time history.
        #numsteps# is the length of the input acceleration time history.
        #T# ([#numSDOFs# x 1]) contains the values of eigenperiods for which
            the response spectra are requested. #numSDOFs# is the number of
            SDOF oscillators being analysed to produce the spectra.
        #ksi# (scalar) is the fraction of critical viscous damping.
        #mu# (scalar) is the target ductility for which the response spectra
            are calculated.
        #n# (scalar) is the maximum number of iterations that can be
            performed until convergence of the calculated ductility to the
            target ductility is achieved. Default value 65.
        #tol1# (scalar) is the tolerance for convergence for the target
            ductility. Default value 0.01.

    Output parameters
    --------------------------------------------
        #PSa# ([#numSDOFs# x 1]) is the Pseudo-Spectral Acceleration.
        #PSv# ([#numSDOFs# x 1]) is the Pseudo-Spectral Velocity.
        #Sd# ([#numSDOFs# x 1]) is the Spectral Displacement.
        #Sv# ([#numSDOFs# x 1]) is the Spectral Velocity.
        #Sa# ([#numSDOFs# x 1]) is the Spectral Acceleration.
        #fyK# ([#numSDOFs# x 1]) is the yield limit that each SDOF must have
            in order to attain ductility equal to #muK#.
        #muK# ([#numSDOFs# x 1]) is the achieved ductility for each period
            (each SDOF).
        #iterK# ([#numSDOFs# x 1]) is the number of iterations needed for
            convergence for each period (each SDOF).

    Example
        fid=fopen('elcentro.dat','r');
        text=textscan(fid,'#f #f');
        fclose(fid);
        dt=0.02;
        xgtt=text{1,2};
        Tspectra=logspace(log10(0.1),log10(3),10)';
        ksi=0.05;
        mu=2;
        n=65;
        tol=0.01;
        [PSa,PSv,Sd,Sv,Sa,fyK,muK,iterK]=CDReSp(dt,xgtt,T,ksi,mu,n,tol);
        plot(Tspectra,Sa)

    __________________________________________________________________________
    Copyright (c) 2018
        George Papazafeiropoulos
        Captain, Infrastructure Engineer, Hellenic Air Force
        Civil Engineer, M.Sc., Ph.D. candidate, NTUA
        Email: gpapazafeiropoulos@yahoo.gr
    _________________________________________________________________________"""

    ## Initial checks
    if nargin<3
        error('Input arguments less than required')
    end
    if nargin>10
        error('Input arguments more than required')
    end
    # set defaults for optional inputs
    optargs = {0.05,2,60,0.01,0.01,'U0-V0-CA',0};
    # skip any new inputs if they are empty
    newVals = cellfun(@(x) ~isempty(x), varargin);
    # overwrite the default values by those specified in varargin
    optargs(newVals) = varargin(newVals);
    # place optional args in memorable variable names
    [ksi,mu,n,tol,dtTol,AlgID,rinf] = optargs{:};
    # required inputs
    if ~isscalar(dt)
        error('dt is not scalar')
    end
    if dt<=0
        error('dt is zero or negative')
    end
    if ~isvector(xgtt)
        error('xgtt is not vector')
    end

    # Calculation
    # Set integration constants
    if all(size(AlgID)==[1,14]):
        # define integration constants explicitly
        w1=AlgID(1);
        w2=AlgID(2);
        w3=AlgID(3);
        W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
        # W2=(1/3+w1/4+w2/5+w3/6)/(1+w1/2+w2/3+w3/4); # definition
        # W3=(1/4+w1/5+w2/6+w3/7)/(1+w1/2+w2/3+w3/4); # definition
        W1L1=AlgID(4);
        W2L2=AlgID(5);
        W3L3=AlgID(6);
        W1L4=AlgID(7);
        W2L5=AlgID(8);
        W1L6=AlgID(9);
        l1=AlgID(10);
        l2=AlgID(11);
        l3=AlgID(12);
        l4=AlgID(13);
        l5=AlgID(14);

    # initialize
    NumSDOF=len(T);
    Sd= np.zeros(NumSDOF,1)
    Sv= np.zeros(NumSDOF,1)
    Sa= np.zeros(NumSDOF,1)
    PSv=np.zeros(NumSDOF,1)
    PSa=np.zeros(NumSDOF,1)
    fyK=np.zeros(NumSDOF,1)
    muK=np.zeros(NumSDOF,1)
    iterK=np.zeros(NumSDOF,1)
    # natural frequencies of SDOFs to be analysed
    omega=2*np.pi/T
    # Flip eigenfrequency vector in order for the half-stepping algorithm
    # (HalfStep function) to work from large to small eigenperiods
    omega=omega(end:-1:1);
    # set initial conditions
    u0=0
    ut0=0
    # set SDOF mass to unity (any value can be specified without change of the
    # results)
    m=1
    # Maximum tolerance for convergence
    tol2=0.01
    # Maximum number of iterations per increment
    jmax=200
    # Infinitesimal variation of acceleration
    dak=eps;
    for j in range(len(T)):
        # Step 3, 7.5.3
        omegaj=omega[j];
        # Check if dt/T>dtTol. If yes, then reproduce the time history with the
        # half step
        if dt*omegaj/(2*np.pi)>dtTol:
            xgtt = HalfStep(xgtt)
            dt = dt/2

        k_hi = m*omegaj**2
        k_lo = k_hi/100
        # Step 4, 7.5.3
        u,ut,utt,p,Es,Ed = DRHA(k_hi,m,dt,xgtt,ksi,u0,ut0,AlgID,rinf)
        upeak=max(abs(u))
        fpeak=k_hi*upeak
        # Step 5, 7.5.3
        # find uy in order to achieve ductility equal to mu
        uy=np.zeros(n+1,1);
        resid=np.zeros(n+1,1);
        pos = max(abs(u));
        neg = max(abs(u))/(mu*1.5);
        uy(1) = pos;
        tol1=tol;
        for in range(n): # k = iteration number
            [um,umt,umtt,p,Ey,Es,Ed,iter] = NLIDABLKIN(dt,xgtt,m,k_hi,k_lo,uy[k],ksi,AlgID,u0,ut0,rinf,tol2,jmax,dak)
            umax=max(abs(um))
            # ductility factor from eq. 7.2.4. in Chopra
            fy=k_hi*uy(k)
            fybark=fy/fpeak
            muk=(umax/upeak)/fybark
            # CONVERGENCE TEST : NEWTON AVERAGE ALGORITHM
            # residual (difference between real and target ductility)
            resid(k) = mu - muk;
            if (abs(resid(k))/mu <= tol1):
                fyK(j)=fy;
                muK(j)=muk;
                iterK(j)=k;
                break
            elif (k>=2):
                # adjust tol1 according to the number of iterations performed
                if (k>round(n/2) and k<=round(3*n/4)):
                    tol1=0.1
                elif (k>round(3*n/4) && k<=round(7*n/8)):
                    tol1=0.2
                elif (k>round(7*n/8)):
                    tol1=0.8

                # assign uy of previous iteration to pos or neg
                if resid(k-1)<0:
                    neg=uy(k-1)
                else:
                    pos=uy(k-1)

                # calculate the next value of uy
                if (resid(k)<0 && resid(k-1)>0):
                    neg=uy(k);
                    uynew=(pos+neg)/2;
                    uy(k+1)=uynew;
                elif (resid(k)>0 && resid(k-1)<0):
                    pos=uy(k);
                    uynew=(pos+neg)/2;
                    uy(k+1)=uynew;
                elif resid(k)<0 && resid(k-1)<0 && resid(k)>=resid(k-1):
                    neg=uy(k);
                    uynew=(pos+neg)/2;
                    uy(k+1)=uynew;
                elif resid(k)<0 && resid(k-1)<0 && resid(k)<=resid(k-1):
                    uynew=(pos+neg)/2;
                    uy(k+1)=uynew;
                elif resid(k)>0 && resid(k-1)>0 && resid(k)>=resid(k-1):
                    uynew=(pos+neg)/2;
                    uy(k+1)=uynew;
                elif resid(k)>0 && resid(k-1)>0 && resid(k)<=resid(k-1)
                    pos=uy(k);
                    uynew=(pos+neg)/2;
                    uy(k+1)=uynew;
            elif (k==1):
                uynew = neg;
                uy(k+1)=uynew;
        
        # find Sd, Sv, Sa
        Sd[j]=umax;
        Sv[j]=max(abs(umt));
        Sa[j]=max(abs(umtt));
        # find PSv, PSa
        PSv(j)=Sd(j)*omegaj;
        PSa(j)=Sd(j)*omegaj**2;

    # Flip output quantities to be compatible with omega
    Sd=  Sd[:-1:1]
    Sv=  Sv[end:-1:1]
    Sa=  Sa[end:-1:1]
    PSv=PSv[end:-1:1]
    PSa=PSa[end:-1:1]
    fyK=fyK[end:-1:1]
    muK=muK[end:-1:1]
    iterK=iterK(end:-1:1);

def LIDA(dt,xgtt,omega,ksi=0.05,u0=0.0,ut0=0.0,AlgID='U0-V0-Opt',rinf=1):
    """Linear Implicit Dynamic Analysis (LIDA)
    
     [u,ut,utt] = LIDA(dt,xgtt,omega,ksi,u0,ut0,rinf)
         Linear implicit direct time integration of second order differential
         equation of motion of dynamic response of linear elastic SDOF systems
    
     Description
     -------------------------
         The General Single Step Single Solve (GSSSS) family of algorithms
         published by X.Zhou & K.K.Tamma (2004) is employed for direct time
         integration of the general linear or nonlinear structural Single
         Degree of Freedom (SDOF) dynamic problem. The optimal numerical
         dissipation and dispersion zero order displacement zero order
         velocity algorithm designed according to the above journal article,
         is used in this routine. This algorithm encompasses the scope of
         Linear Multi-Step (LMS) methods and is limited by the Dahlquist
         barrier theorem (Dahlquist,1963). The force - displacement - velocity
         relation of the SDOF structure is linear.
    
     Input parameters
     -------------------------------
         #dt# (scalar): time step
         #xgtt# ([#nstep# x 1]): column vector of the acceleration history of
             the excitation imposed at the base. #nstep# is the number of time
             steps of the dynamic response.
         #omega# (scalar): eigenfrequency of the structure in rad/sec
         #ksi# (scalar): ratio of critical damping of the SDOF system. Default
             value 0.05.
         #u0# (scalar): initial displacement of the SDOF system. Default value
             0.
         #ut0# (scalar): initial velocity of the SDOF system. Default value 0.
         #AlgID# (string as follows) is the algorithm to be used for the time
             integration. It can be one of the following strings for superior
             optimally designed algorithms:
                 'generalized a-method': The generalized a-method (Chung &
                 Hulbert, 1993)
                 'HHT a-method': The Hilber-Hughes-Taylor method (Hilber,
                 Hughes & Taylor, 1977)
                 'WBZ': The Wood烹ossak忙ienkiewicz method (Wood, Bossak &
                 Zienkiewicz, 1980)
                 'U0-V0-Opt': Optimal numerical dissipation and dispersion
                 zero order displacement zero order velocity algorithm
                 'U0-V0-CA': Continuous acceleration (zero spurious root at
                 the low frequency limit) zero order displacement zero order
                 velocity algorithm
                 'U0-V0-DA': Discontinuous acceleration (zero spurious root at
                 the high frequency limit) zero order displacement zero order
                 velocity algorithm
                 'U0-V1-Opt': Optimal numerical dissipation and dispersion
                 zero order displacement first order velocity algorithm
                 'U0-V1-CA': Continuous acceleration (zero spurious root at
                 the low frequency limit) zero order displacement first order
                 velocity algorithm
                 'U0-V1-DA': Discontinuous acceleration (zero spurious root at
                 the high frequency limit) zero order displacement first order
                 velocity algorithm
                 'U1-V0-Opt': Optimal numerical dissipation and dispersion
                 first order displacement zero order velocity algorithm
                 'U1-V0-CA': Continuous acceleration (zero spurious root at
                 the low frequency limit) first order displacement zero order
                 velocity algorithm
                 'U1-V0-DA': Discontinuous acceleration (zero spurious root at
                 the high frequency limit) first order displacement zero order
                 velocity algorithm
                 'Newmark ACA': Newmark Average Constant Acceleration method
                 'Newmark LA': Newmark Linear Acceleration method
                 'Newmark BA': Newmark Backward Acceleration method
                 'Fox-Goodwin': Fox-Goodwin formula
         #rinf# (scalar): minimum absolute value of the eigenvalues of the
             amplification matrix. For the amplification matrix see eq.(61) in
             Zhou & Tamma (2004). Default value 1.
    
     Output parameters
     ------------------------------------------
         #u# ([#nstep# x 1]): time-history of displacement
         #ut# ([#nstep# x 1]): time-history of velocity
         #utt# ([#nstep# x 1]): time-history of acceleration
    
     Example (Figure 6.6.1 in Chopra, Tn=1sec)
     --------------------------------------------
         dt=0.02;
         fid=fopen('elcentro.dat','r');
         text=textscan(fid,'#f #f');
         fclose(fid);
         xgtt=9.81*text{1,2};
         Tn=1;
         omega=2*np.pi/Tn;
         ksi=0.02;
         u0=0;
         ut0=0;
         rinf=1;
         [u,ut,utt] = LIDA(dt,xgtt,omega,ksi,u0,ut0,rinf);
         D=max(abs(u))/0.0254
    
    __________________________________________________________________________
     Copyright (c) 2018
         George Papazafeiropoulos
         Captain, Infrastructure Engineer, Hellenic Air Force
         Civil Engineer, M.Sc., Ph.D. candidate, NTUA
         Email: gpapazafeiropoulos@yahoo.gr
     _________________________________________________________________________"""

     # required inputs
        # if  not isscalar(dt):
        #     error('dt is not scalar')
        # if dt<=0:
        #     error('dt is zero or negative')
        # if ~isvector(xgtt):
        #     error('xgtt is not vector')
        # if ~isscalar(omega):
        #     error('omega is not scalar')
        # if omega<=0:
        #     error('omega is zero or negative')
        # # optional inputs
        # if ~isscalar(ksi)
        #     error('ksi is not scalar')
        # if ksi<0:
        #     error('ksi is negative')
        # if ~isscalar(u0):
        #     error('u0 is not scalar')
        # if ~isscalar(ut0):
        #     error('ut0 is not scalar')
        # if ~isscalar(rinf):
        #     error('rinf is not scalar')
        # if rinf<0 || rinf>1:
        #     error('rinf is lower than 0 or higher than 1')
    
    ## Calculation
    # Set integration constants
    if all(size(AlgID)==[1,14]):
        # define integration constants explicitly
        w1=AlgID[1]
        w2=AlgID[2]
        w3=AlgID[3]
        W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
        # W2=(1/3+w1/4+w2/5+w3/6)/(1+w1/2+w2/3+w3/4); # definition
        # W3=(1/4+w1/5+w2/6+w3/7)/(1+w1/2+w2/3+w3/4); # definition
        W1L1=AlgID[4]
        W2L2=AlgID[5]
        W3L3=AlgID[6]
        W1L4=AlgID[7]
        W2L5=AlgID[8]
        W1L6=AlgID[9]
        l1= AlgID[10]
        l2= AlgID[11]
        l3= AlgID[12]
        l4= AlgID[13]
        l5= AlgID[14]
    else:
        switch AlgID
            case 'U0-V0-Opt'
                # zero-order displacement & velocity overshooting behavior and
                # optimal numerical dissipation and dispersion
                # rinf must belong to [0 1]
                if rinf<0
                    rinf=0;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
                end
                if rinf>1
                    rinf=1; # mid-point rule a-form algorithm
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
                end
                w1=-15*(1-2*rinf)/(1-4*rinf); # suggested
                w2=15*(3-4*rinf)/(1-4*rinf); # suggested
                w3=-35*(1-rinf)/(1-4*rinf); # suggested
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=1/(1+rinf);
                W2L2=1/2/(1+rinf);
                W3L3=1/2/(1+rinf)^2;
                W1L4=1/(1+rinf);
                W2L5=1/(1+rinf)^2; # suggested
                W1L6=(3-rinf)/2/(1+rinf);
                l1=1;
                l2=1/2;
                l3=1/2/(1+rinf);
                l4=1;
                l5=1/(1+rinf);
            case 'U0-V0-CA'
                # zero-order displacement & velocity overshooting behavior and
                # continuous acceleration
                # rinf must belong to [1/3 1]
                if rinf<1/3
                    rinf=1/3;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1/3');
                end
                if rinf>1
                    rinf=1; # Newmark average acceleration a-form algorithm
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
                end
                w1=-15*(1-5*rinf)/(3-7*rinf); # suggested
                w2=15*(1-13*rinf)/(3-7*rinf); # suggested
                w3=140*rinf/(3-7*rinf); # suggested
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=(1+3*rinf)/2/(1+rinf);
                W2L2=(1+3*rinf)/4/(1+rinf);
                W3L3=(1+3*rinf)/4/(1+rinf)^2;
                W1L4=(1+3*rinf)/2/(1+rinf);
                W2L5=(1+3*rinf)/2/(1+rinf)^2; # suggested
                W1L6=1;
                l1=1;
                l2=1/2;
                l3=1/2/(1+rinf);
                l4=1;
                l5=1/(1+rinf);
            case 'U0-V0-DA'
                # zero-order displacement & velocity overshooting behavior and
                # discontinuous acceleration
                # rinf must belong to [0 1]
                if rinf<0
                    rinf=0;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
                end
                if rinf>1
                    rinf=1; # Newmark average acceleration a-form algorithm
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
                end
                w1=-15; # suggested
                w2=45; # suggested
                w3=-35; # suggested
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=1;
                W2L2=1/2;
                W3L3=1/2/(1+rinf);
                W1L4=1;
                W2L5=1/(1+rinf); # suggested
                W1L6=(3+rinf)/2/(1+rinf);
                l1=1;
                l2=1/2;
                l3=1/2/(1+rinf);
                l4=1;
                l5=1/(1+rinf);
            case 'U0-V1-Opt'
                # zero-order displacement & first-order velocity overshooting
                # behavior and optimal numerical dissipation and dispersion
                # This is the generalized a-method (Chung & Hulbert, 1993)
                # rinf must belong to [0 1]
                if rinf<0
                    rinf=0;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
                end
                if rinf>1
                    rinf=1;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
                end
                w1=-15*(1-2*rinf)/(1-4*rinf);
                w2=15*(3-4*rinf)/(1-4*rinf);
                w3=-35*(1-rinf)/(1-4*rinf);
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=1/(1+rinf);
                W2L2=1/2/(1+rinf);
                W3L3=1/(1+rinf)^3;
                W1L4=1/(1+rinf);
                W2L5=(3-rinf)/2/(1+rinf)^2;
                W1L6=(2-rinf)/(1+rinf);
                l1=1;
                l2=1/2;
                l3=1/(1+rinf)^2;
                l4=1;
                l5=(3-rinf)/2/(1+rinf);
            case 'generalized a-method'
                # zero-order displacement & first-order velocity overshooting
                # behavior and optimal numerical dissipation and dispersion
                # This is the generalized a-method (Chung & Hulbert, 1993)
                # rinf must belong to [0 1]
                if rinf<0
                    rinf=0;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
                end
                if rinf>1
                    rinf=1;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
                end
                w1=-15*(1-2*rinf)/(1-4*rinf);
                w2=15*(3-4*rinf)/(1-4*rinf);
                w3=-35*(1-rinf)/(1-4*rinf);
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=1/(1+rinf);
                W2L2=1/2/(1+rinf);
                W3L3=1/(1+rinf)^3;
                W1L4=1/(1+rinf);
                W2L5=(3-rinf)/2/(1+rinf)^2;
                W1L6=(2-rinf)/(1+rinf);
                l1=1;
                l2=1/2;
                l3=1/(1+rinf)^2;
                l4=1;
                l5=(3-rinf)/2/(1+rinf);
            case 'U0-V1-CA'
                # zero-order displacement & first-order velocity overshooting
                # behavior and continuous acceleration
                # This is the Hilber-Hughes-Taylor method (Hilber, Hughes &
                # Taylor, 1977)
                # rinf must belong to [1/2 1]
                if rinf<1/2
                    rinf=1/2;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1/2');
                end
                if rinf>1
                    rinf=1;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
                end
                w1=-15*(1-2*rinf)/(2-3*rinf);
                w2=15*(2-5*rinf)/(2-3*rinf);
                w3=-35*(1-3*rinf)/2/(2-3*rinf);
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=2*rinf/(1+rinf);
                W2L2=rinf/(1+rinf);
                W3L3=2*rinf/(1+rinf)^3;
                W1L4=2*rinf/(1+rinf);
                W2L5=rinf*(3-rinf)/(1+rinf)^2;
                W1L6=1;
                l1=1;
                l2=1/2;
                l3=1/(1+rinf)^2;
                l4=1;
                l5=(3-rinf)/2/(1+rinf);
            case 'HHT a-method'
                # zero-order displacement & first-order velocity overshooting
                # behavior and continuous acceleration
                # This is the Hilber-Hughes-Taylor method (Hilber, Hughes &
                # Taylor, 1977)
                # rinf must belong to [1/2 1]
                if rinf<1/2
                    rinf=1/2;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1/2');
                end
                if rinf>1
                    rinf=1;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
                end
                w1=-15*(1-2*rinf)/(2-3*rinf);
                w2=15*(2-5*rinf)/(2-3*rinf);
                w3=-35*(1-3*rinf)/2/(2-3*rinf);
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=2*rinf/(1+rinf);
                W2L2=rinf/(1+rinf);
                W3L3=2*rinf/(1+rinf)^3;
                W1L4=2*rinf/(1+rinf);
                W2L5=rinf*(3-rinf)/(1+rinf)^2;
                W1L6=1;
                l1=1;
                l2=1/2;
                l3=1/(1+rinf)^2;
                l4=1;
                l5=(3-rinf)/2/(1+rinf);
            case 'U0-V1-DA'
                # zero-order displacement & first-order velocity overshooting
                # behavior and discontinuous acceleration
                # This is the Wood烹ossak忙ienkiewicz method (Wood, Bossak &
                # Zienkiewicz, 1980)
                # rinf must belong to [0 1]
                if rinf<0
                    rinf=0;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
                end
                if rinf>1
                    rinf=1;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
                end
                w1=-15;
                w2=45;
                w3=-35;
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=1;
                W2L2=1/2;
                W3L3=1/(1+rinf)^2;
                W1L4=1;
                W2L5=(3-rinf)/2/(1+rinf);
                W1L6=2/(1+rinf);
                l1=1;
                l2=1/2;
                l3=1/(1+rinf)^2;
                l4=1;
                l5=(3-rinf)/2/(1+rinf);
            case 'WBZ'
                # zero-order displacement & first-order velocity overshooting
                # behavior and discontinuous acceleration
                # This is the Wood烹ossak忙ienkiewicz method (Wood, Bossak &
                # Zienkiewicz, 1980)
                # rinf must belong to [0 1]
                if rinf<0
                    rinf=0;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
                end
                if rinf>1
                    rinf=1;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
                end
                w1=-15;
                w2=45;
                w3=-35;
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=1;
                W2L2=1/2;
                W3L3=1/(1+rinf)^2;
                W1L4=1;
                W2L5=(3-rinf)/2/(1+rinf);
                W1L6=2/(1+rinf);
                l1=1;
                l2=1/2;
                l3=1/(1+rinf)^2;
                l4=1;
                l5=(3-rinf)/2/(1+rinf);
            case 'U1-V0-Opt'
                # first-order displacement & zero-order velocity overshooting
                # behavior and optimal numerical dissipation and dispersion
                # rinf must belong to [0 1]
                if rinf<0
                    rinf=0;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
                end
                if rinf>1
                    rinf=1; # mid-point rule a-form algorithm
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
                end
                w1=-30*(3-8*rinf+6*rinf^2)/(9-22*rinf+19*rinf^2);
                w2=15*(25-74*rinf+53*rinf^2)/2/(9-22*rinf+19*rinf^2);
                w3=-35*(3-10*rinf+7*rinf^2)/(9-22*rinf+19*rinf^2);
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=(3-rinf)/2/(1+rinf);
                W2L2=1/(1+rinf)^2;
                W3L3=1/(1+rinf)^3;
                W1L4=(3-rinf)/2/(1+rinf);
                W2L5=2/(1+rinf)^3;
                W1L6=(2-rinf)/(1+rinf);
                l1=1;
                l2=1/2;
                l3=1/2/(1+rinf);
                l4=1;
                l5=1/(1+rinf);
            case 'U1-V0-CA'
                # first-order displacement & zero-order velocity overshooting
                # behavior and continuous acceleration
                # rinf must belong to [1/2 1]
                if rinf<1/2
                    rinf=1/2;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1/2');
                end
                if rinf>1
                    rinf=1; # Newmark average acceleration a-form algorithm
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
                end
                w1=-60*(2-8*rinf+7*rinf^2)/(11-48*rinf+41*rinf^2);
                w2=15*(37-140*rinf+127*rinf^2)/2/(11-48*rinf+41*rinf^2);
                w3=-35*(5-18*rinf+17*rinf^2)/(11-48*rinf+41*rinf^2);
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=(1+3*rinf)/2/(1+rinf);
                W2L2=2*rinf/(1+rinf)^2;
                W3L3=2*rinf/(1+rinf)^3;
                W1L4=(1+3*rinf)/2/(1+rinf);
                W2L5=4*rinf/(1+rinf)^3;
                W1L6=1;
                l1=1;
                l2=1/2;
                l3=1/2/(1+rinf);
                l4=1;
                l5=1/(1+rinf);
            case 'U1-V0-DA'
                # first-order displacement & zero-order velocity overshooting behavior
                # and discontinuous acceleration
                # This is the Newmark average acceleration a-form algorithm
                # rinf must belong to [0 1]
                if rinf<0
                    rinf=0;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
                end
                if rinf>1
                    rinf=1;
                    warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
                end
                w1=-30*(3-4*rinf)/(9-11*rinf);
                w2=15*(25-37*rinf)/2/(9-11*rinf);
                w3=-35*(3-5*rinf)/(9-11*rinf);
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=(3+rinf)/2/(1+rinf);
                W2L2=1/(1+rinf);
                W3L3=1/(1+rinf)^2;
                W1L4=(3+rinf)/2/(1+rinf);
                W2L5=2/(1+rinf)^2;
                W1L6=2/(1+rinf);
                l1=1;
                l2=1/2;
                l3=1/(1+rinf)^2;
                l4=1;
                l5=(3-rinf)/2/(1+rinf);
            case 'Newmark ACA'
                # Newmark Average Constant Acceleration method
                w1=-15;
                w2=45;
                w3=-35;
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                # W2=(1/3+w1/4+w2/5+w3/6)/(1+w1/2+w2/3+w3/4); # definition
                # W3=(1/4+w1/5+w2/6+w3/7)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=1;
                W2L2=0.25;
                W3L3=0.25;
                W1L4=0.5;
                W2L5=0.5;
                W1L6=1;
                l1=1;
                l2=0.5;
                l3=0.25;
                l4=1;
                l5=0.5;
            case 'Newmark LA'
                # Newmark Linear Acceleration method
                w1=-15;
                w2=45;
                w3=-35;
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                # W2=(1/3+w1/4+w2/5+w3/6)/(1+w1/2+w2/3+w3/4); # definition
                # W3=(1/4+w1/5+w2/6+w3/7)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=1;
                W2L2=1/6;
                W3L3=1/6;
                W1L4=0.5;
                W2L5=0.5;
                W1L6=1;
                l1=1;
                l2=0.5;
                l3=1/6;
                l4=1;
                l5=0.5;
            case 'Newmark BA'
                # Newmark Backward Acceleration method
                w1=-15;
                w2=45;
                w3=-35;
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                # W2=(1/3+w1/4+w2/5+w3/6)/(1+w1/2+w2/3+w3/4); # definition
                # W3=(1/4+w1/5+w2/6+w3/7)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=1;
                W2L2=0.5;
                W3L3=0.5;
                W1L4=0.5;
                W2L5=0.5;
                W1L6=1;
                l1=1;
                l2=0.5;
                l3=0.5;
                l4=1;
                l5=0.5;
            case 'Fox-Goodwin'
                # Fox-Goodwin formula
                w1=-15;
                w2=45;
                w3=-35;
                W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
                # W2=(1/3+w1/4+w2/5+w3/6)/(1+w1/2+w2/3+w3/4); # definition
                # W3=(1/4+w1/5+w2/6+w3/7)/(1+w1/2+w2/3+w3/4); # definition
                W1L1=1;
                W2L2=1/12;
                W3L3=1/12;
                W1L4=0.5;
                W2L5=0.5;
                W1L6=1;
                l1=1;
                l2=0.5;
                l3=1/12;
                l4=1;
                l5=0.5;
            otherwise
                error('No appropriate algorithm specified.');

    # Transfer function denominator
    Omega=omega*dt;
    D=W1L6+2*W2L5*ksi*Omega+W3L3*Omega.^2;
    A31=-Omega.^2/D;
    A32=-1/D*(2*ksi*Omega+W1L1*Omega.^2);
    A33=1-1/D*(1+2*W1L4*ksi*Omega+W2L2*Omega.^2);
    A11=1+l3*A31
    A12=l1+l3*A32
    A13=l2-l3*(1-A33)
    A21=l5*A31
    A22=1+l5*A32
    A23=l4-l5*(1-A33);
    # Amplification matrix
    A=[[A11, A12, A13],
       [A21, A22, A23],
       [A31, A32, A33]]
    # Amplification matrix invariants
    A1=A(1,1)+A(2,2)+A(3,3);
    A2=A(1,1)*A(2,2)-A(1,2)*A(2,1)+A(1,1)*A(3,3)-A(1,3)*A(3,1)+A(2,2)*A(3,3)-A(2,3)*A(3,2);
    A3=A(1,1)*A(2,2)*A(3,3)-A(1,1)*A(2,3)*A(3,2)-A(1,2)*A(2,1)*A(3,3)+A(1,2)*A(2,3)*A(3,1)+A(1,3)*A(2,1)*A(3,2)-A(1,3)*A(2,2)*A(3,1);
    # Transfer function denominator
    a=[1 -A1 A2 -A3]
    # Transfer function nominator
    B1=1/D*dt**2*l3*W1
    B2=1/D*dt**2*(l3*(1-W1)-(A22+A33)*l3*W1+A12*l5*W1+A13*W1);
    B3=1/D*dt**2*(-(A22+A33)*l3*(1-W1)+A12*l5*(1-W1)+A13*(1-W1)+(A22*A33-A23*A32)*l3*W1-(A12*A33-A13*A32)*l5*W1+(A12*A23-A13*A22)*W1);
    B4=1/D*dt**2*((A22*A33-A23*A32)*l3*(1-W1)-(A12*A33-A13*A32)*l5*(1-W1)+(A12*A23-A13*A22)*(1-W1));
    b = [B1,B2,B3,B4]
    # form initial conditions for filter function
    # equivalent external force
    f=-xgtt
    # stiffness
    k=omega.^2
    # damping constants
    c=2*omega*ksi
    # initial acceleration
    utt0=-f(1)-(k*u0+c*ut0)
    U_1=A\[u0;dt*ut0;dt**2*utt0]
    u_1=U_1(1);
    U_2=A\U_1;
    u_2=U_2(1);
    ypast=[u0,u_1,u_2]
    vinit=np.zeros((1,3))
    vinit(3:-1:1) = filter(-a(4:-1:2),1,ypast)
    # main dynamic analysis
    u=filter(b,a,f,vinit);
    # calculate velocity from the following system of equations:
    # 1st: the first scalar equation of the matrix equation (60) in X.Zhou &
    # K.K.Tamma (2004)
    # 2nd: equation of motion (eq.6.12.3 in Chopra:Dynamics of Structures)
    C_u=omega^2*A(1,3)*dt**2-A(1,1)
    C_f=-A(1,3)*dt**2
    C_ut=A(1,2)*dt-A(1,3)*dt**2*2*ksi*omega;
    L=1/D*l3*dt**2*((1-W1)*[0;f(1:end-1)]+W1*f);
    ut=(u+C_u*[u0;u(1:end-1)]+C_f*[0;f(1:end-1)]-L)/C_ut;
    # calculate acceleration from equation of motion
    utt=-omega^2*u-2*ksi*omega*ut;
    return u,ut,utt

def DRHA(k,m,dt,xgtt,ksi=0.05,u0=0,ut0=0,AlgID='U0-V0-CA',rinf=0):
    """Dynamic Response History Analysis (DRHA) of a SDOF system
    
     [U,V,A,f,Es,Ed] = DRHA(k,m,dt,xgtt,ksi,u0,ut0,rinf)
    
     Description
         Determine the time history of structural response of a SDOF system
    
     Input parameters
         #k# (scalar): is the stiffness of the system.
         #m# (scalar) is the lumped masses of the structure.
         #dt# (scalar): time step of the response history analysis from which
             the response spectrum is calculated
         #xgtt# ([#nstep# x 1]): column vector of the acceleration history of
             the excitation imposed at the base. #nstep# is the number of time
             steps of the dynamic response.
         #ksi# (scalar): ratio of critical damping of the SDOF system. Default
             value 0.05.
         #u0# (scalar): initial displacement of the SDOF system. Default value
             0.
         #ut0# (scalar): initial velocity of the SDOF system. Default value 0.
         #rinf# (scalar): minimum absolute value of the eigenvalues of the
             amplification matrix. For the amplification matrix see eq.(61) in
             Zhou & Tamma (2004). Default value 1.
    
     Output parameters
         #U# ([1 x #nstep#]): displacement time history.
         #V# ([1 x #nstep#]): velocity time history.
         #A# ([1 x #nstep#]): acceleration time history.
         #f# ([1 x #nstep#]): equivalent static force time history.
         #Es# ([1 x #nstep#]): time-history of the recoverable
             strain energy of the system (total and not incremental).
         #Ed# ([1 x #nstep#]): time-history of the energy
             dissipated by viscoelastic damping during each time step
             (incremental). cumsum(#Ed#) gives the time history of the total
             energy dissipated at dof #i# from the start of the dynamic
             analysis.
    
    __________________________________________________________________________
     Copyright (c) 2018
         George Papazafeiropoulos
         Captain, Infrastructure Engineer, Hellenic Air Force
         Civil Engineer, M.Sc., Ph.D. candidate, NTUA
         Email: gpapazafeiropoulos@yahoo.gr
     _________________________________________________________________________"""

    ## Calculation
    # Number of time integration steps
    nstep=len(xgtt)
    # construct lumped mass matrix
    M=diag(m,0)
    # Assemble stiffness matrix
    ndofs=size(M,1)
    K=np.zeros(ndofs+1)
    K=K-diag(k,-1)-diag(k,1)
    K(1:end-1,1:end-1)=K(1:end-1,1:end-1)+diag(k,0)
    K(2:end,2:end)=K(2:end,2:end)+diag(k,0)
    K(end,:)=[]
    K(:,end)=[]
    # Calculate eigenvalues, eigenvectors and their number
    Eigvec,Eigval=eig(K,M)
    # Assemble damping matrix
    C=np.zeros(ndofs)
    for i=1:ndofs
        c=2*ksi* np.sqrt(Eigval(i,i))*(((M*Eigvec(:,i))*Eigvec(:,i).T)*M)
        C=C+c
    
    # Take the eigenvalues in column vector and sort in ascending order of
    # eigenfrequency:
    D1=diag(Eigval,0)
    # Generalized masses Mn for all eigenmodes from eq.(13.1.5) of Chopra
    # (2012).
    Mn=diag(Eigvec.T*M*Eigvec);
    # Ln coefficients from eq.(13.1.5) of Chopra (2012).
    Ln=Eigvec.T*M
    # Gamman coefficients from eq.(13.1.5) of Chopra (2012).
    Gamman=Ln./Mn
    # Eigenperiods of the building
    omega=D1.^0.5
    # Initial displacements
    u0mod=Eigvec\u0
    # Normalization
    u0mod=u0mod./Mn
    # Initial velocities
    ut0mod=Eigvec\ut0
    # Normalization
    ut0mod=ut0mod./Mn
    # Displacements, velocities and accelerations of the response of the
    # eigenmodes of the structure for the given earthquake
    neig=size(Eigvec,1)
    U=np.zeros(neig,nstep)
    V=np.zeros(neig,nstep)
    A=np.zeros(neig,nstep)
    f=np.zeros(neig,nstep)
    for i=1:neig
        [u,ut,utt] = LIDA(dt,xgtt,omega(i),ksi,u0mod(i),ut0mod(i),AlgID,rinf);
        U=U+Gamman(i)*Eigvec(:,i)*u.transpose()
        V=V+Gamman(i)*Eigvec(:,i)*ut.transpose()
        A=A+Gamman(i)*Eigvec(:,i)*utt.transpose()
        f=f+Gamman(i)*omega(i)^2*(M*Eigvec(:,i))*u.transpose()
    
    Es=cumsum(K*U).^2./k(:,ones(1,nstep))/2;
    Ed=cumsum(C*V).*diff([-V;zeros(1,nstep)])*dt;
        
 # def NLIDABLKIN(dt,xgtt,m,k_hi,k_lo,uy,varargin):
    #     """Non Linear Implicit Dynamic Analysis of a bilinear kinematic hardening
    #      hysteretic structure with elastic damping
        
    #      [u,ut,utt,Fs,Ey,Es,Ed,jiter] = NLIDABLKIN(dt,xgtt,m,k_hi,k_lo,uy,ksi,...
    #          AlgID,u0,ut0,rinf,maxtol,jmax,dak)
    #          General linear implicit direct time integration of second order
    #          differential equations of a bilinear elastoplastic hysteretic SDOF
    #          dynamic system with elastic damping, with lumped mass.
        
    #      Description
    #          The General Single Step Single Solve (GSSSS) family of algorithms
    #          published by X.Zhou & K.K.Tamma (2004) is employed for direct time
    #          integration of the general linear or nonlinear structural Single
    #          Degree of Freedom (SDOF) dynamic problem. Selection among 9
    #          algorithms, all designed according to the above journal article, can
    #          be made in this routine. These algorithms encompass the scope of
    #          Linear Multi-Step (LMS) methods and are limited by the Dahlquist
    #          barrier theorem (Dahlquist,1963).
        
    #      Input parameters
    #          #dt# (scalar) is the time step of the integration
    #          #xgtt# ([#NumSteps# x 1]) is the acceleration time history which is
    #              imposed at the lumped mass of the SDOF structure.
    #          #m# (scalar) is the lumped masses of the structure. Define the
    #              lumped masses from the top to the bottom, excluding the fixed dof
    #              at the base
    #          #k_hi# (scalar): is the initial stiffness of the system before
    #              its first yield, i.e. the high stiffness. Give the stiffness of
    #              each storey from top to bottom.
    #          #k_lo# (scalar): is the post-yield stiffness of the system,
    #              i.e. the low stiffness. Give the stiffness of each storey from
    #              top to bottom.
    #          #uy# (scalar): is the yield limit of the stiffness elements of
    #              the structure. The element is considered to yield, if the
    #              interstorey drift between degrees of freedom i and i+1 exceeds
    #              uy(i). Give the yield limit of each storey from top to bottom.
    #          #ksi# (scalar): ratio of critical viscous damping of the system,
    #              assumed to be unique for all damping elements of the structure.
    #          #AlgID# (string as follows) is the algorithm to be used for the time
    #              integration. It can be one of the following strings for superior
    #              optimally designed algorithms:
    #                  'generalized a-method': The generalized a-method (Chung &
    #                  Hulbert, 1993)
    #                  'HHT a-method': The Hilber-Hughes-Taylor method (Hilber,
    #                  Hughes & Taylor, 1977)
    #                  'WBZ': The Wood烹ossak忙ienkiewicz method (Wood, Bossak &
    #                  Zienkiewicz, 1980)
    #                  'U0-V0-Opt': Optimal numerical dissipation and dispersion
    #                  zero order displacement zero order velocity algorithm
    #                  'U0-V0-CA': Continuous acceleration (zero spurious root at
    #                  the low frequency limit) zero order displacement zero order
    #                  velocity algorithm
    #                  'U0-V0-DA': Discontinuous acceleration (zero spurious root at
    #                  the high frequency limit) zero order displacement zero order
    #                  velocity algorithm
    #                  'U0-V1-Opt': Optimal numerical dissipation and dispersion
    #                  zero order displacement first order velocity algorithm
    #                  'U0-V1-CA': Continuous acceleration (zero spurious root at
    #                  the low frequency limit) zero order displacement first order
    #                  velocity algorithm
    #                  'U0-V1-DA': Discontinuous acceleration (zero spurious root at
    #                  the high frequency limit) zero order displacement first order
    #                  velocity algorithm
    #                  'U1-V0-Opt': Optimal numerical dissipation and dispersion
    #                  first order displacement zero order velocity algorithm
    #                  'U1-V0-CA': Continuous acceleration (zero spurious root at
    #                  the low frequency limit) first order displacement zero order
    #                  velocity algorithm
    #                  'U1-V0-DA': Discontinuous acceleration (zero spurious root at
    #                  the high frequency limit) first order displacement zero order
    #                  velocity algorithm
    #                  'Newmark ACA': Newmark Average Constant Acceleration method
    #                  'Newmark LA': Newmark Linear Acceleration method
    #                  'Newmark BA': Newmark Backward Acceleration method
    #                  'Fox-Goodwin': Fox-Goodwin formula
    #          #u0# (scalar) is the initial displacement.
    #          #ut0# (scalar) is the initial velocity.
    #          #rinf# (scalar) is the minimum absolute value of the eigenvalues of
    #              the amplification matrix. For the amplification matrix see
    #              eq.(61) in Zhou & Tamma (2004).
    #          #maxtol# (scalar) is the maximum tolerance of convergence of the Full
    #              Newton Raphson method for numerical computation of acceleration.
    #          #jmax# (scalar) is the maximum number of iterations per increment. If
    #              #jmax#=0 then iterations are not performed and the #maxtol#
    #              parameter is not taken into account.
    #          #dak# (scalar) is the infinitesimal acceleration for the
    #              calculation of the derivetive required for the convergence of the
    #              Newton-Raphson iteration.
        
    #      Output parameters
    #          #u# ([1 x #NumSteps#]) is the time-history of displacement
    #          #ut# ([1 x #NumSteps#]) is the time-history of velocity
    #          #utt# ([1 x #NumSteps#]) is the time-history of acceleration
    #          #Fs# ([1 x #NumSteps#]) is the time-history of the internal
    #              force of the structure analysed.
    #          #Ey# ([1 x #NumSteps#]) is the time history of the sum of the
    #              energy dissipated by yielding during each time step and the
    #              recoverable strain energy of the system (incremental).
    #              cumsum(#Ey#)-#Es# gives the time history of the total energy
    #              dissipated by yielding from the start of the dynamic analysis.
    #          #Es# ([1 x #NumSteps#]) is the time-history of the recoverable
    #              strain energy of the system (total and not incremental).
    #          #Ed# ([1 x #NumSteps#]) is the time-history of the energy
    #              dissipated by viscoelastic damping during each time step
    #              (incremental). cumsum(#Ed#) gives the time history of the total
    #              energy dissipated from the start of the dynamic analysis.
    #          #jiter# ([1 x #NumSteps#]) is the iterations per increment
        
    #      Notation in the code
    #          u   =displacement
    #          un  =displacement after increment n
    #          ut  =velocity
    #          utn =velocity after increment n
    #          utt =acceleration
    #          uttn=acceleration after increment n
        
    #     __________________________________________________________________________
    #      Copyright (c) 2018
    #          George Papazafeiropoulos
    #          Captain, Infrastructure Engineer, Hellenic Air Force
    #          Civil Engineer, M.Sc., Ph.D. candidate, NTUA
    #          Email: gpapazafeiropoulos@yahoo.gr
    #      _________________________________________________________________________"""
    #       # required inputs
    #         # if ~isscalar(dt)
    #         #     error('dt is not scalar')
    #         # end
    #         # if dt<=0
    #         #     error('dt is zero or negative')
    #         # end
    #         # if ~isvector(xgtt)
    #         #     error('xgtt is not vector')
    #         # end
    #         # if ~isscalar(m)
    #         #     error('m is not scalar')
    #         # end
    #         # if m<=0
    #         #     error('m is zero or negative')
    #         # end
    #         # if ~isscalar(k_hi)
    #         #     error('k_hi is not scalar')
    #         # end
    #         # if k_hi<=0
    #         #     error('k_hi is zero or negative')
    #         # end
    #         # if ~isscalar(k_lo)
    #         #     error('k_lo is not scalar')
    #         # end
    #         # if k_lo<=0
    #         #     error('k_lo is zero or negative')
    #         # end
    #         # if ~isscalar(uy)
    #         #     error('uy is not scalar')
    #         # end
    #         # if uy<=0
    #         #     error('uy is zero or negative')
    #         # end
    #         # # optional inputs
    #         # if ~isscalar(ksi)
    #         #     error('ksi is not scalar')
    #         # end
    #         # if ksi<=0
    #         #     error('ksi is zero or negative')
    #         # end
    #         # if ~isscalar(u0)
    #         #     error('u0 is not scalar')
    #         # end
    #         # if ~isscalar(ut0)
    #         #     error('ut0 is not scalar')
    #         # end
    #         # if ~isscalar(rinf)
    #         #     error('rinf is not scalar')
    #         # end
    #         # if rinf<0 || rinf>1
    #         #     error('rinf is lower than 0 or higher than 1')
    #         # end
    #         # if ~isscalar(maxtol)
    #         #     error('maxtol is not scalar')
    #         # end
    #         # if maxtol<=0
    #         #     error('maxtol is zero or negative')
    #         # end
    #         # if ~isscalar(jmax)
    #         #     error('jmax is not scalar')
    #         # end
    #         # if jmax<0
    #         #     error('jmax is negative')
    #         # end
    #         # if floor(jmax)!=jmax
    #         #     error('jmax is not integer')
    #         # end
    #         # if ~isscalar(dak)
    #         #     error('dak is not scalar')
    #         # end
    #         # if dak<=0
    #         #     error('dak is zero or negative')
    #         # end

    #     ## Calculation
    #     # Set integration constants
    #     if all(size(AlgID)==[1,14])
    #         # define integration constants explicitly
    #         w1=AlgID(1);
    #         w2=AlgID(2);
    #         w3=AlgID(3);
    #         W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #         # W2=(1/3+w1/4+w2/5+w3/6)/(1+w1/2+w2/3+w3/4); # definition
    #         # W3=(1/4+w1/5+w2/6+w3/7)/(1+w1/2+w2/3+w3/4); # definition
    #         W1L1=AlgID(4);
    #         W2L2=AlgID(5);
    #         W3L3=AlgID(6);
    #         W1L4=AlgID(7);
    #         W2L5=AlgID(8);
    #         W1L6=AlgID(9);
    #         l1=  AlgID(10);
    #         l2=  AlgID(11);
    #         l3=  AlgID(12);
    #         l4=  AlgID(13);
    #         l5=  AlgID(14);
    #     else
    #         switch AlgID
    #             case 'U0-V0-Opt'
    #                 # zero-order displacement & velocity overshooting behavior and
    #                 # optimal numerical dissipation and dispersion
    #                 # rinf must belong to [0 1]
    #                 if rinf<0
    #                     rinf=0;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
                    
    #                 if rinf>1
    #                     rinf=1; # mid-point rule a-form algorithm
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
                    
    #                 w1=-15*(1-2*rinf)/(1-4*rinf); # suggested
    #                 w2=15*(3-4*rinf)/(1-4*rinf); # suggested
    #                 w3=-35*(1-rinf)/(1-4*rinf); # suggested
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=1/(1+rinf);
    #                 W2L2=1/2/(1+rinf);
    #                 W3L3=1/2/(1+rinf)^2;
    #                 W1L4=1/(1+rinf);
    #                 W2L5=1/(1+rinf)^2; # suggested
    #                 W1L6=(3-rinf)/2/(1+rinf);
    #                 l1=1;
    #                 l2=1/2;
    #                 l3=1/2/(1+rinf);
    #                 l4=1;
    #                 l5=1/(1+rinf);
    #             case 'U0-V0-CA'
    #                 # zero-order displacement & velocity overshooting behavior and
    #                 # continuous acceleration
    #                 # rinf must belong to [1/3 1]
    #                 if rinf<1/3
    #                     rinf=1/3;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1/3');
    #                 end
    #                 if rinf>1
    #                     rinf=1; # Newmark average acceleration a-form algorithm
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
    #                 end
    #                 w1=-15*(1-5*rinf)/(3-7*rinf); # suggested
    #                 w2=15*(1-13*rinf)/(3-7*rinf); # suggested
    #                 w3=140*rinf/(3-7*rinf); # suggested
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=(1+3*rinf)/2/(1+rinf);
    #                 W2L2=(1+3*rinf)/4/(1+rinf);
    #                 W3L3=(1+3*rinf)/4/(1+rinf)^2;
    #                 W1L4=(1+3*rinf)/2/(1+rinf);
    #                 W2L5=(1+3*rinf)/2/(1+rinf)^2; # suggested
    #                 W1L6=1;
    #                 l1=1;
    #                 l2=1/2;
    #                 l3=1/2/(1+rinf);
    #                 l4=1;
    #                 l5=1/(1+rinf);
    #             case 'U0-V0-DA'
    #                 # zero-order displacement & velocity overshooting behavior and
    #                 # discontinuous acceleration
    #                 # rinf must belong to [0 1]
    #                 if rinf<0
    #                     rinf=0;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
    #                 end
    #                 if rinf>1
    #                     rinf=1; # Newmark average acceleration a-form algorithm
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
    #                 end
    #                 w1=-15; # suggested
    #                 w2=45; # suggested
    #                 w3=-35; # suggested
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=1;
    #                 W2L2=1/2;
    #                 W3L3=1/2/(1+rinf);
    #                 W1L4=1;
    #                 W2L5=1/(1+rinf); # suggested
    #                 W1L6=(3+rinf)/2/(1+rinf);
    #                 l1=1;
    #                 l2=1/2;
    #                 l3=1/2/(1+rinf);
    #                 l4=1;
    #                 l5=1/(1+rinf);
    #             case 'U0-V1-Opt'
    #                 # zero-order displacement & first-order velocity overshooting
    #                 # behavior and optimal numerical dissipation and dispersion
    #                 # This is the generalized a-method (Chung & Hulbert, 1993)
    #                 # rinf must belong to [0 1]
    #                 if rinf<0
    #                     rinf=0;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
    #                 end
    #                 if rinf>1
    #                     rinf=1;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
    #                 end
    #                 w1=-15*(1-2*rinf)/(1-4*rinf);
    #                 w2=15*(3-4*rinf)/(1-4*rinf);
    #                 w3=-35*(1-rinf)/(1-4*rinf);
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=1/(1+rinf);
    #                 W2L2=1/2/(1+rinf);
    #                 W3L3=1/(1+rinf)^3;
    #                 W1L4=1/(1+rinf);
    #                 W2L5=(3-rinf)/2/(1+rinf)^2;
    #                 W1L6=(2-rinf)/(1+rinf);
    #                 l1=1;
    #                 l2=1/2;
    #                 l3=1/(1+rinf)^2;
    #                 l4=1;
    #                 l5=(3-rinf)/2/(1+rinf);
    #             case 'generalized a-method'
    #                 # zero-order displacement & first-order velocity overshooting
    #                 # behavior and optimal numerical dissipation and dispersion
    #                 # This is the generalized a-method (Chung & Hulbert, 1993)
    #                 # rinf must belong to [0 1]
    #                 if rinf<0
    #                     rinf=0;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
    #                 end
    #                 if rinf>1
    #                     rinf=1;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
    #                 end
    #                 w1=-15*(1-2*rinf)/(1-4*rinf);
    #                 w2=15*(3-4*rinf)/(1-4*rinf);
    #                 w3=-35*(1-rinf)/(1-4*rinf);
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=1/(1+rinf);
    #                 W2L2=1/2/(1+rinf);
    #                 W3L3=1/(1+rinf)^3;
    #                 W1L4=1/(1+rinf);
    #                 W2L5=(3-rinf)/2/(1+rinf)^2;
    #                 W1L6=(2-rinf)/(1+rinf);
    #                 l1=1;
    #                 l2=1/2;
    #                 l3=1/(1+rinf)^2;
    #                 l4=1;
    #                 l5=(3-rinf)/2/(1+rinf);
    #             case 'U0-V1-CA'
    #                 # zero-order displacement & first-order velocity overshooting
    #                 # behavior and continuous acceleration
    #                 # This is the Hilber-Hughes-Taylor method (Hilber, Hughes &
    #                 # Taylor, 1977)
    #                 # rinf must belong to [1/2 1]
    #                 if rinf<1/2
    #                     rinf=1/2;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1/2');
    #                 end
    #                 if rinf>1
    #                     rinf=1;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
    #                 end
    #                 w1=-15*(1-2*rinf)/(2-3*rinf);
    #                 w2=15*(2-5*rinf)/(2-3*rinf);
    #                 w3=-35*(1-3*rinf)/2/(2-3*rinf);
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=2*rinf/(1+rinf);
    #                 W2L2=rinf/(1+rinf);
    #                 W3L3=2*rinf/(1+rinf)^3;
    #                 W1L4=2*rinf/(1+rinf);
    #                 W2L5=rinf*(3-rinf)/(1+rinf)^2;
    #                 W1L6=1;
    #                 l1=1;
    #                 l2=1/2;
    #                 l3=1/(1+rinf)^2;
    #                 l4=1;
    #                 l5=(3-rinf)/2/(1+rinf);
    #             case 'HHT a-method'
    #                 # zero-order displacement & first-order velocity overshooting
    #                 # behavior and continuous acceleration
    #                 # This is the Hilber-Hughes-Taylor method (Hilber, Hughes &
    #                 # Taylor, 1977)
    #                 # rinf must belong to [1/2 1]
    #                 if rinf<1/2
    #                     rinf=1/2;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1/2');
    #                 end
    #                 if rinf>1
    #                     rinf=1;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
    #                 end
    #                 w1=-15*(1-2*rinf)/(2-3*rinf);
    #                 w2=15*(2-5*rinf)/(2-3*rinf);
    #                 w3=-35*(1-3*rinf)/2/(2-3*rinf);
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=2*rinf/(1+rinf);
    #                 W2L2=rinf/(1+rinf);
    #                 W3L3=2*rinf/(1+rinf)^3;
    #                 W1L4=2*rinf/(1+rinf);
    #                 W2L5=rinf*(3-rinf)/(1+rinf)^2;
    #                 W1L6=1;
    #                 l1=1;
    #                 l2=1/2;
    #                 l3=1/(1+rinf)^2;
    #                 l4=1;
    #                 l5=(3-rinf)/2/(1+rinf);
    #             case 'U0-V1-DA'
    #                 # zero-order displacement & first-order velocity overshooting
    #                 # behavior and discontinuous acceleration
    #                 # This is the Wood烹ossak忙ienkiewicz method (Wood, Bossak &
    #                 # Zienkiewicz, 1980)
    #                 # rinf must belong to [0 1]
    #                 if rinf<0
    #                     rinf=0;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
    #                 end
    #                 if rinf>1
    #                     rinf=1;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
    #                 end
    #                 w1=-15;
    #                 w2=45;
    #                 w3=-35;
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=1;
    #                 W2L2=1/2;
    #                 W3L3=1/(1+rinf)^2;
    #                 W1L4=1;
    #                 W2L5=(3-rinf)/2/(1+rinf);
    #                 W1L6=2/(1+rinf);
    #                 l1=1;
    #                 l2=1/2;
    #                 l3=1/(1+rinf)^2;
    #                 l4=1;
    #                 l5=(3-rinf)/2/(1+rinf);
    #             case 'WBZ'
    #                 # zero-order displacement & first-order velocity overshooting
    #                 # behavior and discontinuous acceleration
    #                 # This is the Wood烹ossak忙ienkiewicz method (Wood, Bossak &
    #                 # Zienkiewicz, 1980)
    #                 # rinf must belong to [0 1]
    #                 if rinf<0
    #                     rinf=0;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
    #                 end
    #                 if rinf>1
    #                     rinf=1;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
    #                 end
    #                 w1=-15;
    #                 w2=45;
    #                 w3=-35;
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=1;
    #                 W2L2=1/2;
    #                 W3L3=1/(1+rinf)^2;
    #                 W1L4=1;
    #                 W2L5=(3-rinf)/2/(1+rinf);
    #                 W1L6=2/(1+rinf);
    #                 l1=1;
    #                 l2=1/2;
    #                 l3=1/(1+rinf)^2;
    #                 l4=1;
    #                 l5=(3-rinf)/2/(1+rinf);
    #             case 'U1-V0-Opt'
    #                 # first-order displacement & zero-order velocity overshooting
    #                 # behavior and optimal numerical dissipation and dispersion
    #                 # rinf must belong to [0 1]
    #                 if rinf<0
    #                     rinf=0;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
    #                 end
    #                 if rinf>1
    #                     rinf=1; # mid-point rule a-form algorithm
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
    #                 end
    #                 w1=-30*(3-8*rinf+6*rinf^2)/(9-22*rinf+19*rinf^2);
    #                 w2=15*(25-74*rinf+53*rinf^2)/2/(9-22*rinf+19*rinf^2);
    #                 w3=-35*(3-10*rinf+7*rinf^2)/(9-22*rinf+19*rinf^2);
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=(3-rinf)/2/(1+rinf);
    #                 W2L2=1/(1+rinf)^2;
    #                 W3L3=1/(1+rinf)^3;
    #                 W1L4=(3-rinf)/2/(1+rinf);
    #                 W2L5=2/(1+rinf)^3;
    #                 W1L6=(2-rinf)/(1+rinf);
    #                 l1=1;
    #                 l2=1/2;
    #                 l3=1/2/(1+rinf);
    #                 l4=1;
    #                 l5=1/(1+rinf);
    #             case 'U1-V0-CA'
    #                 # first-order displacement & zero-order velocity overshooting
    #                 # behavior and continuous acceleration
    #                 # rinf must belong to [1/2 1]
    #                 if rinf<1/2
    #                     rinf=1/2;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1/2');
    #                 end
    #                 if rinf>1
    #                     rinf=1; # Newmark average acceleration a-form algorithm
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
    #                 end
    #                 w1=-60*(2-8*rinf+7*rinf^2)/(11-48*rinf+41*rinf^2);
    #                 w2=15*(37-140*rinf+127*rinf^2)/2/(11-48*rinf+41*rinf^2);
    #                 w3=-35*(5-18*rinf+17*rinf^2)/(11-48*rinf+41*rinf^2);
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=(1+3*rinf)/2/(1+rinf);
    #                 W2L2=2*rinf/(1+rinf)^2;
    #                 W3L3=2*rinf/(1+rinf)^3;
    #                 W1L4=(1+3*rinf)/2/(1+rinf);
    #                 W2L5=4*rinf/(1+rinf)^3;
    #                 W1L6=1;
    #                 l1=1;
    #                 l2=1/2;
    #                 l3=1/2/(1+rinf);
    #                 l4=1;
    #                 l5=1/(1+rinf);
    #             case 'U1-V0-DA'
    #                 # first-order displacement & zero-order velocity overshooting behavior
    #                 # and discontinuous acceleration
    #                 # This is the Newmark average acceleration a-form algorithm
    #                 # rinf must belong to [0 1]
    #                 if rinf<0
    #                     rinf=0;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 0');
    #                 end
    #                 if rinf>1
    #                     rinf=1;
    #                     warning('Minimum absolute eigenvalue of amplification matrix is set to 1');
    #                 end
    #                 w1=-30*(3-4*rinf)/(9-11*rinf);
    #                 w2=15*(25-37*rinf)/2/(9-11*rinf);
    #                 w3=-35*(3-5*rinf)/(9-11*rinf);
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=(3+rinf)/2/(1+rinf);
    #                 W2L2=1/(1+rinf);
    #                 W3L3=1/(1+rinf)^2;
    #                 W1L4=(3+rinf)/2/(1+rinf);
    #                 W2L5=2/(1+rinf)^2;
    #                 W1L6=2/(1+rinf);
    #                 l1=1;
    #                 l2=1/2;
    #                 l3=1/(1+rinf)^2;
    #                 l4=1;
    #                 l5=(3-rinf)/2/(1+rinf);
    #             case 'Newmark ACA'
    #                 # Newmark Average Constant Acceleration method
    #                 w1=-15;
    #                 w2=45;
    #                 w3=-35;
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 # W2=(1/3+w1/4+w2/5+w3/6)/(1+w1/2+w2/3+w3/4); # definition
    #                 # W3=(1/4+w1/5+w2/6+w3/7)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=1;
    #                 W2L2=0.25;
    #                 W3L3=0.25;
    #                 W1L4=0.5;
    #                 W2L5=0.5;
    #                 W1L6=1;
    #                 l1=1;
    #                 l2=0.5;
    #                 l3=0.25;
    #                 l4=1;
    #                 l5=0.5;
    #             case 'Newmark LA'
    #                 # Newmark Linear Acceleration method
    #                 w1=-15;
    #                 w2=45;
    #                 w3=-35;
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 # W2=(1/3+w1/4+w2/5+w3/6)/(1+w1/2+w2/3+w3/4); # definition
    #                 # W3=(1/4+w1/5+w2/6+w3/7)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=1;
    #                 W2L2=1/6;
    #                 W3L3=1/6;
    #                 W1L4=0.5;
    #                 W2L5=0.5;
    #                 W1L6=1;
    #                 l1=1;
    #                 l2=0.5;
    #                 l3=1/6;
    #                 l4=1;
    #                 l5=0.5;
    #             case 'Newmark BA'
    #                 # Newmark Backward Acceleration method
    #                 w1=-15;
    #                 w2=45;
    #                 w3=-35;
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 # W2=(1/3+w1/4+w2/5+w3/6)/(1+w1/2+w2/3+w3/4); # definition
    #                 # W3=(1/4+w1/5+w2/6+w3/7)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=1;
    #                 W2L2=0.5;
    #                 W3L3=0.5;
    #                 W1L4=0.5;
    #                 W2L5=0.5;
    #                 W1L6=1;
    #                 l1=1;
    #                 l2=0.5;
    #                 l3=0.5;
    #                 l4=1;
    #                 l5=0.5;
    #             case 'Fox-Goodwin'
    #                 # Fox-Goodwin formula
    #                 w1=-15;
    #                 w2=45;
    #                 w3=-35;
    #                 W1=(1/2+w1/3+w2/4+w3/5)/(1+w1/2+w2/3+w3/4); # definition
    #                 # W2=(1/3+w1/4+w2/5+w3/6)/(1+w1/2+w2/3+w3/4); # definition
    #                 # W3=(1/4+w1/5+w2/6+w3/7)/(1+w1/2+w2/3+w3/4); # definition
    #                 W1L1=1;
    #                 W2L2=1/12;
    #                 W3L3=1/12;
    #                 W1L4=0.5;
    #                 W2L5=0.5;
    #                 W1L6=1;
    #                 l1=1;
    #                 l2=0.5;
    #                 l3=1/12;
    #                 l4=1;
    #                 l5=0.5;
    #             otherwise
    #                 error('No appropriate algorithm specified.');


    #     ## Calculation
    #     # number of analysis increments
    #     NumSteps = len(xgtt);
    #     # Initialize output
    #     u =  np.zeros(1,NumSteps);
    #     ut = np.zeros(1,NumSteps);
    #     utt = np.zeros(1,NumSteps);
    #     Fs = np.zeros(1,NumSteps);
    #     Ey = np.zeros(1,NumSteps);
    #     Es = np.zeros(1,NumSteps);
    #     Ed = np.zeros(1,NumSteps);
    #     jiter = np.zeros(1,NumSteps);
    #     # set initial values of displacement, velocity, acceleration (u0,ut0 and
    #     # utt0 respectively) at n=0.
    #     u(1)=u0;
    #     ut(1)=ut0;
    #     # construct lumped mass matrix
    #     M=diag(m,0);
    #     # calculation for first increment
    #     k=k_hi;
    #     d=0;
    #     FKC0,K0,C0,k,d = BLKIN(u0,ut0,k_hi,k_lo,uy,M,ksi,k,d)
    #     utt0=-xgtt(1)-M\FKC0
    #     utt(1)=utt0
    #     Fs(1)=FKC0
    #     # initial assignments
    #     FKCn=FKC0
    #     Kn=K0
    #     Cn=C0
    #     un=u0
    #     utn=ut0
    #     uttn=utt0
    #     # integration increments n
    #     for n=1:NumSteps-1
    #         # effective force
    #         Feffn1k=-FKCn-Kn*(W1L1*dt*utn+W2L2*dt**2*uttn) -W1L4*dt*Cn*uttn+M*(((1-W1)*xgtt(n)+W1*xgtt(n+1))-uttn);
    #         # effective mass
    #         Meffn=W1L6*M+W2L5*dt*Cn+W3L3*dt**2*Kn
    #         # initial estimate of da
    #         dan=Meffn\Feffn1k
            
    #         # start iteration number k
    #         j=1;
    #         # set initial quotient of variation of da equal to maxtol
    #         quda=maxtol
    #         # full Newton-Raphson iterations k
    #         while max(abs(quda))>=maxtol && j<=jmax:
    #             # iteration k+1 of increment n+1
    #             # _________________________________________________________________
    #             #/
    #             # displacement, velocity, acceleration, internal force, stiffness
    #             # and damping for #uttn+dan#
    #             # calculate the residual #Rn1k# at #uttn+dan#
    #             # update kinematic quantities
    #             un1k=un+l1*utn*dt+l2*uttn*dt**2+l3*dan*dt**2;
    #             utn1k=utn+l4*uttn*dt+l5*dan*dt;
    #             uttn1k=uttn+dan;
    #             # force due to stiffness and damping
    #             FKCn1k, Kn1k, Cn1k,_,_ = BLKIN(un1k,utn1k,k_hi,k_lo,uy,M,ksi,k,d)
    #             # effective force
    #             Feffn1k=-FKCn1k...
    #                 -Kn1k*(W1L1*dt*utn1k+W2L2*dt**2*uttn1k)...
    #                 -Cn1k*W1L4*dt*uttn1k...
    #                 +M*((1-W1)*xgtt(n)+W1*xgtt(n+1)-uttn1k);
    #             # effective mass
    #             Meffn1k=Kn1k*W3L3*dt**2+Cn1k*W2L5*dt+M*W1L6;
    #             # residual
    #             Rn1k=Feffn1k-Meffn1k*dan;
    #             #\_________________________________________________________________

    #             # _________________________________________________________________
    #             #/
    #             # displacement, velocity, acceleration, internal force, stiffness
    #             # and damping for #uttn+(dan+dak)#
    #             # calculate the derivative at #uttn+dan# as:
    #             # #dR/da=(dRn1k-Rn1k)/(uttn+(dan+dak)-(uttn+dan))=(dRn1k-Rn1k)/dak#
    #             # update kinematic quantities
    #             dun1k=un+l1*utn*dt+l2*uttn*dt**2+l3*(dan+dak)*dt**2;
    #             dutn1k=utn+l4*uttn*dt+l5*(dan+dak)*dt;
    #             duttn1k=uttn+(dan+dak);

    #             # force due to stiffness and damping
    #             [dFKCn1k,dKn1k,dCn1k,~,~]=BLKIN(dun1k,dutn1k,k_hi,k_lo,uy,M,ksi,k,d);
    #             # effective force
    #             dFeffn1k=-dFKCn1k...
    #                 -dKn1k*(W1L1*dt*dutn1k+W2L2*dt**2*duttn1k)...
    #                 -dCn1k*W1L4*dt*duttn1k...
    #                 +M*((1-W1)*xgtt(n)+W1*xgtt(n+1)-duttn1k);
    #             # effective mass
    #             dMeffn1k=dKn1k*W3L3*dt**2+dCn1k*W2L5*dt+M*W1L6;
    #             # residual
    #             dRn1k=dFeffn1k-dMeffn1k*duttn1k;
    #             #\_________________________________________________________________
                
    #             # Full Newton-Raphson update:
    #             # #da_new=da-Rn1k/(dR/da)=da*(1-Rn1k/(dRn1k/dak)/da)#
    #             # (to be checked for while loop termination)
    #             quda=(Rn1k./(dRn1k-Rn1k).*dak)./dan
    #             # test if derivative becomes zero
    #             a=np.isinf(quda)
    #             if any(a)
    #                 break
    #                 #quda = np.zeros(size(quda));
                
    #             # update da
    #             dan=(1-quda).*dan
    #             # update iteration number
    #             j=j+1
            
    #         # _____________________________________________________________________
    #         #/
    #         # displacement and its derivatives after iteration k+1 of increment
    #         # n+1
    #         un1k=un+l1*utn*dt+l2*uttn*dt**2+l3*dan*dt**2;
    #         utn1k=utn+l4*uttn*dt+l5*dan*dt;
    #         uttn1k=uttn+dan;
    #         # internal force, stiffness and damping after iteration k+1 of
    #         # increment n+1
    #         [FKCn1k,Kn1k,Cn1k,k,d] = BLKIN(un1k,utn1k,k_hi,k_lo,uy,M,ksi,k,d);
    #         #\_____________________________________________________________________
            
    #         # assignments to output parameters
    #         u(n+1)=un1k
    #         ut(n+1)=utn1k
    #         utt(n+1)=uttn1k
    #         Fs[n+1]=FKCn1k
    #         Ey[n+1]=-(cumsum(FKCn1k-Cn1k*utn1k)+cumsum(FKCn-Cn*utn))/2.*diff([un1k-un;0]);
    #         Es[n+1]=cumsum(FKCn1k-Cn1k*utn1k).^2./k_hi/2;
    #         Ed[n+1]=-(cumsum(Cn1k*utn1k)+cumsum(Cn*utn))/2.*diff([un1k-un;0]);
    #         jiter(n+1)=j-1
    #         # assignments for next increment
    #         FKCn=FKCn1k
    #         Kn=Kn1k
    #         Cn=Cn1k
    #         un=un1k
    #         utn=utn1k
    #         uttn=uttn1k

    # def BLKIN(u,ut,k_hi,k_lo,uy,M,ksi,k,d):
    #      """Bilinear elastoplastic hysteretic model with elastic viscous damping
        
    #      [f,K,C,k,d] = BLKIN(u,ut,k_hi,k_lo,uy,M,ksi,k,d)
    #          Define the internal force vector, tangent stiffness matrix and
    #          tangent damping matrix of a bilinear elastoplastic hysteretic
    #          structure with elastic damping as a function of displacement and
    #          velocity.
        
    #      Description
    #      ---------------------------
    #          The MDOF structure modeled with this function consists of lumped
    #          masses connected with stiffness and damping elements in series. Each
    #          lumped mass has one degree of freedom. The first degree of freedom is
    #          at the top of the structure and the last at its fixed base. However,
    #          the last degree of freedom is not included in the input arguments of
    #          the function, i.e. not contained in #ndof#, as it is always fixed.
    #          The nonlinear stiffness is virtually of the bilinear type, where an
    #          initial stiffness and a post-yield stiffness are defined. The
    #          unloading or reloading curve of this model are parallel to the
    #          initial loading curve, and a hysteresis loop is created by
    #          continuously loading and unloading the structure above its yield
    #          limit. This behavior can be viewed as hardening of the kinematic
    #          type.
    #          An appropriate reference for this function definition is Hughes,
    #          Pister & Taylor (1979): "Implicit-explicit finite elements in
    #          nonlinear transient analysis". This function should be defined in
    #          accordance with equations (3.1), (3.2) and (3.3) of this paper. This
    #          representation has as special cases nonlinear elasticity and a class
    #          of nonlinear �rate-type� viscoelastic materials. Tangent stiffness and
    #          tangent damping matrices are the "consistent" linearized operators
    #          associated to f in the sense of [Hughes & Pister, "Consistent
    #          linearization in mechanics of solids", Computers and Structures, 8
    #          (1978) 391-397].
        
    #      Input parameters
    #      ---------------------
    #          #u# (scalar): absolute displacement.
    #          #ut# (scalar): absolute velocity.
    #          #k_hi# (scalar): initial stiffness of the system before its first
    #              yield, i.e. the high stiffness.
    #          #k_lo# (scalar): post-yield stiffness of the system, i.e. the low
    #              stiffness.
    #          #uy# (scalar): yield limit of the structure. The structure is
    #              considered to yield, if the displacement exceeds uy(i).
    #          #M# (scalar): lumped mass.
    #          #ksi# (scalar): ratio of critical viscous damping of the system,
    #              assumed to be unique for all damping elements of the structure.
    #          #k# (scalar): is the stiffness vector which takes into account
    #              any plastic response of the structure. It is used to record the
    #              status of the structure so that it is known before the next
    #              application of this function at a next (time) step. Initialize by
    #              setting #k#=#k_hi#.
    #          #d# (scalar): is the equilibrium displacement vector which takes into
    #              account any plastic response of the structure. It is used to
    #              record the status of the structure so that it is known before the
    #              next application of this function at a next (time) step.
    #              Initialize by setting #d#=zeros(#ndof#,1).
        
    #      Output parameters
    #      ----------
    #          #f# (scalar): internal force vector of the structure (sum of forces
    #              due to stiffness and damping) at displacement #u# and velocity
    #              #ut#
    #          #K# (scalar): tangent stiffness matrix (nonlinear function of
    #              displacement #u# and velocity #ut#). It is equivalent to the
    #              derivative d(#f#)/d(#u#)
    #          #C# (scalar): tangent damping matrix (nonlinear function of
    #              displacement #u# and velocity #ut#). It is equivalent to the
    #              derivative d(#f#)/d(#u#)
    #          #k# (scalar): is the stiffness vector which takes into account any
    #              plastic response of the structure. It is used to record the
    #              status of the structure so that it is known before the next
    #              application of this function at a next (time) step.
    #          #d# (scalar): is the equilibrium displacement vector which takes into
    #              account any plastic response of the structure. It is used to
    #              record the status of the structure so that it is known before the
    #              next application of this function at a next (time) step.
        
    #      Verification:
    #      -------------------------
    #          u=0:0.2:4;
    #          ut=0.001*ones(1,numel(u));
    #          u=[u,u(end:-1:1)];
    #          ut=[ut,-ut];
    #          u=[u,-u];
    #          ut=[ut,ut(end:-1:1)];
    #          u=[u u];
    #          ut=[ut ut];
    #          k_hi=1000;
    #          k_lo=1;
    #          uy=2;
    #          M=1;
    #          ksi=0.05;
    #          k=k_hi;
    #          d=0;
    #          f=zeros(1,numel(u));
    #          for i=1:numel(u)
    #              [f(i),K,C,k,d] = BLKIN(u(i),ut(i),k_hi,k_lo,uy,M,ksi,k,d);
    #          end
    #          figure()
    #          plot(u,f)
        
    #     __________________________________________________________________________
    #      Copyright (c) 2018
    #          George Papazafeiropoulos
    #          Captain, Infrastructure Engineer, Hellenic Air Force
    #          Civil Engineer, M.Sc., Ph.D. candidate, NTUA
    #          Email: gpapazafeiropoulos@yahoo.gr
    #      _________________________________________________________________________"""

    #     # Elastic tangent stiffness matrix
    #     K=k_hi;
    #     # Elastic tangent damping matrix
    #     C=2*ksi* np.sqrt(K*M);
    #     # force from stiffness (not damping) of the current storey
    #     fK=k*(u(1)-d);
    #     # eq.(46) in ...
    #     fy=k_lo*(u(1))+(k_hi-k_lo)*(uy.*sign(ut(1)));
    #     # check for yielding or load reversal
    #     if k==k_hi && ut(1)>0 && fK>fy
    #         # check for yielding
    #         # the system has just exceeded its positive yield force level
    #         k=k_lo;
    #         d=(1-k_hi/k_lo)*uy;
    #     elif k==k_hi && ut(1)<0 && fK<fy
    #         # check for yielding
    #         # the system has just exceeded its negative yield force level
    #         k=k_lo;
    #         d=(k_hi/k_lo-1)*uy;
    #     elif k==k_lo && fK*(ut(1))<0
    #         # check for load reversal
    #         # the system reloads from negative ultimate displacement or unloads
    #         # from positive ultimate displacement
    #         k=k_hi;
    #         d=(u(1))-k_lo/k_hi*(u(1)-d);
        
    #     fK_bak=k*(u(1)-d);
    #     # Update the elastic tangent stiffness matrix
    #     K=k;
    #     # internal force due to stiffness and damping
    #     f = fK_bak + C*ut;

def FASp(dt,xgtt):
    """Single sided Fourier amplitude spectrum
    
     [f,U] = FASp(dt,xgtt)
         Fourier amplitude spectrum of an earthquake.
    
     Input parameters
         #dt# (scalar) is the time step of the input acceleration time history
             #xgtt#.
         #xgtt# ([#numsteps# x 1]) is the input acceleration time history.
             #numsteps# is the length of the input acceleration time history.
    
     Output parameters
         #f# ([#2^(nextpow2(len(#xgtt#))-1)# x 1]): frequency range in
             which the Fourier amplitudes are calculated.
         #U# ([#2^(nextpow2(len(#xgtt#))-1)# x 1]): Fourier amplitudes
    
    __________________________________________________________________________
     Copyright (c) 2018
         George Papazafeiropoulos
         Captain, Infrastructure Engineer, Hellenic Air Force
         Civil Engineer, M.Sc., Ph.D. candidate, NTUA
         Email: gpapazafeiropoulos@yahoo.gr
     _________________________________________________________________________"""

    # required inputs
    # if ~isscalar(dt)
    #     error('dt is not scalar')
    # if dt<=0
    #     error('dt is zero or negative')
    # if ~isvector(xgtt)
    #     error('xgtt is not vector')

    ## Calculation
    # Nyquist frequency (highest frequency)
    Ny = (1/dt)/2 
    # number of points in xgtt
    L  = len(xgtt)
    # Next power of 2 from length of xgtt
    NFFT = 2**nextpow2(L)
    # frequency spacing
    df = 1/(NFFT*dt)
    # Fourier amplitudes 
    U = abs(fft(xgtt,NFFT))*dt
    # Single sided Fourier amplitude spectrum
    U = U(2:Ny/df+1)
    # frequency range
    f = np.linspace(df,Ny,Ny/df).T
    
def HalfStep(u):
     """Reproduce signal with half time step
    
     uNew = HalfStep(u)
    
     Input parameters
         #u# ([#n# x 1]): input signal with time step dt.
    
     Output parameters
         #uNew# ([#n# x 1]): output signal with time step dt/2.
    
     Verification:
         u=0.2:0.2:4;
         uNew=HalfStep(u);
         figure()
         plot((1:numel(u)),u)
         hold on
         plot((1:0.5:numel(u)),uNew)
    
    __________________________________________________________________________
     Copyright (c) 2018
         George Papazafeiropoulos
         Captain, Infrastructure Engineer, Hellenic Air Force
         Civil Engineer, M.Sc., Ph.D. candidate, NTUA
         Email: gpapazafeiropoulos@yahoo.gr
     _________________________________________________________________________"""


    ## Initial checks
    if nargin>1
        error('Input arguments more than required')
    
    # required inputs
    if ~isvector(u)
        error('u is not vector')
    
    t=False
    if size(u,2)!=1:
        u=u(:)
        t=True
    
    ## Calculation
    a = np.array([([0;u(1:end-1)]+u)/2,u]).T;
    uNew = a(:)
    uNew(1)=[]
    if t:
        uNew=uNew.T

def paramEC8(GroundType,SeismicZone,ImportanceFactor):
    """[#S#,#Tb#,#Tc#,#Td#,#ag#,#b#]=paramEC8(#GroundType#,#SeismicZone#,...
         #ImportanceFactor#)
         Calculation of the properties of the Type 1 design response spectrum
         of EC8. The values are taken from Table 3.2 and Figure 3.2 of
         EC8-1[3.2.2.2(2)P]
    
     Input parameters
     ---------------------------
         #GroundType# (scalar string) is type of the ground type. Possible
             values are: 'A', 'B', 'C', 'D', 'E'.
         #SeismicZone# (scalar integer) is the seismic zone. Possible
             values are: 1, 2, 3.
         #ImportanceFactor# (scalar double) is the importance factor.
    
     Output parameters
     ---------------------------
         #S# (scalar): soil factor
         #Tb# (scalar): lower limit of the period of the constant spectral
             acceleration branch 
         #Tc# (scalar): upper linlit of the period of the constant spectral
             acceleration branch 
         #Td# (scalar): value defining the beginning of the constant
             displacement response range of the spectrum
         #ag# (scalar): design ground acceleration on type A ground
         #b# (scalar): lower bound factor for the horizontal design spectrum
    
     Example:
     ---------------------------
         q=3; # behavior factor
         GroundType='A';
         SeismicZone=1;
         ImportanceFactor=1;
         [S,Tb,Tc,Td,ag,b]=paramEC8(GroundType,SeismicZone,ImportanceFactor)
    
    __________________________________________________________________________
     Copyright (c) 2018
         George Papazafeiropoulos
         Captain, Infrastructure Engineer, Hellenic Air Force
         Civil Engineer, M.Sc., Ph.D. candidate, NTUA
         Email: gpapazafeiropoulos@yahoo.gr
     _________________________________________________________________________"""

    # Calculate S,Tb,Tc,Td according to ground type
    # switch GroundType
    #     case 'A'
    #         S=1;
    #         Tb=0.15;
    #         Tc=0.4;
    #     case 'B'
    #         S=1.2;
    #         Tb=0.15;
    #         Tc=0.5;
    #     case 'C'
    #         S=1.15;
    #         Tb=0.2;
    #         Tc=0.6;
    #     case 'D'
    #         S=1.35;
    #         Tb=0.2;
    #         Tc=0.8;
    #     case 'E'
    #         S=1.4;
    #         Tb=0.15;
    #         Tc=0.5;
    
    # Td=2; # alternatively, according to national annex (Td=2.5)
    # # agRHor: reference peak ground acceleration on type A ground
    # switch SeismicZone
    #     case 1
    #         agR=0.16*9.81; # seismic zone I
    #     case 2
    #         agR=0.24*9.81; # seismic zone II
    #     case 3
    #         agR=0.32*9.81; # seismic zone III

    # # Importance class & Importance factor (given in EC8-1[4.2.5(4)]
    # ag=ImportanceFactor*agR; # EC8-1[3.2.1(3)]
    # b=0.2; # EC8-1[3.2.2.5(4)P-NOTE]
    pass

def specAccEC8(T,q,S,Tb,Tc,Td,ag,b):
    """#SA# = specAccEC8(#T#,#q#,#S#,#Tb#,#Tc#,#Td#,#ag#,#b#)
         Calculation of PSA according to EC8-1[3.2.2.5(4)P]
    
     Input parameters
         #T# ([#n# x 1]): period vector
         #q# (scalar): behavior factor
         #S# (scalar): soil factor
         #Tb# (scalar): lower limit of the period of the constant spectral
             acceleration branch 
         #Tc# (scalar): upper linlit of the period of the constant spectral
             acceleration branch 
         #Td# (scalar): value defining the beginning of the constant
             displacement response range of the spectrum
         #ag# (scalar): design ground acceleration on type A ground
         #b# (scalar): lower bound factor for the horizontal design spectrum
    
     Output parameters
         #SA# ([#n# x 1]): spectral acceleration
    
     Example:
         T=(0.04:0.04:4)'
         q=3;
         GroundType='A';
         SeismicZone=1;
         ImportanceFactor=1;
         [S,Tb,Tc,Td,ag,b]=paramEC8(GroundType,SeismicZone,ImportanceFactor);
         SA = specAccEC8(T,q,S,Tb,Tc,Td,ag,b);
         plot(T,SA)
    
    __________________________________________________________________________
     Copyright (c) 2018
         George Papazafeiropoulos
         Captain, Infrastructure Engineer, Hellenic Air Force
         Civil Engineer, M.Sc., Ph.D. candidate, NTUA
         Email: gpapazafeiropoulos@yahoo.gr
     _________________________________________________________________________"""

    # T1= 0<=T & T<=Tb
    # SA(T1)=ag*S*(2/3+T(T1)./Tb.*(2.5./q-2/3))
    # T2= Tb<=T & T<=Tc
    # SA(T2)=ag*S*2.5/q
    # T3= Tc<=T & T<=Td
    # SA(T3)=max(ag*S.*2.5/q*Tc/T(T3),b*ag)
    # T4= Td<=T
    # SA(T4)=max(ag*S*2.5/q.*Tc*Td/T(T4).^2,b*ag)
    # SA=SA.transpose()
    pass
    