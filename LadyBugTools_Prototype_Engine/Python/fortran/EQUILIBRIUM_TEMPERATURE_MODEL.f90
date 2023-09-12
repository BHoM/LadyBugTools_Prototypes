!***********************************************************************
subroutine equibtemp(albedo,depth,tstep,vpd,u,ta,tn,solrad,longrad,fc,tw0,z0,rn,le,deltas,tw,evap,ierr)
!***********************************************************************
! SUBROUTINE TO CALCULATE THE EVAPORATION, USING DAILY DATA, FROM A
! WATER BODY USING THE EQUILIBRIUM TEMPERATURE MODEL OF de Bruin,
! H.A.R., 1982, J.Hydrol, 59, 261-274
! INPUT:
! ALBEDO - ALBEDO OF THE WATER BODY
! DEPTH - DEPTH OF THE WATER BODY (m)
! TSTEP - THE TIME STEP FOR THE MODEL TO USE (days)
! VPD - VAPOUR PRESSURE DEFICIT (mb)
! U - WIND SPEED (m s-1)
! TA - AIR TEMPERATURE (deg.C)
! TN - WET BULB TEMPERATURE (deg.C)
! SOLRAD - DOWNWELLING SOLAR RADIATION (W m-2 per day)
! LONGRAD - DOWNWELLING LONG WAVE RADIATION (W m-2 per day)
! FC - CLOUDINESS FACTOR
! TW0 - TEMPERATURE OF THE WATER ON THE PREVIOUS TIME STEP (deg.C)
! OUTPUT:
! RN - NET RADIATION (W m-2 per day)
! LE - LATENT HEAT FLUX (W m-2 per day)
! DELTAS - CHANGE IN HEAT STORAGE (W m-2 per day)
! TW - TEMPERATURE OF THE WATER AT THE END OF THE TIME PERIOD
! (deg.C)
! EVAP - EVAPORATION CALCULATED USING THE PENMAN-MONTEITH FORMULA
! (mm per day)
! IERR - ERROR FLAG
! 0 = OK
! 1 = ALBEDO =< 0 OR => 1
! 2 = DEPTH =< 0
! 3 = AIR TEMPERATURE < WET BULB TEMPERATURE
! 4 = DOWNWELLING SOLAR RADIATION =< 0
! 5 = WIND SPEED < 0.01 m/s
! 6 = VPD =< 0
! CONSTANTS
! LAMBDA - LATENT HEAT OF VAPORISATION (MJ kg-1)
! GAMMA - PSCHROMETRIC CONSTANT (kPa deg.C-1)
! RHOW - DENSITY OF WATER (kg m-3)
! CW - SPECIFIC HEAT OF WATER (MJ kg-1 deg.C-1)
! RHO - DENSITY OF AIR (kg m-3)
! CP - SPECIFIC HEAT OF AIR (KJ kg-1 deg.C-1
! SIGMA - STEFAN-BOLTZMANN CONSTANT (MJ m-2 deg.C-4 d-1)
! K - VON KARMAN CONSTANT
! DEGABS - DIFFERENCE BETWEEN DEGREES KELVIN AND DEGREES CELSIUS
! ZR - HEIGHT OF MEASUREMENTS ABOVE WATER SURFACE (m) ASSUMED TO
! BE SCREEN HEIGHT
! OTHERS
! DELTAW - SLOPE OF THE TEMPERATURE-SATUARTION WATER VAPOUR CURVE
! AT WET BULB TEMPERATURE
! (kPa deg C-1)
! DELTAA - SLOPE OF THE TEMPERATURE-SATUARTION WATER VAPOUR CURVE
! AT AIR TEMPERATURE
! (kPa deg C-1)
! TAU - TIME CONSTANT OF THE WATER BODY (days)
! TE - EQUILIBRIUM TEMPERATURE (deg. C)
    ! WINDF - SWEER'S WIND FUNCTION
!
implicit none
integer ierr
real albedo,cp,cw,degabs,depth,deltaa,deltas,deltaw,evap,
& evappm,fc,gamma,k,lambda,le,lepm,longrad,lradj,ra,rho,rhow,rn,
& rns,sigma,solrad,sradj,ta,tau,te,tn,tstep,tw,tw0,u,ut,vpd,vpdp,
& windf,z0,zr
real alambdat,delcalc,psyconst
!
! SETUP CONSTANTS
!
lambda=alambdat(ta)
gamma=psyconst(100.0,lambda)
rhow=1000.0
cw=0.0042
rho=1.0
cp=1.013
sigma=4.9e-9
k=0.41
degabs=273.13
zr=10.0
!
! INITIALISE OUTPUT VARIABLES
!
ierr=0
deltas=0.0
evap=0.0
evappm=0.0
le=0.0
lepm=0.0
rn=0.0
tw=0.0
!
! CHECK FOR SIMPLE ERRORS
!
if (albedo.le.0.0.or.albedo.ge.1.0) then
ierr=1
return
endif
if (depth.le.0) then
ierr=2
return
endif
if (tn.gt.ta) ierr=3
if (solrad.le.0.) ierr=4
ut=u
if (ut.le.0.01) then
ierr=5
ut=0.01
endif
if (vpd.le.0.0) then
ierr=6
vpd=0.0001
endif
!
! CONVERT FROM W m-2 TO MJ m-2 d-1
!
sradj=solrad*0.0864
lradj=longrad*0.0864
!
! CONVERT FROM mbar TO kPa
!
vpdp=vpd*0.1
!
! CALCULATE THE SLOPE OF THE TEMPERATURE-SATURATION WATER VAPOUR CURVE
! AT THE WET BULB TEMPERATURE (kPa deg C-1)
!
deltaw=delcalc(tn)
!
! CALCULATE THE SLOPE OF THE TEMPERATURE-SATURATION WATER VAPOUR CURVE
! AT THE AIR TEMPERATURE (kPa deg C-1)
!
deltaa=delcalc(ta)
!
! CALCULATE THE NET RADIATION FOR THE WATER TEMPERATURE (MJ m-2 d-1)
!
lradj=fc*sigma*(ta+degabs)**4*(0.53+0.067*sqrt(vpd))
rn=sradj*(1.-albedo)+lradj-fc*(sigma*(ta+degabs)**4+
& 4.*sigma*(ta+degabs)**3*(tw0-ta))
!
! CALCULATE THE NET RADIATION WHEN THE WATER TEMPERATURE EQUALS THE
! WET BULB TEMPERATURE. ASSUMES THE EMISSIVITY OF WATER IS 1
! (MJ m-2 d-1)
!
rns=sradj*(1.-albedo)+lradj-fc*(sigma*(ta+degabs)**4+
& 4.*sigma*(ta+degabs)**3*(tn-ta))
!
! CALCULATE THE WIND FUNCTION (MJ m-2 d-1 kPa-1) USING THE METHOD OF
! Sweers, H.E., 1976, J.Hydrol., 30, 375-401, NOTE THIS IS FOR
! MEASUREMENTS FROM A LAND BASED MET. STATION AT A HEIGHT OF 10 m
! BUT WE CAN ASSUME THAT THE DIFFERENCE BETWEEN 2 m AND 10 m IS
! NEGLIGIBLE
!
windf=(4.4+1.82*ut)*0.864
!
! CALCULATE THE TIME CONSTANT (d)
!
tau=(rhow*cw*depth)/
& (4.0*sigma*(tn+degabs)**3+windf*(deltaw+gamma))
!
! CALCULATE THE EQUILIBRIUM TEMPERATURE (deg. C)
!
te=tn+rns/(4.0*sigma*(tn+degabs)**3+windf*(deltaw+gamma))
!
! CALCULATE THE TEMPERATURE OF THE WATER (deg. C)
!
tw=te+(tw0-te)*exp(-tstep/tau)
!
! CALCULATE THE CHANGE IN HEAT STORAGE (MJ m-2 d-1)
!
deltas=rhow*cw*depth*(tw-tw0)/tstep
!
! z0 - ROUGHNESS LENGTH
! DUE TO SMOOTHNESS OF THE SURFACE THE ROUGHNESS LENGTHS OF MOMENTUM
! AND WATER VAPOUR CAN BE ASSUMED TO BE THE SAME
!
z0=0.001
!
! CALCULATE THE AERODYNAMIC RESISTANCE ra (s m-1)
!
ra=alog(zr/z0)**2/(k*k*ut)
!
! CALCULATE THE PENMAN-MONTEITH EVAPORATION
!
le=((deltaa*(rn-deltas)+86.4*rho*cp*vpdp/ra)/(deltaa+
& gamma))
evap=le/lambda
!
! CONVERT THE FLUXES TO W m-2
!
rn=rn/0.0864
le=le/0.0864
deltas=deltas/0.0864
return
end
!***********************************************************************
function delcalc(ta)
!***********************************************************************
! FUNCTION TO CALCULATE THE SLOPE OF THE VAPOUR PRESSURE CURVE
! INPUT
! TA - AIR TEMPERATURE (deg. C)
! OUTPUT
! DELCALC - SLOPE OF THE VAPOUR PRESSURE CURVE (kPa deg. C-1)
!
implicit none
real delcalc,ta,ea
ea=0.611*exp(17.27*ta/(ta+237.3))
delcalc=4099*ea/(ta+237.3)**2
return
end
!***********************************************************************
function alambdat(t)
!***********************************************************************
!
! FUNCTION TO CORRECT THE LATENT HEAT OF VAPORISATION FOR TEMPERATURE
!
! INPUT:
! T = TEMPERATURE (deg. C)
! OUTPUT:
! ALAMBDAT = LATENT HEAT OF VAPORISATION (MJ kg-1)
!
implicit none
real alambdat,t
alambdat=2.501-t*2.2361e-3
return
end
!***********************************************************************
function psyconst(p,alambda)
!***********************************************************************
! FUNCTION TO CALCULATE THE PSYCHROMETRIC CONSTANT FROM ATMOSPHERIC
! PRESSURE AND LATENT HEAT OF VAPORISATION
! SEE ALLEN ET AL (1994) ICID BULL. 43(2) PP 35-92
! INPUT:
! P = ATMOSPHERIC PRESSURE (kPa)
! ALAMBDA = LATENT HEAT OF VAPORISATION (MJ kg-1)
! OUTPUT:
! PSYCONST = PSYCHROMETRIC CONSTANT (kPa deg. C-1)
!
implicit none
real psyconst,p,alambda,cp,eta
cp=1.013
eta=0.622
psyconst=(cp*p)/(eta*alambda)*1.0e-3
return
end