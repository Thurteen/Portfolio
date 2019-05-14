close all
clear all
clc

% Parameters for Simulation

numberoforbits = 5;
SCInertia = diag([0.33,0.37,0.35]); % kg/m^2
MaxDipoleMoment = 2.0; % A*m^2
EarthMagMoment = 7.8379*10^6; % T*km^3
GeoTiltAngle = 11.44*pi/180; % rad
EarthRotation = 7.292116*10^(-5); % rad/s
SampleTime = 10; % s
StepTime = 0.1; % s
inclinationarray = linspace(0,90,4);
Altitudearray = linspace(6521,8021,3);

NumSims = length(Altitudearray)*length(inclinationarray);
SimsLeft = NumSims

% Main Program Loop
for i =1:length(Altitudearray)
        % Initial Conditions
        r = -4 + (8).*rand(3,1);
        q0 = [ 0; 0; 0; 1];
        Omega0 = [ r(1); r(2); r(3)];
    for j = 1:length(inclinationarray)
        
        % Relevant parameters for iteration
        Rorbit = Altitudearray(i); %km
        Inclination = inclinationarray(j)*pi/180; % rad
        Torbit = 2*pi*sqrt(((Rorbit*1000)^3)/(3.986*10^14)); % s
        SimTime = numberoforbits*Torbit; % s
        DipoleMag = EarthMagMoment / Rorbit^3; % T
        OrbitalSpeed = 2*pi/Torbit; % rad/s
        TotalTime = zeros(round(SimTime/SampleTime)+1,1);
        
        % Bdot Gain Calculation
        Beta1 = 0;
        CosZeta = cos(Inclination)*cos(GeoTiltAngle) + sin(Inclination)*sin(GeoTiltAngle)*cos(Beta1);
        XI = atan2(-sin(GeoTiltAngle)*sin(Beta1),sin(Inclination)*cos(GeoTiltAngle) - cos(Inclination)*sin(GeoTiltAngle)*cos(Beta1));
        if sin(XI) ==0
            SinZeta = sin(Inclination)*cos(GeoTiltAngle)-cos(Inclination)*sin(GeoTiltAngle)*cos(Beta1) / cos(XI);
        else
            SinZeta = -sin(GeoTiltAngle)*sin(Beta1)/sin(XI);
        end
        BdotGain = 2 * OrbitalSpeed * (1+SinZeta)*min(diag(SCInertia));
        
        % Generate state vectors
        YtBdot = [q0' Omega0' zeros(1,3); zeros(round(SimTime/SampleTime),10)];
        YtRate = [q0' Omega0' zeros(1,3); zeros(round(SimTime/SampleTime),10)];
        YtState = [q0' Omega0' zeros(1,3); zeros(round(SimTime/SampleTime),10)];
        
        % Solve Differential Equations
        for k=1:round(SimTime/SampleTime),
            YsBdot = ode4(@(t,Y)AttitudeDynamics(t,Y,Inclination,OrbitalSpeed,EarthRotation,DipoleMag,GeoTiltAngle,SCInertia,MaxDipoleMoment,BdotGain,StepTime,1,k),...
            TotalTime(k):StepTime:(TotalTime(k)+SampleTime),YtBdot(k,:)');
            YtBdot(k+1,:) = YsBdot(end,:);
            YsRate = ode4(@(t,Y)AttitudeDynamics(t,Y,Inclination,OrbitalSpeed,EarthRotation,DipoleMag,GeoTiltAngle,SCInertia,MaxDipoleMoment,BdotGain,StepTime,2,k),...
            TotalTime(k):StepTime:(TotalTime(k)+SampleTime),YtRate(k,:)');
            YtRate(k+1,:) = YsRate(end,:);
            YsState = ode4(@(t,Y)AttitudeDynamics(t,Y,Inclination,OrbitalSpeed,EarthRotation,DipoleMag,GeoTiltAngle,SCInertia,MaxDipoleMoment,BdotGain,StepTime,3,k),...
            TotalTime(k):StepTime:(TotalTime(k)+SampleTime),YtState(k,:)');
            YtState(k+1,:) = YsState(end,:);
            TotalTime(k+1) = TotalTime(k)+SampleTime;
        end
                
        Bdot = sprintf('%s_%d_%d.dat','Bdot',i,j);
        Rate = sprintf('%s_%d_%d.dat','Rate',i,j);
        State = sprintf('%s_%d_%d.dat','State',i,j);
      
        mout(end+1,:) = mout(end,:);
        
        BdotM = [YtBdot mout(:,1:3)];
        RateM = [YtRate mout(:,4:6)];
        StateM = [YtState mout(:,7:9)];
       
        csvwrite(Bdot,BdotM)
        csvwrite(Rate,RateM)
        csvwrite(State,StateM)
        
        SimsLeft = SimsLeft - 1
        
    end
end

13

%% Functions

% Attitude Dynamic Model
function dydt = AttitudeDynamics(t,Y,Inclination,OrbitalSpeed,EarthRotation,DipoleMag,GeoTiltAngle,SCInertia,MaxDipoleMoment,BdotGain,StepTime,ControlScheme,k)
    OrbitBodyAttTransform = (quat2dcm([Y(4) Y(1:3)']));
    Beta1 = EarthRotation*t;
    b = OrbitBodyAttTransform * MagneticField(OrbitalSpeed*t,Inclination,GeoTiltAngle,Beta1,DipoleMag);
    Omegar = Y(5:7)-OrbitBodyAttTransform*[0;-OrbitalSpeed;0];
    dydt(1:3,1) = (1/2)*(Y(4)*Omegar - cross(Omegar,Y(1:3)));
    dydt(4,1) = -(1/2)*Omegar'*Y(1:3);
    Bdot = (b-Y(8:10,1))/StepTime;
    dydt(8:10,1) = Bdot;
    % Chose Control Scheme
    if ControlScheme == 1
        ControlLaw = BdotControl(BdotGain/norm(b)^2,Bdot);
    elseif ControlScheme == 2
        ControlLaw = RateFeedbackControl(BdotGain/norm(b)^2,b,Y);
    elseif ControlScheme == 3
        ControlLaw = StateFeedbackControl(Y);
    end
    % Restrict dipole coils to saturation limit
    for j=1:length(ControlLaw)
        if abs(ControlLaw(j))>MaxDipoleMoment
            ControlLaw(j)=sign(ControlLaw(j))*MaxDipoleMoment;
        end
    end
    if ControlScheme == 3
        dydt(5:7,1) = mldivide(SCInertia,(ControlLaw-cross(Y(5:7),SCInertia*Y(5:7))));
    else
        dydt(5:7,1) = mldivide(SCInertia,(cross(ControlLaw,b)-cross(Y(5:7),SCInertia*Y(5:7))));
    end
    
    % Record dipole moments
    persistent m
    if mod(floor(t),10) == 0
        a = (ControlScheme-1)*3 + 1;
        c = ControlScheme*3;
        m(k,a:c) = ControlLaw;
        assignin('base','mout',m)
    end
end
    
function u=BdotControl(BdotGain,Bdot)
    u = -BdotGain*Bdot;
end

function u=RateFeedbackControl(BdotGain,b,Y)
    u = -(BdotGain*cross(b,Y(5:7)));
end

function u=StateFeedbackControl(Y)
     u = -0.0001*Y(5:7) - 0.0001*Y(1:3);
end

function b=MagneticField(OrbitalSpeedTime,Inclination,GeoTiltAngle,Beta1,DipoleMag)
    CosZeta = cos(Inclination)*cos(GeoTiltAngle) + sin(Inclination)*sin(GeoTiltAngle)*cos(Beta1);
    XI = atan2(-sin(GeoTiltAngle)*sin(Beta1),sin(Inclination)*cos(GeoTiltAngle) - cos(Inclination)*sin(GeoTiltAngle)*cos(Beta1));
    if sin(XI) ==0
        SinZeta = sin(Inclination)*cos(GeoTiltAngle)-cos(Inclination)*sin(GeoTiltAngle)*cos(Beta1) / cos(XI);
    else
        SinZeta = -sin(GeoTiltAngle)*sin(Beta1)/sin(XI);
    end
    b = DipoleMag * [SinZeta*cos(OrbitalSpeedTime - XI); -CosZeta; 2*SinZeta*sin(OrbitalSpeedTime - XI)];
end