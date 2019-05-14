close all;
clear all;
clc;

% Interplanetary Transfer Calculations

format long

% variables

phaseangle = 60;
mue = 3.986004418*10^14; % m^3 / s^2
mus = 1.3288930830720663 * 10^20; % m^3 / s^2
musat = 3.7931187 * 10^16; % m^3 / s^2
rpo = 60000000; % m
rpo2 = 10^8;
aearth = 1.514652576193799 * 10^11; % m
asaturn = 1.427587767049750 * 10^12; % m
currentphase = 269.5017-74.8029;
eearth = 0.02214170359913805;
esaturn = 0.05410408096766922;

% Phasing Manouever

aearthphasing = (mue * (sqrt(rpo^3 / mue) * (1 - (phaseangle / 360)))^2)^(1/3);
eearthphasing = (rpo / aearthphasing) -1;
deltav1 = (sqrt(mue*((2 / rpo) - (1/aearthphasing))) - sqrt(mue/rpo));
ToF1 = (1-(60/360))*2*pi*sqrt(rpo^3 / mue) / 60 / 60;

% Inclination

vpark = sqrt(mue / rpo);
deltav3 = 2*vpark*sin(pi / 12);

% Hohmann Transfer

% Transfer Orbit

%%%%%%%%%%%%%%%%%%%%

satinc = 2.487 * pi / 180;

theta1 = 45.434* pi / 180;
theta2 = 248.504*pi/180;
satapta1 = (339.12 + theta2) * pi /180;

theta4 = 249.59* pi / 180;
satapta2 = (338.55 + theta4) * pi /180;

%%%%%%%%%%%%%%%%%%%%

rearth = aearth*(1 - eearth^2) / (1 + eearth*cos(theta1));

rsaturn1 = asaturn*(1 - esaturn^2) / (1 + esaturn*cos(theta2));
rsaturnadj1 = rsaturn1*cos(satinc*sin(satapta1));
sath1 = rsaturn1 * sin(satinc*sin(satapta1));

aht2 = (rearth + rsaturn1) / 2
ToFHT2 = pi * sqrt(aht2^3 / mus) /3600 / 24
eHT = 1 - (rearth / aht2)

rendurance = (aht2*(1 - eHT^2)) / (1 + eHT * cos(pi));
rsaturn2 =  asaturn*(1 - esaturn^2) / (1 + esaturn*cos(theta4));
rsaturnadj2 = rsaturn2*cos(satinc*sin(satapta2));
sath2 = rsaturn2 * sin(satinc*sin(satapta2));

tablevalues = [rearth;rsaturn1;sath1;aht2;ToFHT2;eHT;rendurance,;rsaturn2;rsaturnadj2;sath2];

rownames = {'rearth';'rsaturn1';'sath1';'aht2';'ToFHT2';'eHT';'rendurance';'rsaturn2';'rsaturnadj2';'sath2'};

T = table(tablevalues,'RowNames',rownames);

%2017-Dec-07 00:00
%2018-Mar-06 00:00
%2018-mar-02 12:00
%2024-may-09 21:30

T;

% departure trajectory

vperitrans = sqrt(mus*((2 / rearth) - (1/aht2)));
vearth = sqrt(mus*((2 / rearth) - (1/aearth)));
vpark = sqrt(mue/rpo);
v4 = sqrt(mue*((2 / rpo) + ((vperitrans- vearth)^2)/mue)) - vpark;

% phihyp = (pi + acos(1/(1+(rpo/mue)*(vperitrans-vearth)^2)))*180 / pi;