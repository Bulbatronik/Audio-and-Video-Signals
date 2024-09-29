%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Source localization based on spatial filtering                           
% 
% Implement a source localization system. The signals are recorded
% by an eStick device, namely, an ULA made of 16 mics. The distance between
% two consecutive microphones is 3cm. Use a delay and sum beamformer to 
% identify the direction of arrival (DOA) of a source inspecting the 
% pseudospectrum.
% 
% Audio signals course
% Mirco Pezzoli
% 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear 
close all

%% Source Localization experiment
% Implement a source localization system. The signals are recorded
% by an eStick device, a linear array made by 16 mics. The distance between
% the microphone is 3cm. Use a delay and sum beamformer to identify the
% direction of arrival (DOA) of a source inspecting the pseudospectrum.

c = 340;            % speed of sound
nfft = 512;         % Nfft
d = 0.03;           % Microphone distance
% Load signals from the data folder
fileName = './data/class_2022.wav';
[s, fs] = audioread(fileName);
micN = size(s,2);   % Number of microphones

% Compute the STFT for each channel, use stft() if you are on MATLAB 2019a
% or later, otherwise use the provided stft function
% UNCOMMENT IF NO DEFAULT STFT
addpath('stft')
winLength = 256;
analysisWin = hamming(winLength,'periodic');
synthesisWin = hamming(winLength,'periodic');
hop = winLength / 8;

for nCh = 1:16
%     [S(:,:,nCh), fAx, tAx] = stft(s(:,nCh), fs, 'FFTLength', nfft, ...
%         'Centered', false);
% UNCOMMENT IF NO DEFAULT STFT
   [S(:,:,nCh), fAx, tAx] = stft(s(:,nCh), analysisWin, hop, nfft, fs);
end
S = S(1:(nfft/2 +1),:,:);   % Use only the positive frequency axis (if using Matlab's stft)
fAx = fAx(1:nfft/2+1);      % Frequency axis
fLen = nfft/2+1;            % Frequency axis length
tLen = length(tAx);         % Time axis length

%% Compute the DAS filter using the function dasfilter.
% Consider 30 direction uniformely sampled from -pi/2 to pi/2
% Define a function called dasfilter(fAx,thetaA,d,micN,c) that computes the
% DAS filters
nDir = 30;
thetaAx = linspace(-pi/2,pi/2,nDir);

h = dasfilter(fAx,thetaAx,d,micN,c);
% meanT = size(hatR,2);
% p = zeros(meanT,fLen,nDir);

%% Filter the microphone signals spatially 
% Use the provided function spatial filter

y = spatialfilter(S, h);

% Compute the pseudospectrum (absolute value of filter output) only for
% frequencies higher than 600Hz.
minF = 600;                             % Min frequency
[~, minFidx] = min(abs(fAx - minF));     

% Compute the pseudospectrum (absolute value of filter output) only for
% frequencies below Nyquist condition.
maxF = c/(2*d);                             % Max frequency
[~, maxFidx] = min(abs(fAx - maxF));     

% Compute the average of the pseudospectum in the frequency using geomean
% function (see doc geomean) for all the time instants
p = squeeze(geomean(abs(y(minFidx:maxFidx,:,:)).^2,1));
p = p ./ max(p,[],2);   % normalize the pseudospectrum

% Plot the average pseudospectrum
% use imagesc to plot the average pseudospectrum for all the time instants
figure
imagesc(rad2deg(thetaAx),tAx, p ./ max(p,[],2)), title('Average p')
xlabel('DOA [deg]'), ylabel('Time [s]')

%% Localize the source
% Identify the DOA in the pseudospectrum
[~, doaIdx] = max(p,[],2);

% Plot the array setup
figure
micY = zeros(1,micN);                       % x corrdinate of the mics
micX = ((0:d:(d*micN-d)) - d*micN/2)+d/2;   % y coordinate
scatter(micX, micY, 'filled')               % Use scatter to plot
title('Estimated DOA')
xlim([-1,1]), ylim([0,1]);
hold on
% Plot the doa vector every 50 frame. Use the function quiver to draw an
% arrow in the DOA direction
for tt = 1:50:tLen  
   quiver(0,0,sin(thetaAx(doaIdx(tt))),cos(thetaAx(doaIdx(tt))));
   pause(0.5)
end










