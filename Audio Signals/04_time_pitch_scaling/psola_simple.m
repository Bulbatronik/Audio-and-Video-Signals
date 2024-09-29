%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PSOLA                           
% Audio Signals course
% 2021
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc, clear all, close all


%% Read audio track
% Read signal
% ...

% Convert to mono if stereo
% ...

% Plot waveform
% ...


%% Detect fundamental frequency from Fourier analysis
% Compute FFT magnitude
% ...

% Plot the spectrum
% ...

% Detect fundamental frequency of the harmonic sound
% hint: check 'max' and 'findpeaks' functions 
% ...


%% Generate analysis train of pulses
% Compute the period associated to the fundamental frequency
% ...

% Generate a train of pulses according to the computed period
% ...

% Compute pulses locations
% ...


%% Generate synthesis train of pulses
% Choose a pitch scaling factor
alpha = .9;

% Compute the new period associated to the scaling factor
% ...

% Generate a train of pulses according to the computed period
% ...

% Compute pulses locations
% ...


%% Window
% Generate a Hann window with the correct lenght for synthesis
% ...

% Check windows ovrelapping condition
% ...


%% PSOLA (simplified)
% Allocate memory for synthesis signal
% ...

% Loop over all synthesis windows
% for ...

%     Select synthesis window position
%     ...

%     Select analysis window
%     ...

%     Copy analysis window to synthesis position
%     ...

% end


%% Play original and pitch-shifted signals
% ...

