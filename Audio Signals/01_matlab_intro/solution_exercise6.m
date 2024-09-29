%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Introduction to Matlab for audio processing                              
% Audio Signals course
% 2022
% Mirco Pezzoli
% Exercise 6
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
clc

%% Simulate additive noise.
% Let's simulate the acquisition of a microphone signal. The signal is
% sampled at 8kHz, and the signal to noise ratio is 5dB. The acquired
% signal is composed of two sinusoidal tone with frequency 500Hz and 1000Hz
% respectively. The signal length is equal to 5ms.
% Compute the signal samples and plot it. 


fs = 8000;                      % Sampling frequency
ts = 1/fs;                      % Sampling period
t = 0 : ts : 0.005;             % Time axis
A = 0.5;                        % Signal amplitude
f1 = 500;                       % First tone frequency
f2 = 1000;                      % Second tone frequency

phi = pi/2;
x1 = A*sin(2*pi*f1*t + phi);    % First tone
x2 = A*sin(2*pi*f2*t + phi);    % Second tone
x = x1 + x2;

SNR_dB = 5;
std_v = A / (10^(SNR_dB/20)); % std. dev. of noise

v = std_v*randn(size(x));     % white noise samples
s = x + v;                    % noisy signal

figure()
plot(t,s); axis([0 0.005 -2.5 2.5]) % plot the signal
xlabel('Time [s]');

%%  Part 2
% We repeat the measurement using an higher quality microphone with peak 
% SNR of 60dB. Simulate this second acquisition and comment the result. 

SNR_dB = 60;
std_v = A / (10^(SNR_dB/20)); % std. dev. of noise

v = std_v*randn(size(x));     % white noise samples
s = x + v;                    % noisy signal

figure()
plot(t,s); axis([0 0.005 -2.5 2.5]) % plot the signal
xlabel('Time [s]');



