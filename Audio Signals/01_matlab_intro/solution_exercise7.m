%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Introduction to Matlab for audio processing                              
% Audio Signals course
% 2022
% Mirco Pezzoli
% Exercise 7
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear 
close all
clc

%% Frequency analysis
% We want to insepct the frequency content of a signal. The signal is
% composed of a square wave at 100Hz. 
% Choose a proper sampling frequency, and compute teh signal with a 
% length of 0.05s. Plot the signal in time domain.
fs = 8000;                      % Sampling frequency
ts = 1/fs;                      % Sampling period
t = 0 : ts : 0.05;              % Time axis
A = 0.5;                        % signal amplitude
f = 100;                        % Square tone frequency

signalLen = length(t);
xSquare = A * square(2*pi*f*t);


%% Define in a new variable the same signal corrupted by a white noise with 
% SNR equal to 10dB. 
SNR_dB = 10;
std_v = A / (10^(SNR_dB/20)); % std. dev. of noise

v = std_v*randn(size(xSquare));     % white noise samples
xSquareNoise = xSquare + v;                    % noisy signal


%% Compute the DFT of the signal
% We examine the frequency spectrum of the signal, plot the magnitude and
% the phase of the spectrum in one figure for each signal 
fAx = 0:fs/(signalLen-1):fs; % Frequency axis
X = fft(xSquare);
Xnoise = fft(xSquareNoise);

figure()
subplot(2,1,1)
plot(fAx, abs(X));
xlabel('Frequency [Hz]'), ylabel('Magnitude')
subplot(2,1,2)
plot(fAx, unwrap(angle(X)));
xlabel('Frequency [Hz]'), ylabel('Phase')

figure()
subplot(2,1,1)
plot(fAx, abs(Xnoise));
xlabel('Frequency [Hz]'), ylabel('Magnitude')
subplot(2,1,2)
plot(fAx, unwrap(angle(Xnoise)));
xlabel('Frequency [Hz]'), ylabel('Phase')

% What are the differences between the two spectrum? What is the effect of
% noise. Can we identify a structure in the magnitude spectrum?

%% Now analyze the signal of a sine wave at 1kHz.
% Compute the DFT and inspect the spectrum. How many harmonics are
% presents?

f = 1000;                      % Square tone frequency
x = A * sin(2*pi*f*t);
X = fft(x);
figure()
plot(fAx, abs(X));
xlabel('Frequency [Hz]'), ylabel('Magnitude')
title('Sinusoidal wave at 1kHz')
% Change the sampling frequency to 2000Hz. What would you expect to see in
% the spectrum?

fs = 2000;                      % Sampling frequency
ts = 1/fs;                      % Sampling period
t = 0 : ts : 0.05;               % Time axis
signalLen = length(t);
fAx = 0:fs/(signalLen-1):fs; % Frequency axis

xDownsampled = A * sin(2*pi*f*t);
Xdownsampled = fft(xDownsampled);
figure()
plot(fAx, abs(Xdownsampled));
xlabel('Frequency [Hz]'), ylabel('Magnitude')
title('Aliased signal')
