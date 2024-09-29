%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Overlap and add                            
% Audio Signals course
% 2022
% Mirco Pezzoli
% Exercise 1
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear
close all

%% FIR filtering example
% Generate a signal with a sawtooth and square wave. The sampling 
% frequency is equal to 8kHz and the fundamental frequency is 500Hz and 
% 1kHz. The signal length is 0.06s. 

fs = 8000;         % Sample frequency [Hz]
ts = 1/fs;          % Sample period [s]
t = 0:ts:0.06;     % Time axis [s]
f1 = 500;           % Frequency of the signals [Hz]
f2 = 1000;
xSawtooth = sawtooth(2*pi*f1*t);
xSquare = square(2*pi*f2*t);

% Plot the signals
figure()
subplot(2,1,1)
plot(t, xSquare)
xlabel('Time [s]'), title('Square wave')
subplot(2,1,2)
plot(t, xSawtooth)
xlabel('Time[s]'), title('Sawtooth wave');

% Design an ideal low pass filter with cut off frequency 1.5kHz. The filter
% length in samples is 512. Smooth the ideal filter with an hamming window.
% (hint: doc hamming)

L = 512; % filter length
fc = 1500; % cutoff frequency
% Design the filter using the window method
nfilt = (-(L-1)/2:(L-1)/2); % We center the sinc in the vector
hideal = sin(2*pi*fc*nfilt/fs) ./ (pi*nfilt);
h = hamming(L)'.* hideal; % Smooth the sinc with hamming window

% Plot the filter in time and the magnitude of its spectrum in one window

figure();
subplot(2,1,1);
plot(h);
axis tight
xlabel('Time (samples)');
ylabel('Amplitude');
title('1500 Hz Low-pass filter (Hanning Window) - L = 512');

subplot(2,1,2);
f_ax = linspace(-fs/2,fs/2,fs);
plot(f_ax, 20*log10(abs(fftshift(fft(h,fs)))));
axis tight
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');

% Filter the sawtooth and the square with the filters in time domain and
% plot the output of the filtering

ySawtooth = conv(xSawtooth, h);
ySquare = conv(xSquare, h);

% or use filter function by Matlab
ySawtooth = filter(h, 1, xSawtooth);
ySquare = filter(h, 1, xSquare);

% Plot the results 
figure()
subplot(2,1,1)
plot(ySawtooth)
title('Filtered sawtooth')
subplot(2,1,2)
plot(ySquare)
title('Filtered square')

% Comment the results, what do you expect to see in the spectrum of the
% output signals?







