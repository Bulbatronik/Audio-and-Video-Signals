%% LPC basic utilities
clc, clear all, close all


%% Load the voice signal
[s, Fs] = audioread('voiced_a.wav');


%% Trim the signal to 512 samples
s = s(1:512);
figure(1)
plot(s)


%% Auto-correlation computation
r = xcorr(s);% BUT ZERO SAMPLE IS IN THE MIDDLE!!!
figure(2)
plot(r)


%% Apply Levinson-Durbin recursion
order = 25;
central_lag = ceil(length(r) / 2);% FROM THE CENTER 'TILL THE END
a = levinson(r(central_lag:end), order);


%% Compute frequency response
[H, w] = freqz(1, a, 512);% no zeros, only poles
figure(3)
plot(w, abs(H))


%% Compute residual (i.e., whiten the signal)
e = conv(s, a, 'same'); % Remember the difference between 'valid', 'same' and 'full'
%e = filter(a, 1, s);%zeros
figure(4)
plot(e)


%% Apply shaping filter to residual 
s_hat = filter(1, a, e);%poles
figure(5)
plot(s_hat)
hold on
plot(s)

