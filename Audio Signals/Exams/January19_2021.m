%% January 19, 2021
clear
close all
%% a
Fs = 22050; %Hz
t = 2; % sec
A = 1; % amplitude
delta = 150;

L = Fs*t; % #samples
x = zeros(1,L); % An impulse every delta samples
x(1:delta:L) = A;
%% b
a = [1, -1.5473, 0.4311, 0.2105, 0.2012];
b = [1];
y = filter(b, a, x);
%% c
figure()
plot((0:2000-1)./Fs, y(1:2000))
xlabel('Time [s]'), ylabel('Magnitude');
%% d
windowSize = 0.04 * Fs; %samples
w = ones(1, windowSize);

figure()
plot((0:length(w)-1)./Fs, w)
xlabel('Time [s]'), ylabel('Magnitude');
%% e
x_w = x(1:length(w)).*w;

figure()
plot((0:length(x_w)-1)./Fs, x_w)
xlabel('Time [s]'), ylabel('Magnitude');
%% f
order = 20;

r = xcorr(x_w);
central_lag = ceil(length(r)/2);

a = levinson(r(central_lag:end), order);% whitening filter
e = conv(x_w, a); % residual
%% g
findpeaks(e, 'Threshold',10e-5)
[pks, locs]= findpeaks(e, 'Threshold',10e-5)
ti = 0:1/Fs:2;
%% h
F = mean(abs(diff(locs)))*Fs/length(ti) % Hz
%% i smaller order -> worse performsnce. Up to 4, because of H