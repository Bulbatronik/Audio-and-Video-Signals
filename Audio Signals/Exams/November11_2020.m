%% November 11, 2020
clear
%% a
x_1_f = 200;
x_2_f = 3000;
fs = 8000;
len_sec = 1;
t = 0:1/fs:len_sec;
x_1 = sin(2*pi*x_1_f*t);
x_2 = sin(2*pi*x_2_f*t+pi);
%% b
x = x_1+x_2;
%% c
nSample = length(x_1);

figure
subplot(3,2,1)
plot(t, x_1), hold on
xlabel('Time [s]');
title('x1');

subplot(3,2,3)
plot(t, x_2), hold on
xlabel('Time [s]');
title('x2');

subplot(3,2,5)
plot(t, x), hold on
xlabel('Time [s]');
title('x');

f_x = 0:fs/(nSample-1):fs;

subplot(3,2,2)
X1 = abs(fft(x_1));
plot(f_x, X1), hold on
xlabel('Frequency [Hz]'), ylabel('Magnitude')
title('X1');

subplot(3,2,4)
X2 = abs(fft(x_2));
plot(f_x, X2), hold on
xlabel('Frequency [Hz]'), ylabel('Magnitude')
title('X2');

subplot(3,2,6)
X = abs(fft(x));
plot(f_x, X), hold on
xlabel('Frequency [Hz]'), ylabel('Magnitude')
title('X');
%% d
w_len = 0.4*fs;
w = hann(w_len);

figure
plot((0:length(w)-1)./fs, w)
xlabel('Time [s]');
title('w');
%% e
x_w = x(1:length(w)).*w';
%% f
h = [0.5, 0.5];
% num_samp = 2^nextpow2(length(x_w)+length(h)-1);
y_w = conv(x_w, h);
%% g
Y_w = abs(fft(y_w));
f_y = 0:fs/(length(y_w)-1):fs;
%f_y = (0:length(y_w)-1).*fs;

figure
plot(f_y, Y_w), hold on
xlabel('Frequency [Hz]'), ylabel('Magnitude')
title('Y_w');
%% h
% check untill fn=fs/2
figure
eps = 10^-5;
findpeaks(Y_w(1:floor(length(f_y)/2)), f_y(1:floor(length(f_y)/2)), 'Threshold', eps)
%% i
zcr = sum(abs(diff(y_w>0)))/length(y_w);