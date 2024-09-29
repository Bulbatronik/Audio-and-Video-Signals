%% February 15, 2022
clear
%% a
fs = 8000;
t = 0:1/fs:1;
A1= 0.7;
f1 = 1000;

y_1 = A1*cos(2*pi*f1*t);
%% b
T = 0.1;
f = 1/T;
A2 = 0.5;

y_2 = A2*square(2*pi*f*t, 50);
%% c
y = y_1 + y_2;
%% d
figure
short_time = 0:1/fs:0.2;
plot(short_time, y(1:length(short_time)))
xlabel('time [s]'), ylabel('x, e')
%% e
Y = fft(y);
%% f
figure
f_x = 0:fs/(length(y)-1):fs;
plot(f_x, abs(Y));
xlabel('Frequency [Hz]'), ylabel('Magnitude')

figure 
plot(f_x, unwrap(angle(Y)));
xlabel('Frequency [Hz]'), ylabel('Phase')
%% g
h = [0.5, 1, -0.5];

Nfft = 2^nextpow2(length(y)+length(h)-1);
Y = fft(y, Nfft);
H = fft(h, Nfft);
Y_filt = Y.*H;
y_filt = ifft(Y_filt);