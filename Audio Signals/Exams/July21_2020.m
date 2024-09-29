%% July 21, 2020
clear
%% a
T = 1;
f0 = 800;
fs = 4000;
ts = 1/fs;
t = 0:ts:T; % extra sample
x = sin(2*pi*f0*t);
%% b
figure(1)
plot(t, x)
xlabel('time [s]')
ylabel('x(s)')
%% c
sound(x, fs);
%% d
w_len = round(0.080 * fs);
w = ones(1, w_len);
%% e
x_w = w .* x(1:w_len);
%% f
mag = abs(fft(x));
mag_w = abs(fft(x_w));
%% g
ph = angle(fft(x));
ph_w = angle(fft(x_w));
%% h
f_axis = (0:length(mag)-1) / length(mag) * fs;
figure(2)
plot(f_axis, mag)
xlabel('f [Hz]')
ylabel('mag(f)')
%% i
figure(3)
f_axis = (0:length(mag_w)-1) / length(mag_w) * fs;
plot(f_axis, mag_w)
xlabel('f [Hz]')
ylabel('mag_w(f)')