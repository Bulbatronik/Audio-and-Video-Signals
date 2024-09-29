%% June 16, 2022
clear;
%% a
f_s = 8000;
f_1 = 1000;
t = 0:1/f_s:1;
t = t(1:8000); % making sure I'm taking 8000 samples
y_1 = 0.85 * cos(2*pi*f_1*t);
%% b
f_s = 8000;
f_1 = 1000;
%% c
t = 0:1/f_s:1;
t = t(1:8000);
y_2 = 0.7 * sin(2*pi*500*t + pi/4);
%% d
y = y_1 + y_2;
%% e
t_max_sec = 0.200;
t_max_samp = round(t_max_sec * f_s);
figure(1)
plot(t(1:t_max_samp), y(1:t_max_samp))
xlabel('time [s]')
ylabel('y')
%% f
Y = fft(y);
%% g
f = linspace(0, f_s, length(Y)); % f = 0:f_s/(length(Y)-1):f_s;
f = f(1:end-1);
figure(2)
plot(f, abs(Y))
xlabel('freq. [Hz]')
ylabel('abs(Y)')
figure(3)
plot(f, unwrap(angle(Y)))
xlabel('freq. [Hz]')
ylabel('angle(Y)')
%% h
zcr_y = f_s * sum(abs(diff(y>0))) / length(y);
zcr_y_1 = f_s * sum(abs(diff(y_1>0))) / length(y);
zcr_y_2 = f_s * sum(abs(diff(y_2>0))) / length(y);
%% i
% - y_1 has a freqeucny of 1000 Hz --> 
% 1000 periods per second --> 2000 zero crossings
% per second (2 per period)
% - The same applies to y_2