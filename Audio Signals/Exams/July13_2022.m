%% July 13, 2022 
clear
%% a
fs = 16000;
y_1_f = 1000;
y_1_len_sec = 1;
t = 0:1/fs:y_1_len_sec;
y_1 = 0.5 * cos(2*pi*y_1_f*t);
%% b
y_pulse = zeros(1, length(t));
y_pulse(1:100:end) = 0.7;
%% c
b = [1, -1.5, 0.4, 0.2, 0.2];
y_2 = filter(b, 1, y_pulse);
%% d
y = y_1 + y_2;
%% e
plot_len_sec = 0.1;
t_short = 0:1/fs:plot_len_sec;

figure()
plot(t_short, y(1:length(t_short)))
xlabel('time [s]'), ylabel('y')
%% f) Define the FIR filter h composed by the three samples [0.3, 0.3, 0.3].
h = [0.3, 0.3, 0.3];
%% g) Obtain the signal y_filt that consists of y filtered with h.
% num_samp = 2^nextpow2(length(y));
% num_samp = 2^(ceil(log2(length(y)+length(h)-1)))
num_samp = 2^nextpow2(length(y)+length(h)-1);
Y = fft(y, num_samp);
H = fft(h, num_samp);
Y_filt = Y.*H;
y_filt = ifft(Y_filt);
%% h
plot_len_sec = 0.025;
t_short = 0:1/fs:plot_len_sec;

figure()
plot(t_short, y_filt(1:length(t_short)))
xlabel('time [s]'), ylabel('y filt')