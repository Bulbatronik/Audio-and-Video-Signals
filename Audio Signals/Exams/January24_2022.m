%% January 24, 2022
clear
%% a
fs = 16000;
t = 0:1/fs:2;
y_1 = 0.7*sin(2*pi*2000*t);
%% b + c
b = [1, -1.5, 0.4, 0.2, 0.2];
train = zeros(1, 1*fs);
train(1:160:end) = 1;
y_2 = filter(b, 1, train);
pad = zeros(1,length(y_1)-length(y_2));
y_2 = [y_2, pad];
%% d 
y = y_1 + y_2;

figure
plot(t, y)
xlabel('Time, [s]')
ylabel('y [s]')
%% e
t_short = 0:1/fs:(0.025-1/fs);

figure
plot(t_short, y(1:length(t_short)))
xlabel('Time, [s]')
ylabel('y [0.025 s]')
%% f
h = [0.3, 0.3, 0.3];
%% g
Nfft = 2^nextpow2(length(h)+length(y)-1);
y_filt = ifft(fft(h, Nfft).*fft(y, Nfft));
%% h
t_y_filt = 0:1/fs:(0.025-1/fs);

figure
plot(t_y_filt, y_filt(1:length(t_y_filt)))
xlabel('Time, [s]')
ylabel('y_filt [0.025 s]')