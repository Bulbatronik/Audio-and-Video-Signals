%% September 7, 2021
clear
%% a
fs = 16000;
t = 0:1/fs:1;
y = 0.5*sin(2*pi*2000*t);
%% b
figure
plot(t, y)
xlabel('time [s]'), ylabel('y')
%% c
Y = fft(y);
%% d
figure
plot(0:fs/(length(y)-1):fs, abs(Y))
xlabel('Frequency [Hz]'), ylabel('Y')
%% e
frame_length = 0.2 * fs;
w1 = hann(frame_length);
%% f
figure
plot((0:(length(w1)-1))./fs, w1)
xlabel('time [s]'), ylabel('w1')
%% g + h
frame_spacing = fs*10e-3;

N = floor((length(y)-frame_length)/frame_spacing)+1;
z_w = zeros(N, 1);

for i = 1:N
    frame = y(1+frame_spacing*(i-1):frame_length+frame_spacing*(i-1));
    frame = frame.*w1';
    z_w(i) = sum(abs(diff(frame>0)))/length(frame);
end
%% i
t_zcr = (0:1:N-1) * frame_spacing;% HOP SIZE
figure
plot(t_zcr, fs*z_w)
xlabel('time [s]'), ylabel('ZCR')
%% j Crossing rate is constant since in every window we have the same number of periods
% freq 2000->2000 per per sec -> 4000 croosing rate (2 per period)