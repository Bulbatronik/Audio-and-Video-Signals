%% November 5, 2022
clc,clear,close all
load('gong.mat')
%% a
y_trimmed = y(1:Fs*2)';
%% b
figure
plot( 0:1/Fs:(2-1/Fs),y_trimmed)
xlabel('Time, [s]')
ylabel('y_trimmed')
%% c
Y_trimmed = fft(y_trimmed);
figure
plot( 0:Fs/(length(y_trimmed)-1):Fs, abs(Y_trimmed))
xlabel('Frequency, [Hz]')
ylabel('Y_trimmed')
%% d
p = 5;
r = xcorr(y_trimmed);
central_lag = ceil(length(r)/2);
a = levinson(r(central_lag:end),p);
%% e
T = 1/100;
step = floor(T*Fs);
x = zeros(1,Fs*1);
x(1:step:end)=1;
%% f
%a
M = round(0.2*Fs);
w = ones(1,M);
rec = zeros(1,M+length(x)-1);%conv length
%b
R = 0.5*length(w);
N = floor((length(x)-M)/R)+1;
for i=1:N
    %c
    ola_ind = (i-1)*R+1:(i-1)*R+M;
    xm = x(ola_ind).*w;
    %d
    rec_m = filter(1, a, xm);
    %e
    rec(ola_ind) = rec(ola_ind)+rec_m;
end
%% g
figure
plot( 0:Fs/(length(rec)-1):Fs, abs(fft(rec)))
xlabel('Frequency, [Hz]')
ylabel('Y_trimmed')

figure
plot( 0:Fs/(length(rec)-1):Fs, angle(fft(rec)))
xlabel('Frequency, [Hz]')
ylabel('Y_trimmed')


%% Prof's sol
% clc, clear all, close all
% load('gong.mat')
% %% a)
% y_len_sec = 2;
% y_len_samp = round(y_len_sec * Fs);
% y = y(1:y_len_samp);
% %% b)
% t = (0:1:y_len_samp-1) / Fs;
% plot(t, y)
% %% c)
% f = ((0:1:y_len_samp-1) / y_len_samp) * Fs;
% Y = abs(fft(y));
% plot(f, Y)
% %% d)
% r = xcorr(y);
% order = 5;
% central_lag = ceil(length(r) / 2);
% a = levinson(r(central_lag:end), order);
% %% e)
% x_len_sec = 1;
% x_len_samp = round(x_len_sec * Fs);
% x = zeros(1, x_len_samp);
% f0 = 100;
% T0 = 1/f0;
% T0_samp = round(T0 * Fs);
% for i = 1:T0_samp:length(x)
%  x(i) = 1;
% end
% %% f.a)
% win_len_sec = 0.200;
% win_len_samp = round(win_len_sec * Fs);
% win = ones(1, win_len_samp);
% %% f.b)
% hop_size_samp = win_len_samp/2; % 50% overlap, could have been different
% %% f.c/d/e)
% num_win = floor((x_len_samp - win_len_samp) / hop_size_samp) + 1;
% y_out = zeros(1, x_len_samp);
% for i = 1:num_win
%  %% f.c)
%  idx_start = (i-1)*hop_size_samp + 1;
%  idx_end = idx_start + win_len_samp - 1;
%  frame = x(idx_start:idx_end);
%  windowed_frame = frame.*win / 2; % dividing by 2 for COLA
% 
%  %% f.d)
%  filt_frame = filter(1, a, windowed_frame);
% 
%  %% f.e)
%  y_out(idx_start:idx_end) = y_out(idx_start:idx_end) + filt_frame;
% 
% end
% %% g)
% f = ((0:1:length(y_out)-1) / length(y_out)) * Fs;
% Y_out = fft(y_out);
% figure()
% plot(f, abs(Y_out))
% figure()
% plot(f, angle(Y_out))