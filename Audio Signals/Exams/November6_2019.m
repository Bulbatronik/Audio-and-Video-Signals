%% November 6, 2019
clear
%% a
[x, fs] = audioread('audio.wav');
x = x(:, 1);
%% b
t = (0:length(x)-1)./ fs;
y = x + sin(2*pi*440*t);
%% c
figure(1)
plot(t, y);
xlabel('time [s]')
ylabel('y(t)')
grid('on')
%% d
win_len_sec = 0.020; % window length in seconds
win_len_samples = ceil(win_len_sec * fs); % window length in samples
win_len_samples = win_len_samples + 1; % let's use an odd window
win = hann(win_len_samples); % window
y_win = y(1:win_len_samples) .* win'; % first window of the signal
%% e
order = 20;
r = xcorr(y_win); % auto-correlation
central_lag = ceil(length(r) / 2); % right half of the auto-correlation
a = levinson(r(central_lag:end), order); % estimated filter coefficients
% note: to withen the signal y_win, we can compute e=filter(a, 1, y_win)