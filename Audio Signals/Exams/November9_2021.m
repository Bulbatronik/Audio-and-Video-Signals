%% Novenber 9, 2021
clear
%% a
fs = 8000;
t = 0:1/fs:2;
y = 0.8*sin(2*pi*2000*t);
%% b
figure
plot(t, y);
xlabel('Time, [s]')
ylabel('y')
%% c
Y = fft(y);
%% d
figure
plot(0:fs/(length(y)-1):fs, abs(Y));
xlabel('Frequency, [Hz]')
ylabel('Y')
%% e
w_len = ceil(0.4*fs);
w_len = w_len + 1;% odd
w = hann(w_len);
%% f
delta = ceil(0.8*fs);
y_w = y(delta+1:delta+length(w)).*w';
%% g
h = [1, 0 ,-1];
%% h
y_f_t = conv(y_w, h);
%% i
N = length(y_w)+length(h)-1;  %the minimum pad factor
h1 = [h, zeros(1, N-length(h))];
y_w1 = [y_w, zeros(1, N-length(y_w))];
y_f_f = ifft(fft(y_w1).*fft(h1));
%% check i
figure
subplot(2, 1, 1)
plot((0:fs/(length(y_f_t)-1):fs), abs(fft(y_f_t)));
xlabel('Frequency, [Hz]')
ylabel('Y (time)')

subplot(2,1,2);
plot((0:fs/(length(y_f_f)-1):fs), abs(fft(y_f_f)));
xlabel('Frequency, [Hz]')
ylabel('Y (freq)')