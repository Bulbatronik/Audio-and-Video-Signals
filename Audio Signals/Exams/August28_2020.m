%% August 28, 2020
clear
%% a
fs = 4000;
t = 0:1/fs:1;% one extra sample
x0 = sin(2*pi*200*t);
x1 = sin(2*pi*1000*t);
x2 = sin(2*pi*1200*t);
%% b
x = x0 + x1 + x2;
%% c
X0 = abs(fft(x0));
X1 = abs(fft(x1));
X2 = abs(fft(x2));
X = abs(fft(x));
%% d
f_ax = 0:fs/(length(x)-1):fs;

subplot(4,2, 1)
plot(t, x0)
xlabel('Time, [s]')
ylabel('x_0')

subplot(4,2, 2)
plot(f_ax, X0)
xlabel('Frequency, [Hz]')
ylabel('X_0')

subplot(4,2, 3)
plot(t, x1)
xlabel('Time, [s]')
ylabel('x_1')

subplot(4,2, 4)
plot(f_ax, X1)
xlabel('Frequency, [Hz]')
ylabel('X_1')

subplot(4,2, 5)
plot(t, x2)
xlabel('Time, [s]')
ylabel('x_2')

subplot(4,2, 6)
plot(f_ax, X2)
xlabel('Frequency, [Hz]')
ylabel('X_2')

subplot(4,2, 7)
plot(t, x)
xlabel('Time, [s]')
ylabel('x')

subplot(4,2, 8)
plot(f_ax, X)
xlabel('Frequency, [Hz]')
ylabel('X')
%% e
win_len = 0.25*fs - 1;
w = hann(win_len);
%% f
x_w = x(1:win_len).*w';
%% g
h = [1, 1];
y_time = conv(h, x_w);
%% h
H = fft(h, length(h)+length(x_w)-1);
X_w = fft(x_w,length(h)+length(x_w)-1);
y_freq = ifft(H.*X_w);
%% i
t_y = 0:1/fs:(length(y_freq)-1)/fs;

subplot(1,2,1)
plot(t_y, y_freq)
xlabel('Time, [s]')
ylabel('y_freq')

subplot(1,2,2)
plot(t_y, y_time)
xlabel('Time, [s]')
ylabel('y_time')
%% j
f_ax_y = 0:fs/(length(y_freq)-1):fs;

subplot(1,2,1)
plot(f_ax_y, abs(fft(y_freq)))
xlabel('Frequency, [Hz]')
ylabel('Y_freq')

subplot(1,2, 2)
plot(f_ax_y, abs(fft(y_time)))
xlabel('Frequency, [Hz]')
ylabel('Y_time')