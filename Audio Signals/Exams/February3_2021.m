%% February 3, 2021
clear
%% a
fs = 8000;
t = 0:1/fs:1;
x = 2*sin(2*pi*3000*t)+ 2*sin(2*pi*3200*t);
%% b
figure
plot(t, x)
xlabel('Time, [s]')
ylabel('x')
%% c
T = 1/3200;
t_short = 0:1/fs:T;
figure
plot(t_short, x(1:length(t_short)))
xlabel('Time, [s]')
ylabel('x')
%% d
f_ax1 = 0:fs/(length(x)-1):fs;
figure
plot(f_ax1, abs(fft(x)))
xlabel('Frequency, [Hz]')
ylabel('X')
%% e
w_len = ceil(fs*0.01)+1;% odd
w = hann(w_len);
%% f
x_w = x(1:length(w)).*w'; 
%% g 
f_ax2 = 0:fs/(length(x_w)-1):fs;
figure
plot(f_ax2, abs(fft(x_w)))
xlabel('Frequency, [Hz]')
ylabel('X_w')
%% h length is not enough to resolve 2 sinusids
%% i
zcr_x = fs*sum(abs(diff(x>0)))/length(x);
zcr_xw = fs*sum(abs(diff(x_w>0)))/length(x_w);
%% j ??
%% k
h = [0.3, 0.3, 0.3];

y1 = filter(h, 1, x); % ??
%y1 = conv(h, x);
%% l
y2 = ifft(fft(x, length(x)+length(h)-1).*fft(h, length(x)+length(h)-1));