%% July 22, 2021
clear
%% a
fs = 16000;
t = 0:1/fs:3;
f = 2000;
A1= 0.5;
y = A1*sin(2*pi*f*t);
%% b
figure
plot(t, y)
xlabel('time [s]'), ylabel('y')
%% c
A2 = 1;
e = zeros(1,length(t));
e(1:150:end) = 1;
%% d
b = [1, -1.5, 0.4, 0.2, 0.2];
x = filter(b, 1, e);
%% e
figure
time_short = (0:2000-1)./fs;
plot(time_short, x(1:2000));
hold on;
plot(time_short, e(1:2000));
xlabel('time [s]'), ylabel('x, e')
legend('x','e')
%% f + g
x_w = x(1:0.04*fs);

r = xcorr(x_w);
central_lag = ceil(length(r)/2);
a = levinson(r(central_lag:end),5);
err = conv(x_w, a, 'same');
%% h
figure
plot(0:1/fs:(0.04-1/fs),err)
xlabel('time [s]'), ylabel('residual')

[val, pos]=findpeaks(err, "MinPeakHeight",0.5*abs(max(err)));
pitch = mean(abs(diff(pos)))*fs/length(x_w)
