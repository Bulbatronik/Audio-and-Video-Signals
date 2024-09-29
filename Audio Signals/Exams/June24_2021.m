%% June 24, 2021
clear
%% a
fs = 8000;
t = 0:1/fs:1;
x = sin(2*pi*3000*t)+2*sin(2*pi*1500*t);
%% b
figure
plot(t, x)
xlabel('Time, [s]')
ylabel('x')
%% c  
T = 0:1/fs:1/3000;

figure
plot(T, x(1:length(T)))
xlabel('Time, [s]')
ylabel('x_short')
%% d
f_ax = 0:fs/(length(x)-1):fs;
figure
plot(f_ax, abs(fft(x)))
xlabel('Frequency, [Hz]')
ylabel('X')
%% e
z = fs*sum(abs(diff(x>0)))/length(x);%zerocrossrate(x);
disp(z)
%% f 
t1 = 0:1/fs:0.02;
w = ones(1, length(t1));

figure
plot(t1, w)
xlabel('Time, [s]')
ylabel('w')
%% g
x_w = w.*x(1:length(w));
%% h
f_ax = 0:fs/(length(x_w)-1):fs;

figure
plot(f_ax, abs(fft(x_w)))
xlabel('Frequency, [Hz]')
ylabel('X_W')
%% i
h = [0.5, 0.5];
y = ifft(fft(h,length(h)+length(x_w)-1).*fft(x_w,length(h)+length(x_w)-1));
%% j
f_ax = 0:fs/(length(y)-1):fs;

figure
plot(f_ax, abs(fft(y)))
xlabel('Frequency, [Hz]')
ylabel('Y')
%% k
y_t = conv(h, x_w);