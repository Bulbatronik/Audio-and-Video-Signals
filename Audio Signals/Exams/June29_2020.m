%% June 29, 2020
clear
%% a
len = 2; % length in seconds
f = 440; % sinusoidal freq in Hz
fs = 8000; % sampling freq in Hz
ts = 1/fs; % sampling period in seconds
t = 0:ts:len; % time axis in second
x = sin(2*pi*f*t); % sinusoid
%% b
figure
plot(t, x)
xlabel('Time, [s]')
ylabel('x')
%% c
M = floor(0.080 * fs); % window length in sample
w1 = hann(M);
%% d
M = floor(0.080 * fs); % window length in sample
w2 = ones(M, 1);
%% e
% define the number of steps to do
H = floor(0.040 * fs); % hop length in samples
N = floor((length(x) - M)/H) + 1; % number of frames
% container for fetures
z_1 = zeros(N,1);
z_2 = zeros(N,1);

% loop over windows of x
for n=1:N
 % extract one audio window
 frame = x((n-1)*H+1 : (n-1)*H+M); % select the frame
 x_w1 = w1' .* frame; % apply Hann window
 x_w2 = w2' .* frame; % apply rectangular window

 % compute features
 z_1(n,1) = sum(abs(diff(x_w1>0)))/M;
 z_2(n,1) = sum(abs(diff(x_w2>0)))/M;
end
%% f
t_zcr = (0:1:N-1) * 10e-3;% HOP SIZE
figure()
plot(t_zcr,z_1)%T = (0:N-1).* (H/Fs); H - in samples
figure()% 
plot(t_zcr,z_2)