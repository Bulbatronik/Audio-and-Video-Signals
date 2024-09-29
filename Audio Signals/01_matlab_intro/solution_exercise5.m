%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Introduction to Matlab                            
% Audio Signals course
% 2022
% Credits to Federico Borra
% Exercise 5
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Aliasing
% Write a script that
%   - Simulate 4 "continuous" cosine signals
%   - Sampling period = 0.001
%   - Duration = 5 s
%   - Frequencies:
%       f1 = 0.2Hz
%       f2 =0.2+1/DT Hz
%       f3 =0.2+2/DT Hz
%       f4 =0.2+3/DT Hz
%   - DT=0.5s
% Sample the "continuous signals
%   - Sampling period = 0.5 s 
% Plot
%   Make a figure with 4 sub-plots containing the "continuous" and the 
%   relative sampled signals
%% Parameters

duration = 5;   % Duration of the signals
dt=.001;        % "small" sampling period to simulate continuous signals
Fs_c = 1/dt;     

DT=.5;          % Sampling period         
Fs = 1/DT;      % Sampling frequency

f1 = 0.2;       
f2 = 0.2+1/DT;
f3 = 0.2+2/DT;
f4 = 0.2+3/DT;

%% "Continuous" signals

[~,t,x1] = create_sincos(duration, Fs_c, f1, 1);
[~,~,x2] = create_sincos(duration, Fs_c, f2, 1);
[~,~,x3] = create_sincos(duration, Fs_c, f3, 1);
[~,~,x4] = create_sincos(duration, Fs_c, f4, 1);

%% Sampled signals

[~,T,xn_1] = create_sincos(duration, Fs, f1, 1);
[~,~,xn_2] = create_sincos(duration, Fs, f2, 1);
[~,~,xn_3] = create_sincos(duration, Fs, f3, 1);
[~,~,xn_4] = create_sincos(duration, Fs, f4, 1);

%% Plot

figure(1)
subplot(2,2,1)
plot(t,x1); hold on
stem(T,xn_1,'filled')
title('Frequency of the cosine = 1/5 Hz')
xlabel('time')
grid on
axis([0 5 -1.5 1.5])

subplot(2,2,2)
plot(t,x2,'r'); hold on
stem(T,xn_2,'filled')
title('Frequency of the cosine = 1/5 + 1/DT Hz')
xlabel('time')
grid on
axis([0 5 -1.5 1.5])

subplot(2,2,3)
plot(t,x3,'r'); hold on
stem(T,xn_3,'filled')
title('Frequency of the cosine = 1/5 + 2/DT Hz')
xlabel('time')
grid on
axis([0 5 -1.5 1.5])

subplot(2,2,4)
plot(t,x4,'r'); hold on
stem(T,xn_4,'filled')
title('Frequency of the cosine = 1/5 + 3/DT Hz')
xlabel('time')
grid on
axis([0 5 -1.5 1.5])


