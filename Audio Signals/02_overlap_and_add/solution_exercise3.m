%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Overlap and add                            
% Audio Signals course
% 2022
% Mirco Pezzoli
% Exercise 3
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Implement overlap and add
% Input signal: impulse train of lenght 1000
% Filter: low-pass, Nh=33 samples
% Analysis window
%   Length M=33 samples 
%   Hop-size: 50% (M/2+1=17) 
%   Type: Hanning

clc
close all
clear
%% Define the signal and the filter
L = 1000;
x = zeros(1,L); % An impulse every 50 samples
x(1:50:L) = 1;
% Define the filter
Nh = 33;                % Filter length
h = fir1(Nh-1, 0.5);    % Compute the filter

% OLA window definition
M = Nh;                 % Window length
w = hanning(M)';        % Type of window
R = (M-1)/2+1;          % Hop size

Nfft = 2^(ceil(log2(M+Nh-1)));  % length of the FFT, next power of 2 to M+Nh-1
    
H = fft(h, Nfft);               % Filter in frequency domain

y = zeros(1, L+Nh-1);           % Output signal

nframes = floor((L-M)/R)+1;       % Number of frames in the signal

for m = 1:nframes % Sliding windows
    m_idx = (m-1)*R+1 : (m-1)*R+M;
    xm = x(m_idx) .* w;
    
    Xm = fft(xm, Nfft);         % Filter in the frequency domain with multuplication
    Ym = Xm .* H;               
    
    ym = real(ifft(Ym));        % Return to time domain
    ola_idxs = (m-1)*R+1 : (m-1)*R + (M+Nh-1);
    ym = ym(1:M+Nh-1);
    y(ola_idxs) = y(ola_idxs) + ym;         % Put everything back
end

%% Plot the input and the output of the filtering operation using subplot

figure
subplot(2,1,1)
plot(x)
title('Input');

subplot(2,1,2)
plot(y)
title('OLA output');



















