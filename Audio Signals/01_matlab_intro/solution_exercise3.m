%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Introduction to Matlab                            
% Audio Signals course
% 2022
% Credits to Federico Borra
% Exercise 3
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions
% Write the function
% [n, t, x] = create_sincos(duration, Fs, f, sincos)
% Inputs:
%   - duration: duration of the sinusoid in seconds
%   - Fs: sampling frequency
%   - f: frequency of the sinusoid in Hz
%   - sincos: flag: 0 create a sine, 1, create a cosine
% Returns:
%   - n: sample axis
%   - t: time axis
%   - x: the signal
function [n, t, x] = create_sincos(duration, Fs, f, sinvscos)
Ts = 1 / Fs;  % sampling period
t = 0:Ts:duration;  % temporal axis [s]

% create the signal
if sinvscos == 0
    x = sin(t*2*pi*f);  % the signal
else
    x = cos(t*2*pi*f);  % the signal
end
n = 0:(length(x)-1);  % sample axis
    
end
