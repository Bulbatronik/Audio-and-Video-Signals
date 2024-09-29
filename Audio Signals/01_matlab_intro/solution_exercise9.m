%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Introduction to Matlab for audio processing                              
% Audio Signals course
% 2022
% Mirco Pezzoli
% Exercise 9
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Convolution
% load the signal guitar.wav and extract the right channel in s. 
% Define a filter h corresponding to an impulse at time 0.5s. Convolve the 
% signal with h and play the output. Can you hear any difference with 
% respect to the original signal?          
% Plot the filtered signal over the original one, what can you notice?

[s, fs] = audioread('guitar.wav');
s = s(:,2); % right channel
nSample = length(s);

% Define the filter with length 1s
h = zeros(1,fs);        % Filter definition
h(fs/2) = 1;            % Impulse at 0.5 seconds

y = conv(s,h);
disp(length(y))         % nSamples + fs -1
soundsc(y, fs)

figure()
plot((0:nSample-1)./fs, s), hold on
plot((0:length(y)-1)./fs, y)
xlabel('Time [s]');

% Apply the moving average filter to the signal. Set the window size to 20ms
% Plot the signal, what can you notice? Can you percieve any difference
% listening to the filter output?

windowSize = 0.02*fs;
b = (1/windowSize) * ones(1,windowSize);
a = 1;
y = filter(b,a,s); 
soundsc(y, fs)

figure()
plot((0:nSample-1)./fs, s), hold on
plot((0:length(y)-1)./fs, y)
xlabel('Time [s]');
