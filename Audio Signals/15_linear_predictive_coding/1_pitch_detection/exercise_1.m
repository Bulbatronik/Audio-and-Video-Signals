%% Estimate the pitch of a voiced signal
clc, clear all, close all


%% Load the voice signal

% Load the signal voiced_a.wav, then try with unvoiced_sh.wav
[s, Fs] = audioread('voiced_a.wav');
% Plot the signal
figure()
plot(s)
title('Signal s')


%% Define the LPC parameters

% Set the window length to 25ms and the prediction order to 25
Ts = 1/Fs;
Time25 = 0.025/Ts;
windowlengthinsamples = round(Time25);

s = s(1:windowlengthinsamples);
figure()
plot(s)
title('Windowed s')

% Set the number N of frames to 10

N = 1;


%% Loop over all the windows
for n=0:N-1
    
    % Select window and compute r auto-correlation
    
    r = xcorr(s);

    % Plot autocorrelation
    
    figure()
    plot(r)
    title(['Autocorrelation of windowed s for N =', num2str(n)])
    
    % Compute the filter parameters using the function levinson  (see doc
    % levinson)
    
    order = 25;
    central_lag = ceil(length(r) / 2);
    a = levinson(r(central_lag:end), order);
    
    % Compute the prediction error using filter or conv
    
    error = conv(s,a);
    figure()
    plot(error)
    title(['Estimated error for N = ', num2str(n)])
    hold on
    
    % Find pitch-related peaks
    
    [pks,locs] = findpeaks(error,'MinPeakHeight',0.14);
    
    fpitch = 1/((locs(2)- locs(1))*Ts);
    %frequency_pitch is equal to the opposite of the difference in time of two pics
    
    % Compute the shaping filter H using freqz function
    
    [H, w] = freqz(1, a, 512);
    figure(3)
    plot(w, abs(H))
    title(['Shapping filter for N = ', num2str(n)])
    
    % Compute the DFT of the original signal
    
    S = fft(s);
    Sabs = abs(S);
    Snorm = Sabs/max(Sabs);
    
    faxes = -Fs/2 : Fs/length(s) : Fs/2 - 1/length(s);
    
    figure()
    plot(faxes, Snorm)
    title(['DFT of the original signal for N = ', num2str(n)])
    
    % Plot the Magnitude of the signal spectrum, and on the same graph, the
    % the LPC spectrum estimation
    
    % Plot the predition error e in time
    
    figure()
    plot(error)
    title(['Estimated error on time for N = ', num2str(n)])
    
    % Plot the prediction error magnitude spectrum
    
    E = fft(error);
    Eabs = abs(E);
    Enorm = Eabs / max(Eabs);
    
    figure()
    plot(Enorm)
    title(['Estimated error on frequency for N = ', num2str(n)])
    
    
    % Use the function pause to stop the for loop and check the plot
    %pause();
    
end

% Can you distiniguish the voiced by the unvoiced signal from one of the
% plots?
