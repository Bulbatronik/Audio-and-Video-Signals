%% Example for cross synthesis of two sounds
clc, clear all, close all


%% Load data


%% Define Hanning windowing parameters


%% Define LPC parameters


%% Trim audio samples to the same lenght and normalize amplitude


%% Initialize output vector


%% Loop over each windoe and perform cross-synthesis
for j=1:N_frames
    
    % Select analysis windows and perform auto correlation
    
    
    % Estiamte LPC filter coefficients
   
    
    % Compute LPC residuals
    
    
    % Synthesize output signal and write into buffer

end


%% Normalize output signal amplitude


%% Listen to the result

