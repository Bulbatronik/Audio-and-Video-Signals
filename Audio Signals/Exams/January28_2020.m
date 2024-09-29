%% January 28, 2020
clear
%% a
[x, fs] = audioread('audio.wav');
x = x(:, 1);
%% b
h = [-0.5; 0; 0.5];
%% c
win_len_sec = 0.020; % window length in seconds
win_len_samples = ceil(win_len_sec * fs); % window length in samples
win_len_samples = win_len_samples + 1; % let's use an odd window
win = hann(win_len_samples);
%% d
% a. 
hop_size = ceil(win_len_samples / 2) - 1; % Hann window with 50% overlap
x_len = length(x);
y = zeros(x_len, 1); % Output signal
nframes = floor((x_len - win_len_samples) / hop_size); % Number of frames in the signal

for m = 1:nframes % Sliding windows
    m_idx = (m-1) * hop_size + 1 : (m-1) * hop_size + win_len_samples; % Select correct samples
    x_win = x(m_idx) .* win; % Apply window
    
    % b. [2] Apply the filter h in either the frequency or time domain
    %%% Option 1: Freq. domain
    H = fft(h, win_len_samples); % Turn filter into freq. domain (can be done once outside of the loop)
    X_win = fft(x_win, win_len_samples); % Turn the windowed signal in the freq. domain
    Y_win = X_win .* H; % Convolve in time = Multiply in freq.
    
    y_win = real(ifft(Y_win)); % Go back in time domain

    %%% Option 2: Time domain
    % y_win = filter(h, 1, x_win);

    % c. [2] Re-assemble the windows in the correct way
    % y(m_idx) = y(m_idx) + y_win; 
    ola_idxs = (m-1)*hop_size+1 : (m-1)*hop_size + (x_len+win_len_samples-1);
    y_win = y_win(1:win_len_samples+length(h)-1);
    y(ola_idxs) = y(ola_idxs) + y_win; % Put everything back
end
%% e
t = (0:length(y)-1) / fs;
figure(1)
plot(t, y)
xlabel('time [s]')
ylabel('y(t)')
grid('on')