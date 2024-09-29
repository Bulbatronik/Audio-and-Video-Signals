%% February 19, 2020
clear
%% a
[x, fs] = audioread('audio.wav');
x = x(:, 1);
%% b
t = (0:length(x))/fs;

figure
plot(t, x);
xlabel('time [s]')
ylabel('x')
%% c
frame_length = floor(40e-3 * fs);
w = hann(frame_length);
%% d + e
frame_spacing = floor(10e-3 * Fs);

N = floor((length(x)-frame_length)/frame_spacing) + 1;

par = zeros(1,N);
for i=1:N

    frame = x(1+(i-1)*frame_spacing:frame_length+(i-1)*frame_spacing);
    x_w = frame.*w;   
        
    par(i,1) = sum(abs(diff(frame>0))) / frame_length;%(x_w);
    par(i,2) = sum(frame.^2);
end

figure(1), hold on
plot(par(:, 1), par(:, 2), '*g')
xlabel('ZCR')
ylabel('Power')





