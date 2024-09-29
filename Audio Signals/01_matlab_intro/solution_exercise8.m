%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Introduction to Matlab for audio processing                              
% Audio Signals course
% 2022
% Mirco Pezzoli
% Exercise 8
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Filtering
%  Define a function that implement a moving averaging filter with
%  arbitrary window length y = movingaverage(x, winLen)
% The definition of moving average is y(n) = (1/winLen) (x(n) + x(n-1) +
% + ... + x(n-winLen-1))
% 
function y = movingaverage(x, winlen)
    b = (1/winlen) * ones(1,winlen);
    a = 1;
    y= filter(b,a,x);
end