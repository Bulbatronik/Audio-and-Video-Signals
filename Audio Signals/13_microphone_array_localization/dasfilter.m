function h = dasfilter(fAx,thetaAx,d,nMic,c)
%DASFILTER This function computes the DAS filter 
%  Compute the filter coefficients based on the delay and sum filter
%  definition for all the frequencies in fAx
%
% Audio Signals course
% Mirco Pezzoli
% 2022
fLen = length(fAx);
nDoa = length(thetaAx);
h = zeros(nMic, nDoa, fLen);     % Steering Vector

for mm = 1:nMic
    for aa = 1:nDoa
        h(mm,aa,:) = exp(-1i*(2*pi*fAx/c) * (d*sin(thetaAx(aa))*(mm-1)));
    end
end
h = h ./ nMic;
end

