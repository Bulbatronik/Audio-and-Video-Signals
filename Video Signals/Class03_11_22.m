clear, clc, close all
i = imread('cameraman.tif');
figure
imshow(i)% gray value intesities
%%
I = fft2(i);% complex values
% sometime represent just with the positive freq.
% ASSUMPTION - PERIODIC IN SPACE DOMAIN
figure
surfc(abs(I))
Is = fftshift(I);
figure
surfc(abs(Is))
figure
surfc(log(abs(Is)))
% almost vertical and horiz lines in the image -> energy dispersion
%% filter with zeros at the borders
mask = zeros(256,256);
Is(129,129)%highest value (origin)
mask(129-10:129+10,129-10:129+10)=1;
figure
surfc(mask)
%% aply it
Isf = Is.*mask;
figure
surfc(log(abs(Isf)))
%% back to space domain
If = ifftshift(Isf);
ifilt = ifft2(If);
ifilt=(ifilt-min(ifilt(:)))/(max(ifilt(:))-min(ifilt(:)));% rescale between 0 and 1
figure
imshow(ifilt)% low passed version