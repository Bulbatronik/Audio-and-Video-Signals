clc;
clear all;
close all;

A = im2double(imread('Lena_grayscale.bmp'));
figure 
subplot(1,2,1)
imshow(A)
title('Original Lena image')
subplot(1,2,2) 
imhist(A) 
title('Original Lena histogram');

% Additive Gaussian noise
N_a = 0.1 * randn(size(A)); %0.1 is the std
% Additive Rayleigh noise, typical of radar images
%N_a = random('rayl', 0.1, size(A,1), size(A,2));
% Additive Gamma noise, typical of laser images
%N_a = random('gam', 1.5, 0.1, size(A,1), size(A,2));
% Additive Exponential noise
%N_a = random('exp', 0.1, size(A,1), size(A,2));
% Additive uniform noise
%N_a = 0.5 * rand(size(A)) - 0.25;
% General noise (e.g. exponeantial) exploiting uniform distribution and inverse of pdf
% pdf: e^(-lambda*x)
% cdf: 1-e^(-lambda*x)
%N_a_u = rand(size(A));
%N_a = -log(1-N_a_u)*0.1; % lambda = 0.1
figure
subplot(1,2,1)
imagesc(N_a)
daspect([1 1 1])
colorbar()
title('Noise we are adding')
subplot(1,2,2)
hist(N_a(:))
title('Noise histogram')

A_n = A + N_a;
% clamp values on [0, 1]:
A_n(A_n > 1) = 1;
A_n(A_n < 0) = 0;
figure
subplot(1,2,1)
imshow(A_n)
title('Lena with additive noise');
subplot(1,2,2)
imhist(A_n)
title('Histogram of Lena with additive noise');


%% Salt-and-pepper noise
% density 0.1: proportion of altered pixels (Matlab default: 0.05)
% typical of transmission errors
clc;
clear all;
close all;

A = im2double(imread('Lena_grayscale.bmp'));
figure 
subplot(1,2,1)
imshow(A)
title('Original Lena image')
subplot(1,2,2) 
imhist(A) 
title('Original Lena histogram');

A_n = imnoise(A, 'salt & pepper', 0.1);

figure
subplot(1,2,1)
imshow(A_n)
title('Lena image with salt-and-pepper noise');
subplot(1,2,2)
imhist(A_n)
title('Lena histogram with salt-and-pepper noise');

% Adding Salt-and-pepper noise step by step

N = numel((A));
p = 0.1;

k = randi(N,1,round(N*p));
mid_index = round(numel(k)/2);
A_n = A;
A_n(k(1:mid_index)) = 1;
A_n(k((mid_index+1):end)) = 0;

figure
subplot(1,2,1)
imshow(A_n)
title('Lena image with salt-and-pepper noise');
subplot(1,2,2)
imhist(A_n)
title('Lena histogram with salt-and-pepper noise');

%% Denoising in spatial domain
clc;
clear all;
close all;

A = im2double(imread('Lena_grayscale.bmp'));
figure 
subplot(1,2,1)
imshow(A)
title('Original Lena image')
subplot(1,2,2) 
imhist(A) 
title('Original Lena histogram');

% add salt-and-pepper noise to the image
A_n = imnoise(A, 'salt & pepper', 0.1);

figure
subplot(1,2,1)
imshow(A_n)
rmse = sqrt(sum((A-A_n).^2,'all'));
title(sprintf('Lena image with salt-and-pepper noise (%0.2f)',rmse));
subplot(1,2,2)
imhist(A_n)
title('Lena histogram with salt-and-pepper noise');


% filter the noisy image with AVERAGE filter
F_mean_3x3 = fspecial('average', 3);  % average filter 3x3
A_mean_3x3 = conv2(A_n, F_mean_3x3, 'same'); 
figure
subplot(1,2,1)
imshow(A_mean_3x3)
rmse = sqrt(sum((A-A_mean_3x3).^2,'all'));
title(sprintf('Lena image with salt-and-pepper noise after average filter 3x3 (%0.2f)',rmse));
subplot(1,2,2)
imhist(A_mean_3x3)
title('Lena histogram with salt-and-pepper noise after average filter 3x3');

F_mean_5x5 = fspecial('average',5);  % average filter 5x5
A_mean_5x5 = conv2(A_n, F_mean_5x5, 'same'); 
figure
subplot(1,2,1)
imshow(A_mean_5x5);
rmse = sqrt(sum((A-A_mean_5x5).^2,'all'));
title(sprintf('Lena image with salt-and-pepper noise after average filter 5x5 (%0.2f)',rmse));
subplot(1,2,2)
imhist(A_mean_5x5)
title('Lena histogram with salt-and-pepper noise after average filter 5x5');

% filter the noisy image with MEDIAN filter 
A_median_3x3 = medfilt2(A_n,[3 3]);   % median filter 3x3
figure
subplot(1,2,1)
imshow(A_median_3x3);
rmse = sqrt(sum((A-A_median_3x3).^2,'all'));
title(sprintf('Lena image with salt-and-pepper noise after median filter 3x3 (%0.2f)',rmse));
subplot(1,2,2)
imhist(A_median_3x3)
title('Lena histogram with salt-and-pepper noise after median filter 3x3');
%% bilateral filtering
clc
close all
clear all

A = im2double(imread('Lena_grayscale.bmp'));
figure 
imshow(A)
title('Original Lena image')

% Additive Gaussian noise
N_a = 0.05 * randn(size(A)); %0.1 is the std

A_n = A + N_a;
% clamp values on [0, 1]:
A_n(A_n > 1) = 1;
A_n(A_n < 0) = 0;

figure
imshow(A_n)
rmse = sqrt(sum((A-A_n).^2,'all'));
title(sprintf('Lena with additive noise (%0.2f)',rmse));

w_kernel = 5;
sigma_d = 2;

F_gaussian = fspecial('gaussian',w_kernel,sigma_d); 
A_gaussian = conv2(A_n, F_gaussian, 'same'); 
figure
imshow(A_gaussian);
rmse = sqrt(sum((A-A_gaussian).^2,'all'));
title(sprintf('gaussian output (%0.2f)',rmse));

w = (w_kernel-1)/2;
% sigma_r = 0.2;
patch = imcrop(A_n,[1, 1, 30 30]);
patchVar = std2(patch)^2;
sigma_r = sqrt(2*patchVar);
% Pre-compute Gaussian distance weights.
[X,Y] = meshgrid(-w:w,-w:w);
G = exp(-(X.^2+Y.^2)/(2*sigma_d^2));
% Apply bilateral filter.
dim = size(A_n);
B = A_gaussian;

% 
for i = 1+w:dim(1)-w
   for j = 1+w:dim(2)-w
      
         I = A_n((i-w):(i+w),(j-w):(j+w));
      
         % Compute Gaussian intensity weights.
         H = exp(-(I-A_n(i,j)).^2/(2*sigma_r^2));
      
         % Calculate bilateral filter response.
         F = H.*G;
         B(i,j) = sum(F(:).*I(:))/sum(F(:));
         
%          figure(1)
%          subplot(2,2,1)
%          imshow(I)
%          title('patch')
%          subplot(2,2,2)
%          imagesc(G)
%          colormap gray
%          axis equal
%          axis tight
%          title('gaussian kernel')
%          subplot(2,2,3)
%          imagesc(H)
%          colormap gray
%          axis equal
%          axis tight
%          title('intensity kernel')
%          subplot(2,2,4)
%          imshow(F)
%          colormap gray
%          axis equal
%          axis tight
%          title('bilateral kernel')
%          pause()
               
   end
   
end

% for i = 1:dim(1)
%    for j = 1:dim(2)
%       
%          % Extract local region.
%          iMin = max(i-w,1);
%          iMax = min(i+w,dim(1));
%          jMin = max(j-w,1);
%          jMax = min(j+w,dim(2));
%          I = A_n(iMin:iMax,jMin:jMax);
%       
%          % Compute Gaussian intensity weights.
%          H = exp(-(I-A_n(i,j)).^2/(2*sigma_r^2));
%       
%          % Calculate bilateral filter response.
%          F = H.*G((iMin:iMax)-i+w+1,(jMin:jMax)-j+w+1);
%          B(i,j) = sum(F(:).*I(:))/sum(F(:));
%          
%          
%                
%    end
%    
% end

figure
imshow(B)
rmse = sqrt(sum((A-B).^2,'all'));
title(sprintf('bilateral filtering output (%0.2f)',rmse));

B = imbilatfilt(A_n,sigma_r^2,sigma_d);
figure
imshow(B)
rmse = sqrt(sum((A-B).^2,'all'));
title(sprintf('bilateral filtering output w matlab (%0.2f)',rmse));

%% Denoising in frequency domain -> Wiener filtering (from theory, step by step)
clc;
clear all;
close all;

f_I = im2double(imread('Lena_grayscale.bmp'));
figure; imshow(f_I); title('Original image');

LEN = 31;
THETA = 11;
h_D = fspecial('motion', LEN,THETA); % create PSF (point spread function)
figure; imagesc(h_D); colorbar; axis equal; axis tight; 
title('Degrading filter kernel (31px motion blur, angle 11 degrees)');
colormap gray

f_D = conv2(f_I, h_D, 'full');
figure; imshow(f_D);
title('Degraded image (31px motion blur, angle 11 degrees)');

% Wiener step-by-step (noiseless case --> inverse filtering)

F_D = fft2(f_D);
H_D = fft2(h_D, size(f_D,1), size(f_D,2));

H_R = 1./H_D;
F_I_hat = F_D .* H_R;
f_I_hat = ifft2(F_I_hat);
f_I_hat = f_I_hat(1:size(f_I,1),1:size(f_I,2));  
figure;imshow(f_I_hat);
title('Image restored by Wiener filter (h_D known and noiseless case)');
%%
% additive noise
f_n = 0.02*randn(size(f_D));
f_O = f_D + f_n;
f_O(f_O > 1) = 1;
f_O(f_O < 0) = 0;
figure;imshow(f_O);
title('Image degraded by spatial filter and additive noise');

F_O = fft2(f_O);
H_R = 1./H_D;

F_I_hat = F_O .* H_R;
f_I_hat = ifft2(F_I_hat);
f_I_hat = f_I_hat(1:size(f_I,1),1:size(f_I,2));  
figure;imshow(f_I_hat);
title('Image restored by inverse filter (h_D known but noise)');
%%

% Wiener step-by-step (additive noise; H_D, noise power spectrum and
% original image known

F_I = fft2(f_I, size(f_O,1), size(f_O,2));
F_n = fft2(f_n, size(f_O,1), size(f_O,2));
H_D = fft2(h_D, size(f_O,1), size(f_O,2)); 
Power_F_I = abs(F_I).^2;
Power_F_n = abs(F_n).^2;
Power_H_D = abs(H_D).^2;
% Weiner's filter formula 
H_R = (conj(H_D).*Power_F_I) ./ ((Power_H_D.*Power_F_I) + Power_F_n);
F_I_hat = F_O .* H_R;
f_I_hat = ifft2(F_I_hat, 'symmetric');  

f_I_hat = f_I_hat(1:size(f_I,1), 1:size(f_I,2));  
figure; imshow(f_I_hat);
title(['Image restored by Weiner filter (H_D, noise power spectrum ', ...
        'and original image power spectrum known)']);

NSR = Power_F_n./Power_F_I; %pointwise noise to signal ratio     
%NSR = mean(Power_F_n(:))./mean(Power_F_I(:)); %avg noise to signal ratio   
    
H_R = conj(H_D)./(Power_H_D + NSR);
F_I_hat = F_O .* H_R;
f_I_hat = ifft2(F_I_hat,'symmetric');  
f_I_hat = f_I_hat(1:size(f_I,1),1:size(f_I,2));  
figure;imshow(f_I_hat);
title('Image restored by Weiner filter (H_D and SNR known)');



%% Using the matlab Wiener implementation
clc;
clear all;
close all;

I = im2double(imread('Lena_grayscale.bmp'));
figure, imshow(I);
title('Original Image');

% simulated blurring using a Point Spread Function (PSF)
LEN = 31;
THETA = 31;
PSF = fspecial('motion', LEN, THETA);
figure, imshow( PSF / max(PSF(:)) );
title('PSF');
blurred = conv2(I, PSF, 'full');
figure, imshow(blurred);
title('Blurred');

% restoration using Weiner deconvolution
wnr1 = deconvwnr(blurred, PSF);
figure, imshow(wnr1);
title('Restored, True PSF');

% Added noise and blurring
noise = 0.02 * randn(size(blurred)); % gaussian noise with 0.1 standard deviation
blurredNoisy = blurred + noise;
figure, imshow(blurredNoisy);
title('Blurred & Noisy');

% restoration using the known PSF
wnr4 = deconvwnr(blurredNoisy, PSF);
figure, imshow(wnr4);
title('Inverse Filtering of Noisy Data');

NSR = sum(noise(:).^2) / sum(I(:).^2);
wnr5 = deconvwnr(blurredNoisy, PSF, NSR);
figure, imshow(wnr5);
title('Restored with NSR');

% restoration using as estimate of the noise power 1/2 of the real one
wnr6 = deconvwnr(blurredNoisy, PSF, NSR/2);
figure, imshow(wnr6);
title('Restored with NSR/2');

% improved restoration using the perfect knowledge of the autocorrelation

NP = abs(fft2(noise)).^2;
NPOW = sum(NP(:))/numel(noise); % noise power
NCORR = ifft2(NP,'symmetric'); % noise ACF
IP = abs(fft2(I)).^2;
IPOW = sum(IP(:))/numel(I); % original image power
ICORR = ifft2(IP,'symmetric'); % image ACF

wnr7 = deconvwnr(blurredNoisy,PSF,NCORR,ICORR);
figure, imshow(wnr7);
title('Restored with ACF');

