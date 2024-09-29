clc, clear all; close all;
% additive noise. Pdf is required
A = im2double(imread('Lena_grayscale.bmp'));

figure
subplot(1,2,1)
imshow(A)
title('original image')

subplot(1,2,2)
imhist(A)
title('original image histogram')


% add noise to test noise removal techniques

%gaussian noise
%N_a = 0+ randn(size(A))*0.1;%0.1 std, 0 mean

% raylay noise (for radar images)
%N_a = random('ray', 0.1, size(A,1),size(A,2));% add bright components -> hist smoothening in positive direction

% gamma noise (for lazers)
%N_a = random('gam', 1.5, 0.1, size(A,1),size(A,2));

% exp noise 
%N_a = random('exp', 0.1, size(A,1),size(A,2));

% uniform noise (generate any kind of noise if you know its pdf)
%N_a = 0.5*rand(size(A))-0.25;% [0,1] -> [-0.25, 0.25]

%(generate any kind of noise from the uniform if you know its pdf) EX.expon.
N_a = rand(size(A));
%pdf:e&(-lambda*x)
%cdf:1-e(-lambda*x)
N_a = -log(1-N_a)*0.1;%lambda = 0.1

figure
subplot(1,2,1)
imagesc(N_a)
axis equal; axis tight; colorbar;
title('noise')

subplot(1,2,2)
hist(N_a(:))% to plot the negative values
title('noise histogram')

A_n = A +N_a;
A_n(A_n<0)=0;
A_n(A_n>1)=1;

figure
subplot(1,2,1)
imshow(A_n)
title('noisy image')

subplot(1,2,2)
imhist(A_n)
title('noisy image histogram')
% noise has a smoothing effect on the histogram
%% salt and pepper noise. Typical for transmission errors
clc, clear all; close all;
A = im2double(imread('Lena_grayscale.bmp'));

figure
subplot(1,2,1)
imshow(A)
title('original image')

subplot(1,2,2)
imhist(A)
title('original image histogram')

A_n = imnoise(A,'salt & pepper', 0.1);% add white and black pixels randomly. Hist is not very effected by it (Shape)
% 0.1 - frequency

% implement s&p by hand
% N = numel(A);
% p = 0.1;
% k = randi(N, 1, round(p*N));
% mid_index = round(numel(k)/2);
% A_n = A;
% A_n(k(1:mid_index)) = 1;
% A_n(k(mid_index+1:end)) = 0;

figure
subplot(1,2,1)
imshow(A_n)
title('noisy image')

subplot(1,2,2)
imhist(A_n)
title('noisy image histogram')

% try to remove it now (SPATIAL DOMAIN)

%1) conv with the average filter -> is it good?(NO)
% F_mean = fspecial('average', 3);% kernel (180 rotation)
% A_denoised = conv2(A_n, F_mean, 'same');
% bad result, the s$p are visible, plus you LP the image -> hist is smoother

%2)apply median filter instead
A_denoised = medfilt2(A_n, [3,3]);% kernel (180 rotation)
% can't apply conv, because the operation is non-linear
% good results, simmilar to the orig. Shape of the hist is recovered
% if we have a large number of salts/peppers together -> larger kernel is required, but you LP more

figure
subplot(1,2,1)
imshow(A_denoised)
title('denoised image')

subplot(1,2,2)
imhist(A_denoised)
title('denoised image histogram')
%% back to additive noise removal
clc, clear all; close all;

A = im2double(imread('Lena_grayscale.bmp'));

figure
imshow(A)
title('original image')

% add noise to test noise removal techniques

%gaussian noise
N_a = 0+ randn(size(A))*0.1;%0.1 std, 0 mean

A_n = A +N_a;
A_n(A_n<0)=0;
A_n(A_n>1)=1;

rmse = sqrt(sum((A-A_n).^2, 'all'));

figure
imshow(A_n)
title(sprintf('noisy image (%0.2f)', rmse))

%try gaussizn filter (sort of a weighted average -> LP filter)
w_kernel = 5;
sigma_d = 2;
F_gaussian = fspecial('gaussian', w_kernel, sigma_d);% 180 rotated, but symmetric
%You attenuate the noise, but you destroy the high frequency details (isotropic filter, not very 'smart')
A_gaussian = conv2(A_n, F_gaussian, 'same');

rmse = sqrt(sum((A-A_gaussian).^2, 'all'));

figure
imshow(A_gaussian)
title(sprintf('gausiian output image (%0.2f)', rmse))

% bilateral filter
w = (w_kernel-1)/2;
sigma_r = 0.25;
G = fspecial('gaussian', w_kernel, sigma_d);
dim = size(A_n);
B=A_gaussian;

for i =1+w:dim(1)-w
    for j=1+w:dim(2)-w

        I = A_n((i-w):(i+w),(j-w):(j+w));
        H = exp(-(I-A_n(i,j)).^2/(2*sigma_r^2));% gaussian on the intensity
        F = H.*G;

        B(i,j) = sum(F(:).*I(:))/sum(F(:));
       
%         figure(1)
%         subplot(2,2,1)
%         imshow(I)
%         title('patch')
% 
%         subplot(2,2,2)
%         imagesc(G)
%         colormap gray; axis equal; axis tight
%         title('gaussian kernel')
%         
%         subplot(2,2,3)
%         imagesc(H)
%         colormap gray; axis equal; axis tight
%         title('intensity kernel')
% 
%         subplot(2,2,4)
%         imagesc(F)
%         colormap gray; axis equal; axis tight
%         title('bilateral kernel')
%         
%         pause(0.1);
    end
end

figure
imshow(B)
rmse = sqrt(sum((A-B).^2, 'all'));
title(sprintf('bilateral filtering output (%0.2f)',rmse));
%% directly matlab for bilateral filter
B = imbilatfilt(A_n,sigma_r^2,sigma_d);% variance and std
figure
imshow(B)
rmse = sqrt(sum((A-B).^2,'all'));
title(sprintf('bilateral filtering output w matlab (%0.2f)',rmse));
%% wiener filter (work in the freq domain now)
clc, clear all; close all;
% assumption: degrading filter is known
f_I = im2double(imread('Lena_grayscale.bmp'));

figure
imshow(f_I)
title('original image')

len = 31;% shift
theta = 11;% rotate
h_D = fspecial('motion',len, theta);

figure
imagesc(h_D)
axis equal; axis tight
title('degrading filter kernel')

f_D = conv2(f_I,h_D,"full");% to also compute the conv at the borders. MUST be done on the whole image

% noise part, comment 
f_n = 0.0002*randn(size(f_D));% SMALL NOISE
f_D = f_D + f_n;

figure
imshow(f_D)
title('degraded image')
% ideal filter

F_D = fft2(f_D);% freq domain
% assumption: we know the degrading filter (you know how long you have been exposed and the direction
H_D = fft2(h_D,size(f_D,1),size(f_D,2));

H_R = 1./H_D; %ideal filter

F_I_hat = F_D.*H_R;
f_I_hat = ifft2(F_I_hat);
f_I_hat = f_I_hat(1:size(f_I,1),1:size(f_I,2));% remove the padding

figure
imshow(f_I_hat)
title('restored image')% great noise removal if no noise. Doesn't work if the noise has a random part (see "SMALL NOISE")
%% actual wiener filter
clc, clear all; close all;
% assumption: degrading filter is known
f_I = im2double(imread('Lena_grayscale.bmp'));

figure
imshow(f_I)
title('original image')

len = 31;% shift
theta = 11;% rotate
h_D = fspecial('motion',len, theta);

figure
imagesc(h_D)
axis equal; axis tight
title('degrading filter kernel')

f_D = conv2(f_I,h_D,"full");% to also compute the conv at the borders. MUST be done on the whole image

% noise part, comment 
f_n = 0.0002*randn(size(f_D));% SMALL NOISE
f_D = f_D + f_n;

figure
imshow(f_D)
title('degraded image')
% ideal filter

F_D = fft2(f_D);% freq domain
% assumption: we know the degrading filter (you know how long you have been exposed and the direction
H_D = fft2(h_D,size(f_D,1),size(f_D,2));

% WF
F_I = fft2(f_I, size(f_D,1), size(f_D, 2));
F_n = fft2(f_n,size(f_D,1), size(f_D, 2));%noise
H_D = fft2(h_D,size(f_D,1), size(f_D, 2));
Power_F_I = abs(F_I).^2;
Power_F_n = abs(F_n).^2;
Power_H_D = abs(H_D).^2;

H_R = (conj(H_D).*Power_F_I)./(Power_H_D.*Power_F_I+Power_F_n);% WIENER FORMULA

% another way
%NSR = Power_F_n./Power_F_I;
%NSR = mean(Power_F_n(:))/mean(Power_F_I(:));% less accurate estimation of NSR
%H_R = (conj(H_D))./(Power_H_D+NSR);% wiener formula

%
F_I_hat = F_D.*H_R;
f_I_hat = ifft2(F_I_hat);
f_I_hat = f_I_hat(1:size(f_I,1),1:size(f_I,2));% remove the padding

figure
imshow(f_I_hat)
title('restored image')% almost perfect restoration. sonsidering the noise
%% Wiener MATLAB
clc, clear all; close all;
% assumption: degrading filter is known
f_I = im2double(imread('Lena_grayscale.bmp'));

figure
imshow(f_I)
title('original image')

len = 31;% shift
theta = 11;% rotate
h_D = fspecial('motion',len, theta);

figure
imagesc(h_D)
axis equal; axis tight
title('degrading filter kernel')

f_D = conv2(f_I,h_D,"full");% to also compute the conv at the borders. MUST be done on the whole image
% noise part, comment 
f_n = 0.0002*randn(size(f_D));% SMALL NOISE
f_D = f_D + f_n;

figure
imshow(f_D)
title('degraded image')
% ideal filter

NSR = sum(f_n(:).^2)/sum(f_I(:).^2);% to be estimated properly. If no noise -> remove the input

f_I_hat = deconvwnr(f_D,h_D, NSR);


figure
imshow(f_I_hat)
title('restored image')% almost perfect restoration. sonsidering the noise