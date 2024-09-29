% FOURIER TRANSFORM

%% Fourier transform of an image
clc;
clear all;
close all;

f = im2double(rgb2gray(imread('zebra.jpg')));
figure; imshow(f);

F = fft2(f);

F(1,1) - sum(f(:))

F_magnitude = abs(F);
F_phase = angle(F);
F_power_spectrum = F_magnitude.^2;
figure
imagesc(F_magnitude); colorbar; title('Spectrum modulus');
figure
imagesc(F_phase); colorbar; title('Spectrum phase');
figure
imagesc(F_power_spectrum); colorbar; title('Power spectrum');

F_shift = fftshift(F);

F_magnitude_shift = abs(F_shift);
F_phase_shift = angle(F_shift);
F_power_spectrum_shift = F_magnitude_shift.^2;
figure
imagesc(F_magnitude_shift); colorbar; title('Spectrum modulus');
figure
imagesc(F_phase_shift); colorbar; title('Spectrum phase');
figure
imagesc(F_power_spectrum_shift); colorbar; title('Power spectrum');
figure
imagesc(log(1+F_magnitude_shift)); colorbar;
title('Spectrum log modulus');
figure
imagesc(log(1+F_power_spectrum_shift)); colorbar;
title('Log power spectrum');



%% Synthetic images to better understand FFT

clc
clear all
close all

s = 500;
test = zeros(s,s);

f1 = 0.05;
f2 = 0.02;
f3 = 0.1;

for i=1:s

    %1)
    test(:,i) = sin(2*pi*f1*i); %(276-251)/500 = f1
    %2)
    %test(:,i) = sin(2*pi*f2*i); %(261-251)/500
    %3)
    %test(i,:) = sin(2*pi*f2*i);
    %4)
    %test(:,i) = sin(2*pi*f1*i+2); %shifted %equal to 1)
    %5)
    %test(i,:) = sin(2*pi*f1*i) + sin(2*pi*f2*i); %2 freq    
    %6)
    %test(i,:) = test(i,:) + sin(2*pi*f1*i) + sin(2*pi*f2*i);
    %test(:,i) = test(:,i) + sin(2*pi*(f3)*i);   
    
end

%normalization
test = (test - min(test(:)) )/(max(test(:)) - min(test(:)) );

%test=imrotate(test,30,'crop','bicubic');
%test=test((end/2-150):(end/2+150),(end/2-150):(end/2+150));
%  
%  inner_r = 1;
%  outer_r = 150;
%  for i=1:size(test,1)
%      for j=1:size(test,2)
%          r = sqrt((i-(size(test,1)/2))^2+(j-(size(test,2)/2))^2);
%          if r > inner_r
%              if r > outer_r
%                  test(i,j) = 0.5;
%              else 
%                  alpha = (r - inner_r)/(outer_r-inner_r);
%                  test(i,j) = test(i,j) * (1-alpha) + (alpha)*0.5;
%              end          
%          end
%      end
%  end


figure()
imshow(test)


% test_mod = zeros(2*size(test));
% test_mod(1:(end/2),1:(end/2)) = test;
% test_mod(1:(end/2),(end/2+1):end) = test;
% test_mod((end/2+1):end,1:(end/2)) = test;
% test_mod((end/2+1):end,(end/2+1):end) = test;
% figure;imshow(test_mod)

test_DFT = fft2(im2double(test));
test_DFT = fftshift(test_DFT);

test_DFT_power_spectrum = real(test_DFT).^2 + imag(test_DFT).^2;
figure
imagesc(log(1+test_DFT_power_spectrum));
colorbar; axis equal; axis tight; colormap gray; caxis([0,10])
title('test - Log power spectrum');



%% Fourier transform interpretation (dc shifted to the center)
clc;
clear all;
close all;

%img_gray = im2double(imread('Lena_grayscale.bmp'));
img_gray = im2double(imread('SEM.jpg'));
figure; imshow(img_gray); title('Grayscale image');


gray_DFT = fftshift(fft2(img_gray));
%gray_DFT = (fft2(img_gray));

gray_DFT_magnitude = abs(gray_DFT);
gray_DFT_phase = angle(gray_DFT);
gray_DFT_power_spectrum = gray_DFT_magnitude.^2;
figure
imagesc(log(1+gray_DFT_magnitude));
colorbar; axis equal; axis tight;
title('Spectrum log modulus');
figure
imagesc(gray_DFT_phase);
colorbar; axis equal; axis tight; title('Spectrum phase');
figure
imagesc(log(1+gray_DFT_power_spectrum));
colorbar; axis equal; axis tight; colormap gray; title('Log power spectrum');


%% Inverse Fourier transform 
clc;
clear all;
close all;

%img_gray = im2double(imread('Lena_grayscale.bmp'));
img_gray = im2double(imread('SEM.jpg'));
figure; imshow(img_gray); title('Grayscale image');


gray_DFT = fftshift(fft2(img_gray));
% You can skip ifftshift if you didn't apply fftshift
gray_DFT = ifftshift(gray_DFT);

img_gray_reconstructed = ifft2(gray_DFT, 'symmetric');
figure
imshow(img_gray_reconstructed); title('Reconstructed image');


%% Phase swap between images
clc;
clear all;
close all;

img_gray_1 = im2double(imread('Lena_grayscale.bmp'));
figure; imshow(img_gray_1); title('Lena Grayscale');

gray_DFT_1 = fft2(img_gray_1);
gray_DFT_magnitude_1 = abs(gray_DFT_1);
gray_DFT_phase_1 = angle(gray_DFT_1);

img_gray_2 = im2double(imread('SEM.jpg'));
img_gray_2 = imresize(img_gray_2, size(img_gray_1));
figure; imshow(img_gray_2);
title('SEM Grayscale resized to Lena dimensions');

gray_DFT_2 = fft2(img_gray_2);
gray_DFT_magnitude_2 = abs(gray_DFT_2);
gray_DFT_phase_2 = angle(gray_DFT_2);

% swap the phases and reconstruct the images
gray_DFT_rec_1 = ...
    complex(gray_DFT_magnitude_2.*cos(gray_DFT_phase_1), ...
            gray_DFT_magnitude_2.*sin(gray_DFT_phase_1));
img_gray_reconstructed_1 = ifft2(gray_DFT_rec_1, 'symmetric');
figure
imshow(img_gray_reconstructed_1);
title('Recontructed image: SEM modulus, Lena phase');


gray_DFT_rec_2 = ...
    complex(gray_DFT_magnitude_1.*cos(gray_DFT_phase_2), ...
            gray_DFT_magnitude_1.*sin(gray_DFT_phase_2));
img_gray_reconstructed_2 = ifft2(gray_DFT_rec_2, 'symmetric');
figure
imshow(im2double(img_gray_reconstructed_2));
title('Recontructed image: Lena modulus, SEM phase');


%% Filtering in the frequency domain
clc
clear all
close all

img = imread('Lena_grayscale.bmp');
figure; imshow(img);

DFT = fft2(im2double(img));
DFT = fftshift(DFT);

DFT_power_spectrum = real(DFT).^2 + imag(DFT).^2;
figure
imagesc(log(1+DFT_power_spectrum));
colorbar; axis equal; axis tight; colormap gray; caxis([0, 10])
title('Original Log power spectrum')

D0 = 50;
n = 2;
%DFT_mod = DFT;

H = ones(size(DFT));
mid = [1 + size(DFT,1)/2,1 + size(DFT,2)/2];
%low pass filtering
for i=1:size(DFT,1)
    for j=1:size(DFT,2)
        r = sqrt((i-mid(1))^2+(j-mid(2))^2);
        %ideal filter
        %if r>D0
        %       H(i,j) = 0; 
        %end
        %butterworth
        %H(i,j) = 1 / (1 + (r/D0)^(2*n));
        %gaussian
        H(i,j) = exp(-(r^2)/(2*D0^2));
    end
end


%high pass filtering
H = 1 - H;

%bandpass
% D0 = 60;
% n = 2;
% %DFT_mod = DFT;
% H_2 = ones(size(DFT));
% %low pass filtering
% for i=1:size(DFT,1)
%     for j=1:size(DFT,2)
%         r = sqrt((i-mid(1))^2+(j-mid(2))^2);
%         %ideal filter
%         %if r>D0
%         %       H_2(i,j) = 0; 
%         %end
%         %butterworth
%         %H_2(i,j) = 1 / (1 + (r/D0)^(2*n));
%         %gaussian
%         H_2(i,j) = exp(-(r^2)/(2*D0^2));
%     end
% end
% 
% H = H.*H_2;

figure
imshow(H);
colorbar
title('Filter')

figure
surf(H,'LineStyle','none')
colorbar; colormap gray;
title('Filter')


DFT_filt = H.*DFT ;

DFT_filt_power_spectrum = real(DFT_filt).^2 + imag(DFT_filt).^2;
figure
imagesc(log(1+DFT_filt_power_spectrum));
colorbar; axis equal; axis tight; colormap gray; caxis([0, 10])
title('Filtered Log power spectrum')
    
DFT_filt = ifftshift(DFT_filt);    
img_filtered = (ifft2(DFT_filt, 'symmetric'));
img_filtered = (img_filtered - min(img_filtered(:))) / (max(img_filtered(:)) - min(img_filtered(:)));
figure
imshow(img_filtered);
title('img filtered');