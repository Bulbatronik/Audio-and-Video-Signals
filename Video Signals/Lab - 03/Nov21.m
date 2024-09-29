clc, clear, close all;
f = im2double(rgb2gray(imread('zebra.jpg')));% to grayscale (and double)

figure
imshow(f)% f - discrete function. For each point - value of intensity
% apply the FFT
%%
F = fft2(f);% same spatial resolution with complex values

F(1,1)
sum(f, 'all')% zero value of fft - sum of all pixels
% SAME

F_magnitude = abs(F);
F_phase = angle(F);

figure
imagesc(F_magnitude)%color map image
title('Magnitude of F')
colorbar% zoom 1,1

figure
imagesc(F_phase)
title('Phase of F')
colorbar

% more generic way to repres FT
F_power_spectrum = F_magnitude.^2;

figure
imagesc(F_power_spectrum)
title('Power spectrum of F')
colorbar
%% More conventual repres
close all;
F_shift = fftshift(F);

F_magnitude = abs(F_shift);
F_phase = angle(F_shift);

figure
imagesc(F_magnitude)%color map image
title('Magnitude of F')
colorbar% zoom 1,1

figure
imagesc(F_phase)
title('Phase of F')
colorbar

% more generic way to repres FT
F_power_spectrum = F_magnitude.^2;

figure
imagesc(F_power_spectrum)
title('Power spectrum of F')
colorbar

figure% do like that
imagesc(log(1+F_power_spectrum))
title('log Power spectrum of F')
colorbar
%% start with synthetic image to understand log Power spectrum of F
clc, clear, close all;

s = 500;%square image
test = zeros(s,s);

f1 = 0.05;%frequency
f2 = 0.02;
for i=1:s% sine along the columns
    test(:,i) = sin(2*pi*f2*i);% change freq. If you add +const -> same psd, just the phase is different
end

%normalize -1,1 -> 0, 1
test = (test-min(test(:)))/(max(test(:))-min(test(:)));

figure
imshow(test)
title('Test image')

test_DFT = fftshift(fft2(test));
test_DFT_power_spectrum = real(test_DFT).^2+imag(test_DFT).^2;% abs

figure
imagesc(log(1+test_DFT_power_spectrum))
title('log Power spectrum of test')
colorbar; axis equal; axis tight; colormap gray, caxis([0,10])
% OUTPUT IS ZERO EVERYWHERE EXCEPT THE CENTER(SUM OVER ALL THE VALUES) AND
% THE PEAL OVER X AXIS IN 276, 251
(256 - 251)/500% (position - the center)/size ot the image; f2->(261 - 251)/500 = 0.02
%%
clc, clear, close all;

s = 500;%square image
test = zeros(s,s);

f1 = 0.05;%frequency

for i=1:s% sine along the rows
    test(i,:) = sin(2*pi*f1*i);
end

%normalize -1,1 -> 0, 1
test = (test-min(test(:)))/(max(test(:))-min(test(:)));

figure
imshow(test)
title('Test image')

test_DFT = fftshift(fft2(test));
test_DFT_power_spectrum = real(test_DFT).^2+imag(test_DFT).^2;% abs

figure
imagesc(log(1+test_DFT_power_spectrum))
title('log Power spectrum of test')
colorbar; axis equal; axis tight; colormap gray, caxis([0,10])
% horiz lines are peaks along the vertical axis of psd
%%
clc, clear, close all;

s = 500;%square image
test = zeros(s,s);

f1 = 0.05;%frequency
f2 = 0.02;

for i=1:s% sine along the rows
    test(i,:) = sin(2*pi*f1*i)+sin(2*pi*f2*i);
end

%normalize -1,1 -> 0, 1
test = (test-min(test(:)))/(max(test(:))-min(test(:)));

figure
imshow(test)
title('Test image')

test_DFT = fftshift(fft2(test));
test_DFT_power_spectrum = real(test_DFT).^2+imag(test_DFT).^2;% abs

figure
imagesc(log(1+test_DFT_power_spectrum))
title('log Power spectrum of test')
colorbar; axis equal; axis tight; colormap gray, caxis([0,10])
% now you have five peaks. 2 are related to f1 (higher freq), 2 to f2 (lower freq)
%%
clc, clear, close all;

s = 500;%square image
test = zeros(s,s);

f1 = 0.05;%frequency
f2 = 0.02;
f3 = 0.1;
for i=1:s% sine along the rows
    test(i,:) = test(i,:)+sin(2*pi*f1*i)+sin(2*pi*f2*i);
    test(:,i) = test(:,i) + sin(2*pi*f3*i);
end

%normalize -1,1 -> 0, 1
test = (test-min(test(:)))/(max(test(:))-min(test(:)));

figure
imshow(test)
title('Test image')

test_DFT = fftshift(fft2(test));
test_DFT_power_spectrum = real(test_DFT).^2+imag(test_DFT).^2;% abs

figure
imagesc(log(1+test_DFT_power_spectrum))
title('log Power spectrum of test')
colorbar; axis equal; axis tight; colormap gray, caxis([0,10])
%% cut only the central part
clc, clear, close all;

s = 500;%square image
test = zeros(s,s);

f1 = 0.05;%frequency
f2 = 0.02;
f3 = 0.1;
for i=1:s% sine along the rows
    test(i,:) = test(i,:)+sin(2*pi*f1*i)+sin(2*pi*f2*i);
    test(:,i) = test(:,i) + sin(2*pi*f3*i);
end

test = imrotate(test,30,'crop', 'bicubic');
%normalize -1,1 -> 0, 1
test = (test-min(test(:)))/(max(test(:))-min(test(:)));

figure
imshow(test)
title('Test image')

test_DFT = fftshift(fft2(test));
test_DFT_power_spectrum = real(test_DFT).^2+imag(test_DFT).^2;% abs

figure
imagesc(log(1+test_DFT_power_spectrum))
title('log Power spectrum of test')
colorbar; axis equal; axis tight; colormap gray, caxis([0,10])
% not just simply rotated points(intersections - 7 points). Lines???
% rotation intriduces some edges -> new frequencies -> cut only the central part
%% what if you rotate?
clc, clear, close all;

s = 500;%square image
test = zeros(s,s);

f1 = 0.05;%frequency
f2 = 0.02;
f3 = 0.1;
for i=1:s% sine along the rows
    test(i,:) = test(i,:)+sin(2*pi*f1*i)+sin(2*pi*f2*i);
    test(:,i) = test(:,i) + sin(2*pi*f3*i);
end


test = imrotate(test,30,'crop', 'bicubic');
test = test((end/2-150):(end/2+150),(end/2-150):(end/2+150));
%normalize -1,1 -> 0, 1
test = (test-min(test(:)))/(max(test(:))-min(test(:)));

figure
imshow(test)
title('Test image')

test_DFT = fftshift(fft2(test));
test_DFT_power_spectrum = real(test_DFT).^2+imag(test_DFT).^2;% abs

figure
imagesc(log(1+test_DFT_power_spectrum))
title('log Power spectrum of test')
colorbar; axis equal; axis tight; colormap gray, caxis([0,10])
% new lines still 
%%
test_mod = zeros(2*size(test));
test_mod(1:(end/2),1:(end/2)) = test;
test_mod((end/2+1):end,1:(end/2)) = test;
test_mod((end/2+1):end,(end/2+1):end) = test;
test_mod(1:(end/2),(end/2+1):end) = test;

figure
imshow(test_mod)%now yous see the lines
%% remove those lines
clc, clear, close all;

s = 500;%square image
test = zeros(s,s);

f1 = 0.05;%frequency
f2 = 0.02;
f3 = 0.1;
for i=1:s% sine along the rows
    test(i,:) = test(i,:)+sin(2*pi*f1*i)+sin(2*pi*f2*i);
    test(:,i) = test(:,i) + sin(2*pi*f3*i);
end

test = imrotate(test,30,'crop', 'bicubic');
test = test((end/2-150):(end/2+150),(end/2-150):(end/2+150));
%normalize -1,1 -> 0, 1
test = (test-min(test(:)))/(max(test(:))-min(test(:)));


inner_r = 50;

for i=1:size(test,1)
    for j=1:size(test,2)
        r = sqrt((i-size(test,1)/2)^2+(j-size(test,2)/2)^2);
        if(r>inner_r)
            test(i,j)=0.5;
        end
    end
end

figure
imshow(test)
title('Test image')

test_DFT = fftshift(fft2(test));
test_DFT_power_spectrum = real(test_DFT).^2+imag(test_DFT).^2;% abs

figure
imagesc(log(1+test_DFT_power_spectrum))
title('log Power spectrum of test')
colorbar; axis equal; axis tight; colormap gray, caxis([0,10])
%% remove the edge of the circle
clc, clear, close all;

s = 500;%square image
test = zeros(s,s);

f1 = 0.05;%frequency
f2 = 0.02;
f3 = 0.1;
for i=1:s% sine along the rows
    test(i,:) = test(i,:)+sin(2*pi*f1*i)+sin(2*pi*f2*i);
    test(:,i) = test(:,i) + sin(2*pi*f3*i);
end

test = imrotate(test,30,'crop', 'bicubic');
test = test((end/2-150):(end/2+150),(end/2-150):(end/2+150));
%normalize -1,1 -> 0, 1
test = (test-min(test(:)))/(max(test(:))-min(test(:)));

%alpha blending: RING TRANSITION TO REMOVE THE EDGE
inner_r = 50;
outer_r = 150;%% 

for i=1:size(test,1)
    for j=1:size(test,2)
        r = sqrt((i-size(test,1)/2)^2+(j-size(test,2)/2)^2);
        if(r>inner_r)
            if(r>outer_r)
                test(i,j)=0.5;
            else 
                alpha = (r - inner_r)/(outer_r-inner_r);
                test(i, j) = test(i,j)*(1-alpha)+0.5*alpha;
            end
        end
    end
end

figure
imshow(test)
title('Test image')

test_DFT = fftshift(fft2(test));
test_DFT_power_spectrum = real(test_DFT).^2+imag(test_DFT).^2;% abs

figure
imagesc(log(1+test_DFT_power_spectrum))
title('log Power spectrum of test')
colorbar; axis equal; axis tight; colormap gray, caxis([0,10])
% interpolations -> new frequencies, as well as alpha blending
%%
clc, clear, close all
%f = im2double(rgb2gray(imread('zebra.jpg')));% zebra.jpg
f = im2double(imread('Lena_grayscale.bmp'));
%f = im2double(imread('SEM.jpg'));

figure
imshow(f)
title('original image')

F = fftshift(fft2(f));
F_magnitude = abs(F);
F_power_spectrum = F_magnitude.^2;

figure
imagesc(log(1+F_power_spectrum))
title('log Power spectrum of F')
colorbar; axis equal; axis tight; colormap gray
% lines of fft orthogonal to the edges. PSD to build some anomaly detection
% tool
%% from the freq domain to the spatial domain

%f_reconstructed = ifft2(F, 'symmetric');% shifted only for visualization purposes ->GO BACK
f_reconstructed = ifft2(ifftshift(F), 'symmetric');

figure
imshow(f_reconstructed)
title('Reconstructed image')
%% phase swap between images
clc, clear, close all
img_1 = im2double(imread('Lena_grayscale.bmp'));

figure
imshow(img_1)
title('original image 1')

F1 = fft2(img_1);
F1_magnitude = abs(F1);
F1_phase = angle(F1);

img_2 = im2double(imread('SEM.jpg'));
img_2 = imresize(img_2, size(img_1));

figure
imshow(img_2)
title('original image 2')

F2 = fft2(img_2);
F2_magnitude = abs(F2);
F2_phase = angle(F2);

%swap phases
F_rec_1 = complex(F2_magnitude.*cos(F1_phase), F2_magnitude.*sin(F1_phase));
F_rec_2 = complex(F1_magnitude.*cos(F2_phase), F1_magnitude.*sin(F2_phase));

img_1_rec = ifft2(F_rec_1, 'symmetric');
img_2_rec = ifft2(F_rec_2, 'symmetric');

figure
imshow(img_1_rec)
title('magnitude of two, phase of one')

figure
imshow(img_2_rec)
title('magnitude of one, phase of two')

%% filtering in the drequency domain
clc, clear, close all
f = im2double(imread('Lena_grayscale.bmp'));

figure
imshow(f)
title('original image')

%remove and add later
avg_f = mean(f(:));
f = f - avg_f;
%

F = fftshift(fft2(f));
F_magnitude = abs(F);
F_power_spectrum = F_magnitude.^2;

figure
imagesc(log(1+F_power_spectrum))
title('log Power spectrum of F')
colorbar; axis equal; axis tight; colormap gray

H = ones(size(F));% filter init, just copies the image

% LP filter
D0 = 50;% cutoff frequency, try 0 -> obtain just the average value over the image
n = 20;%order of a filter, try 20
mid = [1+size(H,1)/2, 1+size(H,2)/2];% middle point
for i=1:size(H,1)
    for j=1:size(H,2)
        r = sqrt((i-mid(1))^2+(j-mid(2))^2);
        
        %ideal filter
        %if r>=D0
        %    H(i, j) = 0;
        %end

        %gaussian filter
        H(i,j) = exp(-(r^2)/(2*D0^2));% std = cut-off freq
        
        % butterworth filter
        %H(i,j) = 1/(1+(r/D0)^(2*n));
    end
end

%high pass filter = 1 - low pass filter
H = 1 - H;
%

%bandpass filter = lp + hp
H2 = ones(size(F));%LP
D0 = 80;% cutoff frequency, try 0 -> obtain just the average value over the image
n = 10;%order of a filter, try 20
mid = [1+size(H,1)/2, 1+size(H,2)/2];% middle point
for i=1:size(H2,1)
    for j=1:size(H2,2)
        r = sqrt((i-mid(1))^2+(j-mid(2))^2);
        
        %ideal filter
        %if r>=D0
        %    H2(i, j) = 0;
        %end

        %gaussian filter
        H2(i,j) = exp(-(r^2)/(2*D0^2));% std = cut-off freq
        
        % butterworth filter
        %H2(i,j) = 1/(1+(r/D0)^(2*n));
    end
end

H = H.*H2;%BP
%
figure
imshow(H)
colorbar
title('filter')

figure
surf(H, 'LineStyle', 'none')
colorbar; colormap gray; 
title('filter')

F_filtered = H.*F;
F_filtered_magnitude = abs(F_filtered);
F_filtered_power_spectrum = F_filtered_magnitude.^2;

figure
imagesc(log(1+F_filtered_power_spectrum))
title('log Filtered Power spectrum of F')
colorbar; axis equal; axis tight; colormap gray

img_filtered = ifft2(ifftshift(F_filtered), 'symmetric');

figure
imshow(img_filtered)
title('Filtered image')
% bad results because the filter is ideal
% for gaussian we can't control the steepness (cut-off) 
% WE LOSE HIGH FREQ DETAILS

figure
imagesc(img_filtered)
colorbar

img_filtered = (img_filtered- min(img_filtered(:))/(max(img_filtered(:))-min(img_filtered(:))));
figure
imshow(img_filtered+avg_f)
title('Filtered image (normalized)')
%%
