clc 
close all
clear all
%a)
img = im2double(imread('input_image.png')); 
figure; imshow(img); 
 
%b)
n = 4; 
h = size(img,1)/n; 
w = size(img,2)/n; 
k = 0; 
 
for i=1:h:size(img,1) 
    for j=1:w:size(img,2) 
        k = k+1; 
        img_patch = img( i:(i+h-1), j:(j+w-1) ); 
        %b).I
        DFT_patch = fft2(img_patch); 
        DFT_patch = fftshift(DFT_patch); 
        %b).II
        PS = real(DFT_patch).^2 + imag(DFT_patch).^2; 
        PS_patch_array(:,:,k) = PS; 
    end
end
%c)
DFT_original = fft2(img); 
DFT_original = fftshift(DFT_original);
figure
imagesc(log(abs(DFT_original)))
 
%d)
PS = real(DFT_original).^2 + imag(DFT_original).^2; 
 
%e)
PS_patch = median(PS_patch_array,3); 
 
%f)
PS_patch = imresize(PS_patch,size(PS)); 
 
%g)
PS_patch_med = medfilt2(PS_patch,[25 25]); 
 
%h)
Mask = PS_patch>3*PS_patch_med; 
 
%i)
DFT_restored = DFT_original; 
mid = [1 + size(DFT_original,1)/2,1 + size(DFT_original,2)/2]; 
D0 = 80; 
for i=1:size(DFT_original,1) 
    for j=1:size(DFT_original,2) 
        r = sqrt((i-mid(1))^2+(j-mid(2))^2); 
        %i).I
        if r<D0 
            DFT_restored(i,j) = DFT_original(i,j); 
        %i).II
        else
            if(Mask(i,j) == 1) 
                DFT_restored(i,j) = 0; 
            end
        end
    end
end
%j)
figure
imagesc(log(abs(DFT_restored)))
img_restored = (ifft2(ifftshift(DFT_restored), 'symmetric')); 
figure; imshow(img_restored);