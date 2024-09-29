%% histograms and histograms equalization
clc; clear all; close all;
img_gray = imread('zebra.jpg');
figure
imshow(img_gray)
%% straightforward hist (not optimized)
%[0,255], 1 bin for each value
hist_handmade = zeros(1, 256);
for i=0:255
    for ii = 1:size(img_gray,1)
        for jj = 1:size(img_gray,2)
            if(img_gray(ii,jj) == i)
                hist_handmade(i+1)=hist_handmade(i+1)+1;
            end
        end
    end
end

figure
bar(0:255, hist_handmade)
xlim([0,255])
title('handmade histogram')
% from hist you can see the distribution of pixels by value
% white stripes are darker than the background
%% smarter way
hist_handmade = zeros(1, 256);
for i =0:255
    hist_handmade(i+1) = sum(sum(img_gray == i));% logical matrix
end
figure
bar(0:255, hist_handmade)
xlim([0,255])
title('handmade histogram')
%%
figure
%hist(double(img_gray(:)), 256);% convert to double -> as before (specify # bins)
imhist(img_gray)
%% different # bins
figure
imhist(img_gray)
title('256 bins')

figure% 0 or 1 will be put in the same bin -> smoother
imhist(img_gray, 128)
title('128 bins')

figure
imhist(img_gray, 64)% global trend, but lower intensity resolution
title('64 bins')
%%
img_bw = im2bw(img_gray, 0.5);
figure
imhist(img_bw)% global trend, but lower intensity resolution
title('bw hist')
%% histograms and iterpolation
%%
clc; clear all; close all;
img_gray = imread('zebra.jpg');

figure
subplot(1,2,1)
imshow(img_gray)
title('original image')

subplot(1,2,2)
imhist(img_gray)
title('original hist')

% downsample
img_gray_reduced = imresize(img_gray, 1/3);

figure
subplot(1,2,1)
imshow(img_gray)
title('original image')

subplot(1,2,2)
imhist(img_gray)
title('original hist')

% nn/bl/bc
img_gray_gray_nn = imresize(img_gray_reduced, 3, 'nearest');% bilinear, bicubic

figure
subplot(1,2,1)
imshow(img_gray_gray_nn)
title('x3 image (nn)')

subplot(1,2,2)
imhist(img_gray_gray_nn)
title('x3 hist (nn)')
% 1053/11. We multiply each bin by 9, because of how nn works
% with bicubic we also go outside of the original range (new components, which were not present in the original one)
%% histogram shifting
clc; clear all; close all;
img_gray_compr = imread('zebra_compressed.jpg');

figure
subplot(1,2,1)
imshow(img_gray_compr)
title('image with compressed histogram')

subplot(1,2,2)
imhist(img_gray_compr)
title('hist of the compressed image')

% shift the hist  to brighten it
img_gray_compr_bright = img_gray_compr + 230;
figure
subplot(1,2,1)
imshow(img_gray_compr_bright)
title('image with compressed histogram (bright)')

subplot(1,2,2)
imhist(img_gray_compr_bright)
title('hist of the compressed image (bright)')
% but the overall shape is not changed (contrast)

%% stretch the histogram
img_gray_double = double(img_gray_compr);
min_value = min(img_gray_double(:));
max_value = max(img_gray_double(:));

% rescaling
img_gray_mod = uint8(255*(img_gray_double-min_value)/(max_value-min_value));

figure
subplot(1,2,1)
imshow(img_gray_mod)
title('image with compressed histogram (dilated)')

subplot(1,2,2)
imhist(img_gray_mod)
title('hist of the compressed image (dilated)')
% we have spaces because of the original compression (intermediate values can't be created)
% overall shape is preserved and the range is expanded

% was it a histogram equalization?
h = imhist(img_gray_mod);
h_norm = h/numel(img_gray_mod); % integral must sum to one -> normalize
h_norm_cum = cumsum(h_norm);

figure
plot(h_norm_cum)
title('cdf')% NOT A HIST EQUALIZATION, SINCE IT IS NOT A STRAIGHT LINE
%% histogram equalization
clc, clear all; close all;
img_gray_compr = imread('zebra_compressed.jpg');

figure
imshow(img_gray_compr)
title('original image')

% straightforward
h = imhist(img_gray_compr);
h_norm = h/numel(img_gray_compr); % integral must sum to one -> normalize
h_norm_cum = cumsum(h_norm);

img_gray_equl = h_norm_cum(img_gray_compr+1); %[0, 255]->[1, 256]
img_gray_equal = uint8(round(255*img_gray_equl));

figure
imshow(img_gray_equal)
title('equalized image')

figure
imhist(img_gray_equal)
title('equalized histogram')

figure
plot(h_norm_cum)
title('cdf equalized')
%%
img_gray_equal = histeq(img_gray_compr, 60);% bins

figure
imshow(img_gray_equal)
title('equalized image')

figure
imhist(img_gray_equal)
title('equalized histogram')

h = imhist(img_gray_equal);
h_norm = h/numel(img_gray_equal); % integral must sum to one -> normalize
h_norm_cum = cumsum(h_norm);

figure
plot(h_norm_cum)
title('cdf')
%% histogram equalization to fix overexposed photo
clc; close all; clear all;

img_gray = imread('castle.jpg');

figure
imshow(img_gray)
figure
imhist(img_gray)

% equalize
img_gray_equal = histeq(img_gray, 256);% bins

figure
imshow(img_gray_equal)
figure
imhist(img_gray_equal)


% local histogram equalization
% idea - ignore clouds, focus on the castle

mask = imread('castle_binary.png');
mask = (mask==1);% to make it a logical matrix

figure
imshow(mask);

h = imhist(img_gray(mask));
figure
bar(h)
title('histogram of masked image')

h_norm  = h/sum(h);
h_norm_cum = cumsum(h_norm);

figure% cdf of the castle -> convert to a straight line
plot(h_norm_cum)
title('cdf')

img_gray_equal = h_norm_cum(img_gray+1);% only what was obtained with the mask
img_gray_equal = uint8(round(255*img_gray_equal));

figure
imshow(img_gray_equal);

% to recover the original info of the sky
img_gray_equal_composed = img_gray_equal.*(uint8(mask))+img_gray.*(uint8(1-mask));
% mask = 1 -> equalied castle, mask = 0 -> use the sky from the original sky
figure
imshow(img_gray_equal_composed); % YOU CAN EQUALIZE JUST AN OBJECT, THUS, USING THE WHOLE RANGE FOR EQUALIZATION
%% histogram matching
clc; close all; clear all;
% easy to separate castle from the sky -> equalize just the castle
img_gray = imread('castle.jpg');

figure
imshow(img_gray)
figure
imhist(img_gray)

h = imhist(img_gray);
figure
bar(h)
title('histogram of image')

h_norm  = h/sum(h);
F = cumsum(h_norm);

figure
plot(F)
title('original cdf')

G = F;% first 150 bins correspond to the castle
G(1:150) = linspace(0,G(150), 150); %uniformly disrtibute th castle
hold on
plot(G, 'r--')

img_out = uint8(zeros(size(img_gray)));

for i = 1:255
    j = find(G>=F(i+1),1); % indexes where G is >= F calculated in i
    img_out(img_gray == i) = j-1;
end

figure
imshow(img_out)
figure
imhist(img_out)

h = imhist(img_out);
h_norm  = h/sum(h);
F = cumsum(h_norm);

figure
plot(F)
title('matched cdf')