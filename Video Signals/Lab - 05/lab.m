%% Morphological operators on binary images (erosion/dilation)
clc;
clear all;
close all;

A = zeros(600,600);
A(250:350,250:350) = 1;
A = im2bw(A, 0.5);
figure 
imshow(A); title('Original image (100x100 square)');

B = strel('square', 80);
figure
imshow(B.Neighborhood); title('80x80 morphological structuring element');

A_dil_B = imdilate(A,B);
figure
imshow(A_dil_B);
title('Dilated image using square 80x80 structuring element');

A_er_B = imerode(A,B);
figure
imshow(A_er_B);
title('Eroded image using square 80x80 structuring element');

B_ = strel('rectangle', [100 10]);
figure
imshow(B_.Neighborhood); title('100x10 morphological structuring element');
A_er_B_ = imerode(A,B_);
figure
imshow(A_er_B_);
title('Eroded image using rect 100x10 structuring element');

%% Morphological operators on real-world binary images
clc;
clear all;
close all;

A = im2bw(imread('Lena_grayscale.bmp'), 0.6);
figure
subplot(1,4,1)
imshow(A); title('Original image');

B = strel('square', 5);
% B = strel('disk', 3, 0);
% B = strel('line', 9, 45);
% B = strel('line', 9, 135);
% B = strel('line', 100, 45);
% B = strel('line', 100, 135);
% B = strel('arbitrary', [1 1 1 1; 1 1 1 0; 1 1 0 0; 1 0 0 0]);

subplot(1,4,2)
imshow(B.Neighborhood); title('Structuring element');


A_dil_B = imdilate(A, B);
subplot(1,4,3)
imshow(A_dil_B); title('Dilated image using structuring element');

A_er_B = imerode(A, B);
subplot(1,4,4)
imshow(A_er_B); title('Eroded image using structuring element');

A_er2_B = ~imdilate(~A, B);
figure
imshow(A_er2_B); title('Complementary dilated image using structuring element on complementary image');
%same result if B is symmetrical

A_er_dil_B = imdilate(A_er_B, B);
figure
imshow(A_er_dil_B);
title('Eroded and dilated image using structuring element');

%% Erosion, then dilation (opening) to eliminate details under a given size
clc;
clear all;
close all;

A = zeros(600,600);
% rand() gives uniformly distributed values between 0 and 1
for i=1:100
    % 10x10 square
    A((1:10) + floor(rand(1)*590), (1:10) + floor(rand(1)*590)) = 1;
end

for i=1:20
    % 30x30 square
    A((1:30) + floor(rand(1)*570), (1:30) + floor(rand(1)*570)) = 1;
end


A = im2bw(A, 0.5);
figure
imshow(A); title('Original image');

B = strel('square', 30);

A_er_B = imerode(A, B);
figure
imshow(A_er_B);
title('Eroded image using 30x30 square structuring element');

A_er_dil_B = imdilate(A_er_B, B);
figure
imshow(A_er_dil_B);
title('Dilated image using 30x30 square structuring element on eroded image');

A_open_B = imopen(A,B);
figure
imshow(A_open_B);
title('Opening using 30x30 square structuring element');


%% Dilation to improve the legibility of a text
clc;
clear all;
close all;

% Here black = 1 (text) and white = 0 (no text)
% color encoding is black = 0, white = 1 so we need the negative image 
A = im2bw(imread('Zisserman_page_thr_140_full.jpg'), 0.5);
figure
imshow(A); title('Low legibility text');
A = ~A;

B = zeros(5);
B(round(5/2),:) = 1;
B(:,round(5/2)) = 1;
B = strel('arbitrary', B);
%B = strel('disk', 3);

A_dil_B = imdilate(A, B);
figure
imshow(~A_dil_B); title('Dilated text using default structuring element');

%% Opening and closing for bw image filtering
clc;
clear all;
close all;

A = double(imread('fingerprint.gif'));
imshow(A), title('Noisy fingerprint');

B = strel('arbitrary', [1 1 1; 1 1 1; 1 1 1]);
%B = strel('square', 3);

% opening (erosion followed by dilation)
A_op_B = imopen(A, B);
figure, imshow(A_op_B), title('Fingerprint + opening');

% closing (dilation followed by erosion)
A_cl_B = imclose(A_op_B, B);
figure, imshow(A_cl_B), title('Final filtered fingerprint');

%% Opening and closing to remove salt and pepper noise
I = imread('Lena_grayscale.bmp');
I_bw = im2bw(I,0.5);
figure()
imshow(I_bw)
title('original image')
I_sp = imnoise(255*uint8(I_bw),'salt & pepper',0.1);
I_sp = im2bw(I_sp,0.5);
figure()
imshow(I_sp)
title('original image + salt and pepper')
% Opening should eliminate small details ("salt")
% Closing should fill small holes ("pepper")
SE = strel('arbitrary',[0,1,0;1,1,1;0,1,0]);
%SE = strel('square',3);
Im_open = imopen(I_sp,SE);
figure, imshow(Im_open);
Im_open_close = imclose(Im_open,SE);
figure, imshow(Im_open_close);
title('restored image')


%% Hit-or-miss operator
clc;
clear all;
close all;

A = zeros(600,600);
A(100:500, 100:260) = 1;  % rectangle
A(120:200, 400:480) = 1;  % small square
A(350:449, 420:519) = 1;  % big square
A = im2bw(A, 0.5);
figure
imshow(A); title('Original image');

B = strel('square',100);
figure
imshow(B.Neighborhood); title('Template')

A_er_B = imerode(A,B);
figure; imshow(A_er_B); title('Image eroded with template');

wb_matrix = ones(120);
wb_matrix(10:110, 10:110) = 0;
WB = strel('arbitrary',wb_matrix);
figure
imshow(WB.Neighborhood); title('Local background')

A_c_er_WB = imerode(~A,WB);
figure; imshow(A_c_er_WB); title('Complementary Image eroded with WB');

hit_image = A_er_B & A_c_er_WB;
figure; imshow(hit_image); title('Hit Image');

[x,y] = find(hit_image)

hit_image_2 = bwhitmiss(A,B,WB);
figure; imshow(hit_image_2); title('Hit Image w Matlab function');

%% Boundary extraction
clc;
clear all;
close all;

im = imread('coins.png');
A = im2bw(im, 0.37);
imshow(A), title('Original image');

% border image
B = strel('arbitrary', [1 1 1; 1 1 1; 1 1 1]);  % 8-connected
%B = strel('arbitrary', [0 1 0; 1 1 1; 0 1 0]);  % 4-connected

A_borders = A - imerode(A, B);
figure, imshow(A_borders), title('Image borders');

A_borders = bwperim(A);
figure, imshow(A_borders), title('Image borders (Matlab function)');

%% Region filling
clc
close all
clear all

A=[ 0 0 0 0 0 0 0
    0 0 1 1 0 0 0
    0 1 0 0 1 0 0
    0 1 0 0 1 0 0
    0 0 1 0 1 0 0
    0 0 1 0 1 0 0
    0 1 0 0 0 1 0
    0 1 0 0 0 1 0
    0 1 1 1 1 0 0
    0 0 0 0 0 0 0];

A = A>0.5;
A_comp = ~A;

x0 = zeros(size(A));
x0(3,3) = 1;
%x0(7,5) = 1;

B = [0 1 0
     1 1 1
     0 1 0];

figure;
subplot(1,2,1)
imshow(A)
title('original image')
subplot(1,2,2)
imshow(x0)
title('region filling step 0')
pause(1)
x1 = imdilate(x0,B);
x1 = x1 & A_comp;
imshow(x1)
title('region filling step 1')
pause(1)
i = 1;
while (~isequal(x0,x1))
    i = i+1;
    x0 = x1;
    x1 = imdilate(x0,B);
    x1 = x1 & A_comp;
    subplot(1,2,2)
    imshow(x1)
    title(sprintf('region filling step %d',i))
    pause(1)
end
x1 = x1 | A;
subplot(1,2,2)
imshow(x1)
title('region filling final result')


%% Advanced binary operators
clc;
clear all;
close all;

A = imread('circles.png');
figure, imshow(A), title('Original image');

% convex hull
CH = bwconvhull(A);
figure, imshow(CH), title('Convex Hull image');

A = imread('circbw.tif');
figure, imshow(A), title('Original image');

% label connected components
labels = bwlabel(A, 4); % 8-connected regions
figure, imagesc(labels), axis off, title('Connected regions');

% thinning
thin = bwmorph(A, 'thin', Inf);  % thinning operator
figure, imshow(thin), title('Thinning');

% skeletonization
skel = bwmorph(A, 'skel', Inf);   % skeletonization operator
% preserving some info about object shape
figure, imshow(skel), title('Skeleton');

%% Grayscale dilation and erosion
clc;
clear all;
close all;

A = imread('cameraman.tif');
imshow(A), title('Original')

B = strel('disk', 5);
% B = strel('disk', 3);

% dilation: local maximum
A_dil_B = imdilate(A, B);
figure, imshow(A_dil_B), title('Dilated');

% erosion: local minimum
A_er_B = imerode(A, B);
figure, imshow(A_er_B), title('Eroded');

% morphological gradient
A_morph_grad = A_dil_B - A_er_B;
figure, imshow(A_morph_grad), title('Morphological gradient');
