%% morphological operators
clc; clear; close all

A = zeros(600,600);% synthetic image
A(250:350, 250:350) = 1;

%A =(A==1); %LOGICAL
A = im2bw(A, 0.5);

figure
imshow(A)
title("original image (100x100 square)")
%%
% structuring element generation
B = strel('square',80);
figure
imshow(B.Neighborhood)
title('80x80 morphological structuring element')
% now let's use it
%% delation (at leas one pixel -> expand)
A_dilll_B = imdilate(A, B);
figure
imshow(A_dilll_B)
title('Dilated image using 80x80 square')
% square is bigger (211 - 390). As long as we have a pixel overlap ->
% enlarge by storing one in the center 
%% erosion (all -> squeze)
A_er_B = imerode(A,B);
figure
imshow(A_er_B)
title('Eroded image using 80x80 square')
% square is smaller (289 - 311) As long as we have full overlap -> strore the
% center of the overlap
%% new struct element
B = strel('rectangle', [100,10]);
figure
imshow(B.Neighborhood)
title('100x10 morphological structuring element')

A_er_B = imerode(A,B);% line (where the vertical line fits)
figure
imshow(A_er_B)
title('Eroded image using 100x10 rectangle')
%% square elem
clc; clear, close all;

A = im2bw(imread('Lena_grayscale.bmp'),0.6);

figure
subplot(1, 4, 1)
imshow(A);
title('original image')

B = strel('square', 5);
subplot(1,4,2)
imshow(B.Neighborhood)
title('structuring element')

A_dil_B = imdilate(A,B);
subplot(1,4,3)
imshow(A_dil_B)
title('dilated image')

A_er_B = imerode(A,B);
subplot(1,4,4)
imshow(A_er_B)
title('eroded image')
% dilation removed some background noise
%% disc
clc; clear, close all;

A = im2bw(imread('Lena_grayscale.bmp'),0.6);

figure
subplot(1, 4, 1)
imshow(A);
title('original image')

B = strel('disk', 3, 0);
subplot(1,4,2)
imshow(B.Neighborhood)
title('structuring element')

A_dil_B = imdilate(A,B);
subplot(1,4,3)
imshow(A_dil_B)
title('dilated image')

A_er_B = imerode(A,B);
subplot(1,4,4)
imshow(A_er_B)
title('eroded image')
% now the result is differint. Struct element changes how we are
% considering connections
%% line
clc; clear, close all;

A = im2bw(imread('Lena_grayscale.bmp'),0.6);

figure
subplot(1, 4, 1)
imshow(A);
title('original image')

%B = strel('line', 9, 45);
B = strel('line', 9, 135);
B = strel('line', 100, 45);

subplot(1,4,2)
imshow(B.Neighborhood)
title('structuring element')

A_dil_B = imdilate(A,B);
subplot(1,4,3)
imshow(A_dil_B)
title('dilated image')

A_er_B = imerode(A,B);
subplot(1,4,4)
imshow(A_er_B)
title('eroded image')
% care about 45 deg lines
%% arbitrary
clc; clear, close all;

A = im2bw(imread('Lena_grayscale.bmp'),0.6);

figure
subplot(1, 4, 1)
imshow(A);
title('original image')


B = strel('arbitrary', [1,1,1,1;1,1,1,0;1,1,0,0;1,0,0,0]);

subplot(1,4,2)
imshow(B.Neighborhood)
title('structuring element')

A_dil_B = imdilate(A,B);
subplot(1,4,3)
imshow(A_dil_B)
title('dilated image')

A_er_B = imerode(A,B);
subplot(1,4,4)
imshow(A_er_B)
title('eroded image')
%%
not_A = ~A;
figure
imshow(not_A)

A_dil2_B = imdilate(A,B);
figure
imshow(A_dil2_B)
title('negative of dilated image')

A_er2_B = ~imerode(~A,B);
% ~imdilate(A,B)
figure
imshow(A_er2_B)
title('negative of eroded image')
% struct elem is symmetric -> can perform one from another  throught the ~
%%
close all

figure
imshow(A)
title('original image')

figure
imshow(A_dil_B)
title('dilated image')

figure
imshow(A_er_B)
title('eroded image')

A_er_dill_B = imdilate(A_er_B, B);% opening!
figure
imshow(A_er_dill_B)
title('image eroded and dilated')

A_open_B = imopen(A, B);
figure
imshow(A_open_B)
title('open')
%%
clc; close all, clear;

A = zeros(600, 600);
% let's add 100 10x10 squares ranomly ristributed
for i = 1:100
    A((1:10)+floor(rand(1)*590),(1:10)+floor(rand(1)*590)) = 1;%(1:10) is a pad (space from the border). 
    % To make sure everyting is in the picture
end

for i = 1:20
    %30x30 square
    A((1:30)+floor(rand(1)*590),(1:30)+floor(rand(1)*570)) = 1;%(1:30) is a size.
    % To make sure everyting is in the picture, random value is generate up to 590. 
    % 
end

A = im2bw(A, 0.5);
figure
imshow(A)
title('original image')

B = strel('square', 30);

A_er_B = imerode(A,B);
figure
imshow(A_er_B)
title('after erosion')

A_dil_er_B = imdilate(A_er_B,B);
figure
imshow(A_dil_er_B)
title('after dilation+erosion')% ony large squares
%%
A_open_B = imopen(A,B);
figure
imshow(A_open_B)
title('after opening')% ony large squares
%%
clc; clear; close all;

A = im2bw(imread('Zisserman_page_thr_140_full.jpg'),0.5);

figure % bad scan of a page
imshow(A)

A = ~A; % invert
figure
imshow(A)

%B = strel('disk', 3);
%figure
%imshow(B.Neighborhood)

s = 5;% cross 5x5
B = zeros(s);
B(round(s/2),:) = 1;
B(:, round(s/2)) = 1;
B = strel('arbitrary', B);

figure
imshow(B.Neighborhood)

A_dil_B = ~imdilate(A,B);% invert to go back

figure
imshow(A_dil_B)
%%
clc; close all; clear;

A = double(imread('fingerprint.gif'));

figure %moisy fingerprint 
imshow(A)

B = strel('square', 3);

A_op_B = imopen(A,B);
figure
imshow(A_op_B)
title('opening') % noise is removed, but we have some black dots inside the stripes -> do something
% symmetrical eliment -> use closing(erosion from delation)

A_cl_op_B = imclose(A_op_B,B);
figure
imshow(A_cl_op_B)
title('closing+opening')% small gaps and black dots are removed
%% hit or miss 
clc; clear; close all

A = zeros(600, 600);
A(100:500,100:250) = 1;% rectangle
A(120:200,400:480) = 1;% small square
A(350:449,420:519) = 1;% big square

A = im2bw(A, 0.5);
figure
imshow(A)
title('original image')
% we want to locate just these objects (black square for now only)

% template
B = strel('square', 100);
figure
imshow(B.Neighborhood)
title('template')

%local background around the object
wb_matrix = ones(120);
wb_matrix(10:110,10:110) = 0;
WB = strel('arbitrary', wb_matrix);

figure
imshow(WB.Neighborhood)
title('lcal background')

% H or M
A_er_B = imerode(A,B);
figure
imshow(A_er_B)
title('image eroded')% there is a point in the right hand part (target)

A_c_er_WB = imerode(~A, WB);
figure
imshow(A_c_er_WB)
title('complementary image eroded with local background')% there is a point in the right hand part (target)

% combine two information
hit_image = A_er_B & A_c_er_WB;
figure
imshow(hit_image)
title('hit image')

% small square location
[r, c] = find(hit_image);%non zero value

% the real matlab function
hit_miss = bwhitmiss(A,B,WB);% image, template, local background
figure
imshow(hit_miss)
title('hit image (MATLAB)')
% EVERYTHING MUST BE ISOLATED IN ORDER TO WORK
%% Boundry extraction (orig - eroded)
clc; clear;close all;

A = im2bw(imread('coins.png'),0.37);

figure
imshow(A)

% extract each boundary
B = strel('arbitrary', [111;111;111]);%8 point connection

A_borders = A-imerode(A, B);
figure
imshow(A_borders)
title('borders 8 point connected')

B = strel('arbitrary', [010;111;010]);%4 point connection

A_borders = A-imerode(A, B);
figure
imshow(A_borders)
title('borders 4 point connected')

% in matlab
A_borders = bwperim(A, 4); % specify how they are connected
figure
imshow(A_borders)
title('borders 4 point connected')
%% region filling
clc; close all; clear;

A = [0 0 0 0 0 0 0% shape of one (1)
     0 0 1 1 0 0 0
     0 1 0 0 1 0 0
     0 1 0 0 1 0 0
     0 0 1 0 1 0 0
     0 0 1 0 1 0 0
     0 1 0 0 0 1 0
     0 1 0 0 0 1 0
     0 1 1 1 1 0 0
     0 0 0 0 0 0 0];

A = im2bw(A, 0.5);
figure
imshow(A)

A_compl = ~A;

x0 = zeros(size(A));% startinng element (inside 1)
x0(3, 3) = 1;% where a user clicks

B = [0 1 0; 1 1 1; 0 1 0];% cross
%B = [1 1 1; 1 1 1; 1 1 1];% square - won't work, 4 point connected points -> modify the object
B = strel('arbitrary', B);

figure
subplot(1,2,1)
imshow(A)
title('original image')

subplot(1,2,2)
imshow(x0)
title('region filling step 0')

pause(1)
% perform a dilation
x1 = imdilate(x0, B);
x1 = x1 & A_compl;
imshow(x1);
title('region filling step 1')

i = 1;
while (~isequal(x0,x1)) % equal -> stop
    i = i+1;
    x0 = x1;
    x1 = imdilate(x0,B);
    x1 = x1 & A_compl;

    pause(1)
    imshow(x1)
    title(sprintf('region fililling step %d', i))
end

x1 = x1 | A;
imshow(x1)
title('final result')
%% Advanced binary operations
clc;clear all;close;

%A = imread('circles.png');
A = im2bw(imread('coins.png'),0.37);

figure
imshow(A)
title('original image')

%convexhull
CH = bwconvhull(A);
figure
imshow(CH)% smallest convex object contained in the original image
title('convexhull')

%connected regions
labels = bwlabel(A,4);
figure
imagesc(labels)
title('connected regions')% label to each object

% thinning
A = imread('circles.png');% better seen on this
thin = bwmorph(A,'thin', Inf);
figure
imshow(thin)
%% morph operatopns on grayscale images
clc; clear; close all;

A = imread('cameraman.tif');
imshow(A)
title('original image')

B = strel('disk', 5);

A_dil_B = imdilate(A, B);
figure
imshow(A_dil_B)
title('dilated image')
% instead of checking for intersection, you compute the maximum sliding

A_er_B = imerode(A, B);
figure
imshow(A_er_B)
title('eroted image')
% instead of checking for overlap, you compute the minimum sliding
%% morphological gradient

A_morph_grad = A_dil_B - A_er_B;
figure
%imagesc(A_morph_grad)
imshow(A_morph_grad)
title('gradient')