%% Hough transform - isolated points
clc;
clear all;
close all;

im = zeros(600, 600);
im(30, 30) = 1;
im(30, 570) = 1;
im(570, 30) = 1;
im(570, 570) = 1;
im(300, 300) = 1;

imshow(im), title('Original image');

% Hough transform
[H, T, R] = hough(im);

figure
imagesc(T, R, H);
%imshow(H, 'XData', T, 'YData', R, 'InitialMagnification','fit');
title('Hough Transform of isolated points');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
colormap('hot')
colorbar


%% Hough transform - straight lines
clc;
clear all;
close all;

im = zeros(600, 600);
im(100:480, 200) = 1;
im(100:480, 400) = 1;
im(100, 200:400) = 1;
im(480, 200:400) = 1;

figure, imshow(im), title('Rectangle');

% Hough transform
[H, T, R] = hough(im);

figure
imagesc(T, R, H);
title('Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');


figure
imagesc(T, R, log(H));
title('Log Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');

%% Hough transform - noise
clc;
clear all;
close all;

im = zeros(600, 600);
im(100:480, 200) = 1;
im(100:480, 400) = 1;
im(100, 200:400) = 1;
im(480, 200:400) = 1;

for i = 1:10000
    im( ceil(rand(1)*600), ceil(rand(1)*600) ) = 1;
end
imshow(im), title('Noise image');

% Hough transform
[H, T, R] = hough(im);

figure
imagesc(T, R, H);
title('Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');

figure
imagesc(T, R, log(H));
title('Log Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');
%% Hough transform - image
clc;
clear all;
close all;

im = imread('SEM.jpg');
figure
imshow(im), title('Original image');

edges = edge(im, 'prewitt');
figure
imshow(edges), title('Edge image');

[H, T, R] = hough(edges);

figure
imagesc(T, R, H);
title('Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');

thresh = ceil(0.5*max(H(:)));
radius = 30;
B = strel('square',radius);
max_H_loc = imdilate(H, B);
mask_max_H = (H == max_H_loc) & (H > thresh);      % Find maxima.

% Find row,col coords.
[i_max,j_max] = find(mask_max_H);

% figure
% imagesc(T, R, mask_max_H);
% xlabel('\theta'), ylabel('\rho');
% axis on, axis normal, colormap('hot');

R_max = R(i_max);
T_max = T(j_max);

hold on
plot(T_max,R_max,'s','color','white');

T_max = T_max * pi / 180;

x = linspace(1,size(im,2),10000);

mask = zeros(size(im));
for i=1:numel(T_max)

     x_p = [];
     y_p = [];
     t = T_max(i);
     y = (R_max(i) - x*cos(t) )/ sin(t);
     x_p = x(y<size(im,1) & y>0); 
     y_p = y(y<size(im,1) & y>0);
     for j=1:numel(x_p)
        mask(ceil(y_p(j)),ceil(x_p(j))) = 1;
     end
     
end
figure
imshow(mask)


I_out = uint8(zeros(size(im,1),size(im,2),3));
I_out(:,:,1) = im;
I_out(:,:,2) = im;
I_out(:,:,3) = im;

red = uint8(zeros(size(im,1),size(im,2),3));
red(:,:,1) = 255;

I_out = im.*uint8(1-mask) + red.*uint8(mask);

figure
imshow(I_out)

%% Hough lines detector with matlab function

clear all;
close all;

im = imread('SEM.jpg');
%im = imread('circuit.tif');
figure
imshow(im), title('Original image');

edges = edge(im, 'prewitt');
figure
imshow(edges), title('Edge image');


[H, T, R] = hough(edges);
figure
imagesc(T, R, H);
title('Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');

P = houghpeaks(H,50,'threshold',ceil(0.5*max(H(:))));

x = T(P(:,2));
y = R(P(:,1));
hold on
plot(x,y,'s','color','white');

lines = houghlines(edges,T,R,P,'FillGap',1000,'MinLength',20);
figure, imshow(im), hold on

for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
end
title('Detected lines')

%% IMAGE STICTHING
close all;
clear all;

im1 = imread('grand_canyon_left_01.png');
im2 = imread('grand_canyon_right_01.png');

im1g = rgb2gray(im1);
im2g = rgb2gray(im2);

% detect points
pts1  = detectSURFFeatures(im1g);
pts2  = detectSURFFeatures(im2g);

figure()
imshow(im1);
hold on
scatter(pts1.Location(:,1),pts1.Location(:,2),5,'filled');

figure()
imshow(im2);
hold on
scatter(pts2.Location(:,1),pts2.Location(:,2),5,'filled');
%%
% extract descriptors
[features1,pts1] = extractFeatures(im1g,pts1);
[features2,pts2] = extractFeatures(im2g,pts2);

% match descriptors    
n_points = size(pts1,1);
thr = 0.66;
matches = [];
for i = 1:n_points
    ssd = sum((features2 - features1(i,:)).^2,2);
    [ssd sort_id] = sort(ssd);
    score = ssd(1)/ssd(2);
    if score < thr
       matches = [matches; i sort_id(1)]; 
    end
end

X1 = pts1(matches(:, 1)).Location'; 
X2 = pts2(matches(:, 2)).Location';

figure()
imshow([im1g im2g])
line([X1(1,:);size(im1g,2)+X2(1,:)],[X1(2,:);X2(2,:)]);
%%
% estimate H with outliers rejection
minimal_set_size = 4;      % 4 points homography estimation
numPoints = size(X1, 2);
max_trials = 1000;
trials = 0;
best_score = 0;
thr_inliers = 5; %pixels

%rng(73);
while (trials < max_trials)

    trials = trials + 1;
    % generate hypotesis from random minimal set
    subset = randi(numPoints,1,minimal_set_size);

    H = solveHomography( X1(:,subset), X2(:,subset) );

    % score hypotesis
    Xh = H * [X1; ones(1,size(X1,2))];
    Xh = Xh./Xh(3,:);
    d = sum((X2 - Xh(1:2,:)).^2);
    inliers = find(d < (thr_inliers^2));
    score = numel(inliers);

    if (score > best_score)
      best_score = score;
      H_best = H;
      inliers_best = inliers;
      fprintf('Best inlier ratio: %.2f\n', score / numPoints);  
    end
    
end
% select best hypotesis
inliers = inliers_best;
H = solveHomography( X1(:,inliers), X2(:,inliers) );

figure()
imshow([im1g im2g])
line([X1(1,inliers);size(im1g,2)+X2(1,inliers)],[X1(2,inliers);X2(2,inliers)],'Color',[0 0 1]);
hold on
line([X1(1,setdiff(1:size(X1,2),inliers));size(im1g,2)+X2(1,setdiff(1:size(X1,2),inliers))],[X1(2,setdiff(1:size(X1,2),inliers));X2(2,setdiff(1:size(X1,2),inliers))],'Color',[1 0 0]);

%stitching
image_mosaic(im1, im2, H);