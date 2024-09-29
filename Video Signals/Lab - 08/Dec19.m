% image stiching
clc;clear all; close all;
im1 = imread('grand_canyon_left_01.png');
im2 = imread('grand_canyon_right_01.png');

im1g = rgb2gray(im1);
im2g = rgb2gray(im2);

pts1 = detectSIFTFeatures(im1g);% DETECT KEY POINTS INSIDE THE IMAGE
pts2 = detectSIFTFeatures(im2g);


figure 
imshow(im1)
hold on
scatter(pts1.Location(:,1), pts1.Location(:,2), 5, 'filled')

figure
imshow(im2)
hold on
scatter(pts2.Location(:,1), pts2.Location(:,2), 5, 'filled')
% same scene, but shifted. There are some corresponding points
% project pixels from the left om=n the pixels on the right in an
% itelligent way
%%
%once you have keypoints, get desctiptors/features
[features1, pts1] = extractFeatures(im1g, pts1);
[features2, pts2] = extractFeatures(im2g, pts2);
%each row corresp to a desctiptor

%% now match the desctiptors
n_points = size(pts1, 1);
thr = 0.66; %decision on the match
matches = [];%closest

for i = 1:n_points
    ssd = sum((features2-features1(i,:)).^2,2);
    [ssd, sort_id] = sort(ssd);
    score = ssd(1)/ssd(2);% small difference -> ratio is 1 -> 2 simmilar features
    if score<thr
        matches = [matches; i, sort_id(1)];
    end
end
size(matches,1)
%%
X1 = pts1(matches(:,1)).Location';
X2 = pts2(matches(:,2)).Location';

figure
imshow([im2g, im1g])
line([X1(1,:);size(im1g,2)+X2(1,:)], [X1(2,:); X2(2,:)])

%% RANSAC loop (homography with outlier rejection)
%H - homography. Compute inliers->keep the largest set of inliers 
minimal_set_size = 4;
numPoints = size(X1, 2);
max_trials = 1000;
trials = 0;
best_score = 0;
thr_inliers = 5;

while(trials<max_trials)
    trials = trials + 1;
    subset = randi(numPoints, 1 ,minimal_set_size);% random subset
    
    H = solveHomography(X1(:,subset), X2(:,subset));

    Xh = H*[X1;ones(1,size(X1,2))];% homogenious
    Xh = Xh./Xh(3,:);
    d = sum((X2 - Xh(1:2,:)).^2);%difference
    
    inliers = find(d<(thr_inliers)^2);% squared threshold, because no sqrt(dist)
    score = numel(inliers);
    
    if (score>best_score)
        best_score = score;
        H_best = H;
        inliers_best = inliers;
        fprintf('Best inlier ratio: %.2f\n',score/numPoints)
    end
end

H = solveHomography(X1(:, inliers_best), X2(:, inliers_best));

figure
imshow([im2g, im1g])
line([X1(1,inliers_best);size(im1g,2)+X2(1,inliers_best)], [X1(2,inliers_best); X2(2,inliers_best)],[1 0 0], 'b')
hold on
outliers = setdiff(1:size(X1,2), inliers_best);
line([X1(1,outliers);size(im1g,2)+X2(1,outliers)], [X1(2,outliers); X2(2,outliers)], 'r')
%% stich images
image_mosaic(im1,im2,H);