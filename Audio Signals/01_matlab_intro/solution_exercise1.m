%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Introduction to Matlab                            
% Audio Signals course
% 2022
% Credits to Federico Borra
% Exercise 1
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Vectors
% Create a vector x of 6 elements from 2 to 12, with constant step 2. 
x = 2:2:12;

% Compute the sum of the square of its elements
s = sum(x.^2);

% Create a 2x3 matrix X1 with the elements of x
X1 = reshape(x, [2, 3]);

% Create a 2x6 matrix X2 where the rows are copies of x
X2 = repmat(x, [2, 1]);

%Check that you can multiply X1 transposed and X2
size_X1T = size(X1');
size_X2 = size(X2);
assert(size_X1T(2) == size_X2(1));

% assertion is a quick way, while coding, to check that something is as expected.
% Compute the multiplication
X1' * X2