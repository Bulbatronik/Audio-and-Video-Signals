%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Introduction to Matlab                            
% Audio Signals course
% 2022
% Credits to Federico Borra
% Exercise 2
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions
% Create a function that takes a matrix X as input and
%   -If the input X is a vector return -1
%   -Multiplies each row by the row index
%   -Set to 0 all the elements of the main diagonal
%   -Compute the square elementwise
%   -Sum the columns
%   -Compute and returns the sum of the square of the resulting column vector

function ret = exercise2(X)

% numel returns the NUMber of ELements of the argument (check help)
if(numel(X) == length(X))  % is a vector
    ret = -1;
    return
end

s = size(X);
assert(length(s) == 2);  % assert that is a 2 dimensional array (a matrix)

nrows = s(1);
for irow = 1:nrows
    X(irow, :) = X(irow, :) * irow;
    X(irow, irow) = 0;
end

% note that using sum(x, 2) you specify the dimension. See help sum
ret = sum(sum((X.^2), 2).^2);

end