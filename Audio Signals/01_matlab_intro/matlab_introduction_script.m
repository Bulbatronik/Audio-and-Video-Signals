%% Audio Signals 2022/23 -  Introduction to Matlab
% Credits: Bruno Di Giorgi and Federico Borra
% 
% - updated Mirco Pezzoli
%% Scalar variables

clear

% this is a comment
a = 2;
b = pi;  % constant

% operations on scalar variables
c = a + b;  % also try: -, *, /, ^

% functions on scalar variables
d = sin(a);  % also try: cos, tan, round, floor, ceil, sqrt, log, log2, pow2, ...

% the semicolon at the end of a line suppresses the system output of that line
% the next line prints the values of the variables
a, b, c, d
%%
% without the semicolon, the result of the line is immediately shown 
e = sqrt(a)
%% 
% The |who| statement shows the list of variables currently in the workspace

who
%% 
% The |whos| statement shows the detailed list currently in the workspace

whos
%% 
% |clear| removes all variables from the workspace

clear
who  % The workspace is now empty
%% 
% Typing |help [function_name]| in the command window displays the help for 
% the function

help cos
%% 
% This was a very short help. Type |doc [function_name]| in order to get a more 
% extensive help with examples and figures.

doc cos
%% Matrices

% space or comma separate elements in a row
a = [1 2 3 4];

% get the size of the dimensions
b = size(a);  % returns a vector of sizes
c = length(a);  % returns the largest dimension (useful for vectors)
a, b, c
%%
% semicolon defines a new row
a = [1 2; 3, 4; 5, 6];

% get the size of the dimensions
b = size(a);  % returns a vector of sizes
c = length(a);  % returns the largest dimension (not very useful with a matrix)
a, b, c
%%
% the colon operator creates a vector of equally spaced values given the step
% start:step:end
a = 1:2:6;  % end is not required to be in the vector 
b = 1:6;  % default step=1
a, b
%%
% conversely, linspace(start, end, N) creates a vector of equally spaced values given the number of elements N
a = linspace(1, 6, 4);  % start and end are included
a
%%
% some helpful functions to create matrices
a = zeros(1, 3);
b = ones(1, 3);
c = rand(1, 3);

% with just one argument, they default to 2-dim
d = zeros(3);

% reshape(X, [M, N]) returns an m-by-N matrix with the elements of x taken columnwise 
e = reshape(1:6, [2, 3]);

% horizontal ',' and vertical ';' stack
f = [a, a];  % equivalent to i = horzcat(a, a)
g = [a; a];  % equivalent to l = vertcat(a, a)

% repmat(X, [M, N]) creates an m-by-n tiling of copies of x
h = repmat(1:3, [2, 2]);  % 2 equal rows

a, b, c, d, e, f, g, h
%% 
% Operations on matrices

a = reshape(1:4, [2, 2]);

% single quote ' returns the transposed matrix
at = a';  % equivalent to at = transpose(a)

b = eye(2);  % the identity matrix of given size
c = a + b;

% vectorized versions of scalar operations. Applied element-wise
d = sin(a);  % also try cos, tan, round, floor, ceil, sqrt, log, log2, pow2, ...

a, at, b, c, d
%%
% aggregators: sum(x), prod(x), mean(x), std(x), var(x)
% operates along (read "varying the index of") the first dimension of x having size > 1
b = sum(a);  % also try sum, prod, mean, std, var

% in order to get the sum of all elements, first unroll the array with x(:)
c = a(:);  % unroll
d = sum(c);  % ...then sum
b, c, d
%%
% matrix multiplication
a = reshape(1:4, [2, 2]);
b = eye(2);

c = a * b;

% element-wise multiplication
d = a .* b;  % the dot-operators are element-wise: also try division ./ and power .^
a, b, c, d
%% Indexing

a = 1:4;
% 2 Matlab conventions: 
% - Matlab is 1-indexed (first element of a vector has index equal to 1)
% - uses round brackets for accessing matrix items
first = a(1);
third = a(3);
last = a(end);
a, first, third, last
%%
a = reshape(1:6, [2, 3]);
b = a(1, 1);  % 1-st row, 1-st column
c = a(2, 1);  % 2-nd row, 1-st column
d = a(end, end);  % last row, last column
a, b, c, d 
%% 
% Slicing

a = reshape(1:6, [2, 3]);
b = a(2, :);  % extract the second row (':' means 'take all elements in that dimension')
c = a(:, 2);  % extract the second column
a, b, c
%% 
% Fancy indexing

a = reshape(1:16, [4, 4]);

% Index an array using vectors
a_rows = [1, 3];  % take these rows
a_cols = [2, 3, 4];  % ...and these columns
b = a(a_rows, a_cols);

a, b
%% 
% *Relational and Logical operations*

% Relational operators (>, <, >=, <=, ==, ~=)
a = (2 < 3);
b = (2 == 3);

% Connect multiple conditions with the logical operators (&, |, ~).
c = a | b;
a, b, c
%%
a = reshape(-3:12, [4, 4]);
threshold = 0;

% vectorized version of relational operators (element-wise)
b = a < threshold;  % the output is logical array of 0s (false) and 1s (true)
a, b
%%
% Logical arrays can be used for indexing
c = a;
c(b) = threshold;  % clip values less than the threshold. 

% Short, but still readable, version: a(a < threshold) = threshold

% Connect multiple conditions with the logical operators (&, |, ~).
d = a;
d((d > 2) & (d < 5)) = pi;
c, d
%% Language syntax
%% 
% Conditional execution

% The if statement is used to execute part of the code only if a condition is matched
% keywords: if, elseif, else, end
a = 3;
threshold = 2;
if(a > threshold)
    fprintf('wow, %.2f is greater than %.2f\n', a, threshold);  % '%.2f' is a placeholder for a variable 
elseif(a == threshold)
    fprintf('wow, %.2f is equal than %.2f\n', a, threshold);
else
    fprintf('sorry, %.2f is less than %.2f\n', a, threshold);
end
%% 
% Loops

start = 1;
step = 2;
stop = 8;

for i = start:step:stop
    fprintf('iteration %.0f\n', i);
end

i = start;
while(i <= stop)
    fprintf('iteration %.0f\n', i);
    i = i + step;
end
%% The working directory
%% 
% When you do not specify a path to a file, MATLAB looks for the file in the 
% current folder |pwd| or on the search path |path|

pwd  % print working directory
%% 
% We are going to use other files, so let's make sure that the working directory 
% is the one containing the other files

% change mydir to the 03_matlab_intro folder on your computer
mydir = '/Users/mircopezzoli/Documents/dottorato/Didattica/Audio Signals/2022/03_matlab_intro';
cd(mydir)
%% Creating a function
% A script is a .m file that is used to write a program that performs a complex 
% task. It is usually executed starting from a clear workspace.
% 
% A function is a .m file that is used to write a program that processes  information, 
% so it receives inputs (parameters) and it (usually) returns outputs (results).
% 
% Golden rule: if the code you are writing will be used more than once, write 
% a function. Otherwise, write a script.
% 
% The next lines create a sinusoidal signal, its sample axis and its time axis, 
% given the sampling frequeny, the frequency of the signal and its duration. This 
% is done by using the function |[n, t, x] = create_sinusoid(duration, Fs, f).|
% 
% The .m file with the function must have the same name of the function |create_sinusoid.m|

% lets sample a continuous sinusoid with sampling frequency Fs
Fs = 5;  % sampling frequency [Hz]
duration = 3;
Ts = 1 / Fs;  % sampling period

% create the signal
f = 1;  % frequency of the sinusoid [Hz]

[n, t, x] = create_sinusoid(duration, Fs, f)
%% 
%% Plotting functions
% 
% 
% The command |plot(x,y)| plots vector Y versus vector X

figure();
plot(n, x);

% print some other information in the figure
xlabel('$n$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$x(n)$', 'Interpreter', 'latex', 'FontSize', 18);
title(sprintf('%.0f Hz sinusoid, sampled at %.0f Hz', f, Fs));
%% 
% The command |stem(X,Y)| plots the data sequence Y at the values specified 
% in X

figure();
stem(n, x);

% print some other information in the figure
xlabel('$n$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$x(n)$', 'Interpreter', 'latex', 'FontSize', 18);
title(sprintf('%.0f Hz sinusoid, sampled at %.0f Hz', f, Fs));
%% Multiple plot

% Let's increase the resolution of the signal sampling more frequently
duration = 3;  % [s]
f = 1;  % frequency of the signal [Hz]

Fs = 5;  % sampling frequency [Hz]
Fs2 = 4 * Fs; % we get 4 times more samples

[n1, t1, x1] = create_sinusoid(duration, Fs, f);  
[n2, t2, x2] = create_sinusoid(duration, Fs2, f);

figure();
% this time, in order to compare the two sequences, 
% I will plot them over the time axis instead of the sample axis
% what happens plotting over the sample axis? Try substituting t1 with n1, t1 with n2
plot(t1, x1, 'DisplayName', sprintf('sampled at %.0f Hz', Fs));
hold on  % holds the current plot
stem(t2, x2, 'DisplayName', sprintf('sampled at %.0f Hz', Fs2));
hold off  % returns to the default mode whereby plot commands erase the previous plots
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 18);

% show the legend
legend('show');
%% 
% Subplots

figure();
subplot(2, 1, 1);
plot(n1, x1);
title(sprintf('%.0f Hz sinusoid, sampled at %.0f Hz', f, Fs));
subplot(2, 1, 2);
stem(n2, x2);
title(sprintf('%.0f Hz sinusoid, sampled at %.0f Hz', f, Fs2));
%% 
% Recent Matlab versions (*R2019b and higher*) provide handy multiplot functions 

% We can use the tiledlayout function to define a figure with multiple
% plots
figure();
t = tiledlayout('flow'); % The flow option automatically set the subplots
nexttile                % The nexttile function calls a subplot in the figure
plot(n1, x1);
title(sprintf('%.0f Hz sinusoid, sampled at %.0f Hz', f, Fs));
nexttile
stem(n2, x2);
title(sprintf('%.0f Hz sinusoid, sampled at %.0f Hz', f, Fs2));
title(t, 'Sinusoids at different sampling rate') % We can easily set a title for the whole figure