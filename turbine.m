%extract data from the file
%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data1.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y);

% Print out some data points
fprintf('Plotting Data ...\n')
data = load('turbine file.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples
plotData(X,y)
hold on; 
%make X matrix
X=[X X.^2];
X = [ones(m, 1) X];
%plot hypothesis over it to show that its underoot func
theta = normalEqn(X, y)
predictions=X*theta;
plot(X(:, 2),predictions);
%predict value at these points
 price=0;
 val_1=0.8680968; 
 val_2=0;
predict1 = [1, val_1,val_1.^2;] *theta;
fprintf('For efficiency = 0.8680968, we predict an efficiency of %f\n',...
    predict1);
experimental_value=0.248817 %for 0.868 tsr
fprintf('approx value for 0.8680968 expected is 0.249 \n')
predict2 = [1, val_2,val_2.^2;] * theta;
difference=(predict1-experimental_value)/experimental_value;
accuracy=100-difference