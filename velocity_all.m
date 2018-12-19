%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');
%% Load Data
data_vel_1 = load('vel_1.txt');
vel_1=4.6;
tsr_1 = data_vel_1(:, 1);
efficiency_1 = data_vel_1(:, 2);
m_1 = length(efficiency_1);

data_vel_2 = load('vel_2.txt');
vel_2=6.2;
tsr_2 = data_vel_2(:, 1);
efficiency_2 = data_vel_2(:, 2);
m_2 = length(efficiency_2);

data_vel_3 = load('vel_3.txt');
vel_3=7.4;
tsr_3 = data_vel_3(:, 1);
efficiency_3 = data_vel_3(:, 2);
m_3 = length(efficiency_3);

data_vel_4 = load('vel_4.txt');
vel_4=9.2;
tsr_4 = data_vel_4(:, 1);
efficiency_4 = data_vel_4(:, 2);
m_4 = length(efficiency_4);

data_vel_5 = load('vel_5.txt');
vel_5=3.2;
tsr_5=data_vel_5(:,1);
efficiency_5=data_vel_5(:,2);
m_5 = length(efficiency_5);

data_vel_6 = load('vel_6.txt');
vel_6=6.5;
tsr_6=data_vel_6(:,1);
efficiency_6=data_vel_6(:,2);
m_6 = length(efficiency_6);

fprintf('plotting data ...\n');

subplot(3,2,1)
plot(tsr_1, efficiency_1,'rx','MarkerSize', 10)
pause;
plot(tsr_1, efficiency_1)
title('v=4.6')
xlabel('Tsr');
ylabel('efficiency');
subplot(3,2,2)
plot(tsr_2,efficiency_2,'rx','MarkerSize', 10)
pause;
plot(tsr_2, efficiency_2)
title('v=6.2')
xlabel('Tsr');
ylabel('efficiency');
subplot(3,2,3)
plot(tsr_3,efficiency_3,'rx','MarkerSize', 10)
pause;
plot(tsr_3, efficiency_3)
title('v=7.4')
xlabel('Tsr');
ylabel('efficiency');
subplot(3,2,4)
plot(tsr_4,efficiency_4,'rx','MarkerSize', 10)
pause;
plot(tsr_4, efficiency_4)
title('v=9.2')
xlabel('Tsr');
ylabel('efficiency');
title('v=9.2')
subplot(3,2,5)
plot(tsr_5,efficiency_5,'rx','MarkerSize', 10)
pause;
plot(tsr_5,efficiency_5)
title('v=3.2')
xlabel('Tsr');
ylabel('efficiency');
title('v=3.2')
pause;
subplot(3,2,5)
plot(tsr_6,efficiency_6,'rx','MarkerSize', 10)
pause;
plot(tsr_6,efficiency_6)
title('v=6.5')
xlabel('Tsr');
ylabel('efficiency');
title('v=6.5')
pause;
%add velocity coloumn to the matrix
data_vel_1=[ones(m_1,1)*vel_1 data_vel_1];%[velocity tsr efficiency]
data_vel_2=[ones(m_2,1)*vel_2 data_vel_2];
data_vel_3=[ones(m_3,1)*vel_3 data_vel_3];
data_vel_4=[ones(m_4,1)*vel_4 data_vel_4];
data_vel_5=[ones(m_5,1)*vel_5 data_vel_5];
data_vel_6=[ones(m_6,1)*vel_6 data_vel_6];
data=[data_vel_1;data_vel_2;data_vel_3;data_vel_4;data_vel_5;data_vel_6];
y=data(:,3);
data(:,3)=data(:,2).^2;
X=data;
% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);
[m n]=size(X);
% Add intercept term to X
X=[ones(m,1) X];
fprintf('Running gradient descent ...\n');
% Choose some alpha value
alpha = 0.4;
num_iters = 800;
theta=zeros(n+1,1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');
theta_grad=theta;
% Estimate the price of a vel=9.2, 0.796 tsr
check_value_tsr=1;
price = 0; % You should change this
d = [vel_4 check_value_tsr (check_value_tsr).^2];
design=d;
d = (d - mu) ./ sigma;
d = [1 d];%now add one more coloumn of 1 to d  matrix 
efficiency = d*theta % now you can multiply d with theta to get a value in the same scale
fprintf(['Predicted vel=9.2, tsr=1' ...
         '(using gradient descent):\n %f\n'], efficiency);
fprintf('actual value at tsr=1 is efficiency=0.229 \n')
fprintf('Program paused. Press enter to continue.\n');
pause;
%solving using normal equations
fprintf('Solving with normal equations...\n');
% Calculate the parameters from the normal equation
theta = normalEqn(X, y);
theta_norm=theta;
% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');
% Estimate the price of a vel=9.2,tsr=1
efficiency = 0; % You should change this
efficiency=[1 vel_4 check_value_tsr (check_value_tsr).^2 ]*theta;
fprintf(['Predicted vel=9.2, tsr=1' ...
         '(using normal equation):\n %f\n'], efficiency);
fprintf('actual value at tsr=1 is efficiency=0.229 \n')
fprintf('Program paused. Press enter to continue.\n');
pause;
figure(3);plot(tsr_4,efficiency_4,'rx','MarkerSize', 10);
title('experimental data')
pause;
clf;
hold on;
plot(tsr_4,efficiency_4,'-b');
title('at v=9.2ms experimental vs prediction')
ylabel('efficiency');
xlabel('tsr');
hold on;
fprintf('working for gradient descent..........\n');
pause;
hold on;
d=[vel_4*ones(m_4,1) tsr_4 (tsr_4).^2]; 
d = (d - mu) ./ sigma;
d = [ones(m_4,1) d];
prediction_gradient_descent=d*theta_grad;
plot(tsr_4,prediction_gradient_descent,'-g');
hold on;
prediction_norm=[ones(m_4,1) vel_4*ones(m_4,1) tsr_4 (tsr_4).^2]*theta_norm;
plot(tsr_4,prediction_norm,'-r');
legend('experimental','gradient descent','normal eqn')

positive_difference_value=sqrt((prediction_gradient_descent-efficiency_4).^2);
percentage_variation=(positive_difference_value./efficiency_4)*100;
percentage_variation(1,:)=1;
[m n]=size(percentage_variation);
percentage_variation(m,:)=1;
variation=sum(percentage_variation)/m; 
accuracy=100-variation;
fprintf('accuracy of the gradient descent algorithm is %f\n:...',accuracy)