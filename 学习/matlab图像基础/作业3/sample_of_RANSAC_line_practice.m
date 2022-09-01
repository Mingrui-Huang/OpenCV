clear
close all
clc

%% 生成60个随机点，然后添加11个点的直线，打乱点的顺序
Points = rand(60,2);  
line = 0:0.1:1;
y = -0.5 * line + 0.8 + (rand(1,11)-0.5)/50;  % try

Points = [Points; cat(1, line, y)'];
scatter(Points(:,1), Points(:,2), 10, 'k', 'filled');
hold on
grid on
daspect([1 1 1]);

Points(:,3) = rand(size(Points,1), 1);
Points = sortrows(Points, 3);

X = Points(:, 1);
Y = Points(:, 2);

%% 尝试1000次
n = 1000; % try
tol = 0.02; % 容差值
for i = 1 : n
    choose = randperm(length(X));  % 所有样本点随机排序
    choose = choose(1:2);   % 随机选取2个样本点
    choose_x = X(choose);
    choose_y = Y(choose);
    
    % 1，根据这2个样本点，生成直线方程（待完成）。。。
    t = polyfit(choose_x, choose_y, 1);
    
    % 2，根据容差值tol，结合直线方程生成容差带，并统计落在容差带内的点的个数（待完成）。。。
    all_distance = abs(t(1)*X-Y+t(2))/sqrt(t(1)^2+(-1)^2);
    choose = all_distance < tol;
    find_t(i,:) = t;
    choose_num(i) = sum(choose);
    choose_point{i} = choose;
   
end

%%3，迭代结束后，找出有效样本点数最多的容差带，并显示输出其对应的有效样本点，以及对应的直线（待完成）。。。
[m,index] = max(choose_num);
t = find_t(index, :);
choose = choose_point{index};
choose_x = X(choose);
choose_y = Y(choose);


plot(choose_x, choose_y, 'b*', choose_x, polyval(t,choose_x), 'r-');
legend('所有点', '内点', '最终拟合直线')
hold on
grid on
daspect([1 1 1]);
