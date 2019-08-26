%聚类
x = [0,1,0,1,2,1,2,3,6,7,8,6,7,8,9,7,8,9,8,9;0,0,1,1,1,2,2,2,6,6,6,7,7,7,7,8,8,8,9,9]; 
x = x'; %按行储存数据
num = size(x,1); %数据个数
feature_num = size(x,2); %特征数
x = [x,zeros(num,1)];
%数据预处理
%归一化

% for i = 1:feature_num
%     x(:,i) = (x(:,i)-mean(x(:,i)))./std(x(:,i));
% end

K = 2; %类别数
%判断终止条件
epsilon = 1e-2;
distance = ones(K,1);
%中心点矩阵
center = zeros(K,feature_num);
sum = zeros(K,feature_num);
dist = zeros(K,1);
count = zeros(K,1);
%固定初始中心点
for i = 1:K
    center(i,:) = x(i,1:feature_num); %初始化中心点
end
%随机产生中心点

%new_center记录最新的中心点，center记录上一次迭代的中心点
new_center = center;
%迭代
while max(distance)>epsilon
    center = new_center;
    %计算每一个数据和中心点的欧式距离并分类
    count(:,:) = 0;
    sum(:,:) = 0;
    for i = 1:num
        for j = 1:K
            dist(j) = norm(x(i,1:feature_num)-new_center(j,1:feature_num));
        end
        [~,index] = min(dist);
        x(i,feature_num + 1) = index;
        count(index,1) = count(index,1) + 1;
        sum(index,1:feature_num) = sum(index,1:feature_num) + x(i,1:feature_num);
    end
    new_center = sum./count; %新中心点
    temp = new_center - center;
    for i = 1:K
        distance(i) = norm(temp(i,:));
    end
end
%计算误差(轮廓系数)
sc = zeros();
%1. 簇内距离

for i = 1:num
    d = 0;
    for j = 1:num
        if(x(i,feature_num + 1) == x(j,feature_num + 1))
            d = d + norm(x(i,1:feature_num)-x(j,1:feature_num));
        end
    end
end
%可视化

