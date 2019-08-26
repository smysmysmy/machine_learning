%PCA
x = [1,2,3,4;4,7,6,10;7,8,9,11]; %一行为一组数据

num = size(x,2); %成分个数
p = 0.9; %要求的主成分信息保留率

%数据标准化
for i = 1:num
    x(:,i) = (x(:,i)-mean(x(:,i)))./std(x(:,i));
end
% %核矩阵
% x = K2(x);
%相关系数矩阵的特征值和特征向量
MA = corrcoef(x);
[V,D] = eig(MA);
lamta = zeros(num,2);
for i = 1:num
    lamta(i,1) = D(num+1-i,num+1-i); %特征值降序排序
end
for i = 1:num
    lamta(i,2) = lamta(i,1)/sum(lamta(:,1)); %主成分信息贡献率
end
lamta(:,2) = cumsum(lamta(:,2)); %累计主成分信息贡献率
%输出主成分个数和特征向量
[s,~] = find(lamta(:,2)>=p,1);
Feature_Vector = zeros(num,s);
for i = 1:s
    Feature_Vector(:,i) = V(:,num+1-i); 
end
score = x * Feature_Vector; %降维后的数据
% %KPCA降维后的数据
% score = sqrt(lamta(1:s,2))' * x * Feature_Vector;


disp('主成分信息保留率为')
disp(lamta(s,2))
disp('主成分个数为')
disp(s)




%核函数 直接内积
function K = K1(x)
K = x' * x;
% num = size(x,2);
% K = zeros(num);
% for i = 1:num
%     for j = 1:i
%         K(i,j) = x(:,i)'*x(:,j);
%         K(j,i) = K(i,j);
%     end
% end
end
%RBF核函数
function K = K2(x,gamma)
num = size(x,2);
K = zeros(num);
for i = 1:num
    for j = 1:i
        K(i,j) = exp(-norm(x(:,i)-x(:,j))^2/(2*gamma^2));
        K(j,i) = K(i,j);
    end
end
end