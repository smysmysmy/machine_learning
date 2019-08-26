%����
x = [0,1,0,1,2,1,2,3,6,7,8,6,7,8,9,7,8,9,8,9;0,0,1,1,1,2,2,2,6,6,6,7,7,7,7,8,8,8,9,9]; 
x = x'; %���д�������
num = size(x,1); %���ݸ���
feature_num = size(x,2); %������
x = [x,zeros(num,1)];
%����Ԥ����
%��һ��

% for i = 1:feature_num
%     x(:,i) = (x(:,i)-mean(x(:,i)))./std(x(:,i));
% end

K = 2; %�����
%�ж���ֹ����
epsilon = 1e-2;
distance = ones(K,1);
%���ĵ����
center = zeros(K,feature_num);
sum = zeros(K,feature_num);
dist = zeros(K,1);
count = zeros(K,1);
%�̶���ʼ���ĵ�
for i = 1:K
    center(i,:) = x(i,1:feature_num); %��ʼ�����ĵ�
end
%����������ĵ�

%new_center��¼���µ����ĵ㣬center��¼��һ�ε��������ĵ�
new_center = center;
%����
while max(distance)>epsilon
    center = new_center;
    %����ÿһ�����ݺ����ĵ��ŷʽ���벢����
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
    new_center = sum./count; %�����ĵ�
    temp = new_center - center;
    for i = 1:K
        distance(i) = norm(temp(i,:));
    end
end
%�������(����ϵ��)
sc = zeros();
%1. ���ھ���

for i = 1:num
    d = 0;
    for j = 1:num
        if(x(i,feature_num + 1) == x(j,feature_num + 1))
            d = d + norm(x(i,1:feature_num)-x(j,1:feature_num));
        end
    end
end
%���ӻ�

