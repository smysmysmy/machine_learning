%PCA
x = [1,2,3,4;4,7,6,10;7,8,9,11]; %һ��Ϊһ������

num = size(x,2); %�ɷָ���
p = 0.9; %Ҫ������ɷ���Ϣ������

%���ݱ�׼��
for i = 1:num
    x(:,i) = (x(:,i)-mean(x(:,i)))./std(x(:,i));
end
% %�˾���
% x = K2(x);
%���ϵ�����������ֵ����������
MA = corrcoef(x);
[V,D] = eig(MA);
lamta = zeros(num,2);
for i = 1:num
    lamta(i,1) = D(num+1-i,num+1-i); %����ֵ��������
end
for i = 1:num
    lamta(i,2) = lamta(i,1)/sum(lamta(:,1)); %���ɷ���Ϣ������
end
lamta(:,2) = cumsum(lamta(:,2)); %�ۼ����ɷ���Ϣ������
%������ɷָ�������������
[s,~] = find(lamta(:,2)>=p,1);
Feature_Vector = zeros(num,s);
for i = 1:s
    Feature_Vector(:,i) = V(:,num+1-i); 
end
score = x * Feature_Vector; %��ά�������
% %KPCA��ά�������
% score = sqrt(lamta(1:s,2))' * x * Feature_Vector;


disp('���ɷ���Ϣ������Ϊ')
disp(lamta(s,2))
disp('���ɷָ���Ϊ')
disp(s)




%�˺��� ֱ���ڻ�
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
%RBF�˺���
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