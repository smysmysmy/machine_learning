X=[0,0;1,0;0,1;1,1;2,0;3,0;2,1;3,1;0,4;1,4;0,5;1,5;2,4;3,4;2,5;3,5];
y=[0;0;0;0;1;1;1;1;2;2;2;2;3;3;3;3];
t=templateSVM('KernelFunction','gaussian');
model = fitcecoc(X,y,'Learners',t);
visualizeBoundary(X,y,model)