%data generation
data_num=300;
x = 4*rand(2,data_num)-2;
train = x(1,:).*exp(-x(1,:).^2 - x(2,:).^2) + 0.5;


%display input data
figure(1);
scatter3(x(1,:), x(2,:), train, 10);
title('input data');




w1 = 2*rand(3,2)-1;
b1 = 2*rand(3,1)-1;
w2 = 2*rand(1,3)-1;
b2 = 2*rand(1,1)-1;

layer1 = Affine(w1,b1);
layer2 = Sigmoid();
layer3 = Affine(w2,b2);
layer4 = Sigmoid();
layer5 = MSE();

EPOCH = 100000;
LAMBDA = 0.002;
loss = zeros(1,EPOCH);

for epoch = 1:EPOCH
   
    p = layer1.forward(x);
    y = layer2.forward(p);
    p2 = layer3.forward(y);
    y2 = layer4.forward(p2);
    
    loss(epoch) = layer5.forward(y2, train);
    
    dy2 = layer5.backward();
    dp2 = layer4.backward(dy2);
    dy = layer3.backward(dp2);
    dp = layer2.backward(dy);
    dx = layer1.backward(dp);
    
    layer1.update(LAMBDA);
    layer3.update(LAMBDA);
    
end

[X1 X2] = meshgrid(-2:0.1:2);
pt = layer1.forward([X1(:)';X2(:)']);
yt = layer2.forward(pt);
qt = layer3.forward(yt);
zt = layer4.forward(qt);
figure(3);
zsize = sqrt(size(zt));
mesh(X1,X2,reshape(zt,[zsize(2),zsize(2)]));
title('learning results')
