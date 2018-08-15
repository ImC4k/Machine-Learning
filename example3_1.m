x = [0 0; 0 1; 1 0; 1 1]';

labels = [0 1 1 0];

w1 = 2*rand(2,2)-1; %random weight
b1 = 2*rand(2,1)-1; %random bias

w2 = 2*rand(1,2)-1;
b2 = 2*rand(1,1)-1;

layer1 = Affine(w1,b1);
layer2 = Sigmoid();
layer3 = Affine(w2, b2);
layer4 = Sigmoid();
layer5 = MSE();

EPOCH = 1000;
LAMBDA = 0.01;

for epoch = 1:EPOCH
    p = layer1.forward(x);
    y = layer2.forward(p);
    p2 = layer3.forward(y);
    y2 = layer4.forward(p2);
    loss(epoch) = layer5.forward(y2,labels);
    
    dy2 = layer5.backward();
    dp2 = layer4.backward(dy2);
    dy = layer3.backward(dp2);
    dp = layer2.backward(dy);
    dx = layer1.backward(dp);
    
    layer3.update(LAMBDA);
    layer1.update(LAMBDA);
end

figure(1);
plot(loss)
xlabel('Epoch');
ylabel('loss');

layer1.weights

layer1.bias

layer3.weights

layer3.bias