x = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1]';
train = [0 1 1 0 1 0 0 1];

w1 = 2*rand(3,3)-1;
b1 = 2*rand(3,1)-1;

w2 = 2*rand(2,3)-1;
b2 = 2*rand(2,1)-1;

w3 = 2*rand(1,2)-1;
b3 = 2*rand(1,1)-1;

layer1 = Affine(w1,b1);
layer2 = Sigmoid();
layer3 = Affine(w2, b2);
layer4 = Sigmoid();
layer5 = Affine(w3,b3);
layer6 = Sigmoid();
layer7 = MSE();

EPOCH = 1000000;
LAMBDA = 0.3;

for epoch = 1:EPOCH

    p = layer1.forward(x);
    y = layer2.forward(p);
    p2 = layer3.forward(y);
    y2 = layer4.forward(p2);
    p3 = layer5.forward(y2);
    y3 = layer6.forward(p3);
    loss(epoch) = layer7.forward(y3, train);

    dy3 = layer7.backward();
    dp3 = layer6.backward(dy3);
    dy2 = layer5.backward(dp3);
    dp2 = layer4.backward(dy2);
    dy = layer3.backward(dp2);
    dp = layer2.backward(dy);
    dx = layer1.backward(dp);

    layer1.update(LAMBDA);
    layer3.update(LAMBDA);
    layer5.update(LAMBDA);

end

figure(1);
plot(loss)
xlabel('Epoch');
ylabel('loss');

y3