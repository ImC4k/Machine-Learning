x = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1]';
train = [0 1 1 0 1 0 0 1];

w1 = 2*rand(1,3)-1;
b1 = 2*rand(1,1)-1;


layer1 = Affine(w1, b1);
layer2 = Sigmoid();

layer3 = MSE();

EPOCH = 10000;
LAMBDA = 0.01;

for epoch = 1:EPOCH
    p = layer1.forward(x);
    y = layer2.forward(p);
    loss(epoch) = layer3.forward(y, train);
    
    dy = layer3.backward();
    dp = layer2.backward(dy);
    dx = layer1.backward(dp);
    
    layer1.update(LAMBDA);
end

figure(1);
plot(loss)
xlabel('Epoch');
ylabel('loss');

layer1.weights

layer1.bias
